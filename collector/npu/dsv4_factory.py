"""DeepSeek-V4 (DSv4) attention module factory for Ascend NPU benchmarking.

Two module sources are supported:

``framework`` (default)
    Instantiate vllm-ascend's own DeepSeek-V4 attention module. The class is
    discovered by probing a list of candidate import paths and by matching a
    class-name regex, so it does not hard-code a single vllm-ascend release's
    internal layout. ``--attention-cls module.path:ClassName`` overrides the
    discovery when the layout is known.

``reference``
    A self-contained torch implementation of the documented DSv4 op graph
    (q_a/q_b/kv_a projections -> compressor -> optional indexer topk -> sparse
    attention over sliding window + compressed KV -> two-stage grouped low-rank
    o_proj). Useful when the framework module is unavailable or unstable; the
    output schema is identical.

Op graph (mirrors aiconfigurator's ``operators/dsv4.rs`` weight formula)::

    q_a_proj   [T, h]        x [h, q_lora]
    q_b_proj   [T, q_lora]   x [q_lora, H * (head_dim + qk_rope)]
    kv_a_proj  [T, h]        x [h, head_dim]                (MQA: 1 KV head)
    compressor [T, h]        x [h, head_dim]  (x2 for CSA, x1 for HCA)
    indexer    (CSA only)    wq_b / weights_proj / k proj / FP8 MQA logits / topk
    attention  QK over attn_dim = head_dim + qk_rope ; PV over head_dim
    o_proj A   [T, H*head_dim] x [H*head_dim, o_lora]   (bf16)
    o_proj B   grouped [T, o_lora] x [o_lora, h]  (o_groups groups)
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Attention kinds ──────────────────────────────────────────────────────
CSA = "csa"  # compress_ratio == 4   (compressed sparse attention + indexer)
HCA = "hca"  # compress_ratio 0/128  (sliding window / highly-compressed)

OP_CONTEXT = "dsv4_context"
OP_GENERATION = "dsv4_generation"
SUPPORTED_OP_TYPES = (OP_CONTEXT, OP_GENERATION)

# Candidate modules that may hold the DeepSeek-V4 attention implementation,
# probed in order. vllm-ascend / vLLM move these around between releases.
_CANDIDATE_MODULES = (
    "vllm.model_executor.models.deepseek_v4",
    "vllm.model_executor.models.deepseekv4",
    "vllm_ascend.models.deepseek_v4",
    "vllm_ascend.models.deepseekv4",
    "vllm_ascend.attention.dsv4_v1",
)
_CLASS_NAME_HINTS = ("DeepseekV4", "DeepSeekV4", "DSV4", "Dsv4")

# DeepSeek-V4-Pro defaults (model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json)
PRO_DEFAULTS = dict(
    hidden_size=7168,
    num_heads=128,
    q_lora_rank=1536,
    o_lora_rank=1024,
    o_groups=16,
    head_dim=512,
    qk_rope_head_dim=64,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=1024,
    sliding_window=128,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_MODEL_CONFIGS_DIR = _PROJECT_ROOT / "model_configs"


@dataclass(frozen=True)
class Dsv4ModuleSpec:
    """Immutable DSv4 attention benchmark specification."""

    op_type: str  # OP_CONTEXT or OP_GENERATION
    attn_kind: str  # CSA or HCA
    compress_ratio: int  # 4 for CSA, 128 for HCA (SWA folds into HCA)
    batch: int
    seq_len: int  # prefill seq len (context) or KV cache len (generation)
    num_heads: int  # rank-local heads = 128 // tp_size
    hidden_size: int = PRO_DEFAULTS["hidden_size"]
    q_lora_rank: int = PRO_DEFAULTS["q_lora_rank"]
    o_lora_rank: int = PRO_DEFAULTS["o_lora_rank"]
    o_groups: int = PRO_DEFAULTS["o_groups"]
    head_dim: int = PRO_DEFAULTS["head_dim"]
    qk_rope_head_dim: int = PRO_DEFAULTS["qk_rope_head_dim"]
    index_n_heads: int = PRO_DEFAULTS["index_n_heads"]
    index_head_dim: int = PRO_DEFAULTS["index_head_dim"]
    index_topk: int = PRO_DEFAULTS["index_topk"]
    sliding_window: int = PRO_DEFAULTS["sliding_window"]
    dtype: torch.dtype = torch.bfloat16
    module_source: str = "framework"  # "framework" or "reference"
    attention_cls: str | None = None  # "module.path:ClassName" override
    model_path: str = "deepseek-ai/DeepSeek-V4-Pro"


def _spec_key(spec: Dsv4ModuleSpec) -> str:
    return (
        f"{spec.op_type}_{spec.attn_kind}_{spec.batch}_{spec.seq_len}_{spec.num_heads}"
        f"_{spec.hidden_size}_{spec.head_dim}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reference DSv4 module
# ═══════════════════════════════════════════════════════════════════════════


class ReferenceDeepSeekV4Attention(nn.Module):
    """Self-contained torch implementation of one DeepSeek-V4 attention layer.

    Not a bit-exact reproduction of vllm-ascend's kernel sequence -- it implements
    the same *op graph* with the same shapes so the benchmark measures a
    representative DSv4 attention cost.
    """

    def __init__(self, spec: Dsv4ModuleSpec) -> None:
        super().__init__()
        self.spec = spec
        h = spec.hidden_size
        hd = spec.head_dim
        rope = spec.qk_rope_head_dim

        self.w_qa = nn.Parameter(torch.randn(h, spec.q_lora_rank))
        self.w_qb = nn.Parameter(torch.randn(spec.q_lora_rank, spec.num_heads * (hd + rope)))
        self.w_kva = nn.Parameter(torch.randn(h, hd))
        self.w_kvr = nn.Parameter(torch.randn(h, rope))
        # compressor: CSA runs two stages, HCA one
        self.n_comp = 2 if spec.compress_ratio == 4 else (1 if spec.compress_ratio else 0)
        self.w_comp = nn.ParameterList(nn.Parameter(torch.randn(h, hd)) for _ in range(max(self.n_comp, 1)))
        # grouped low-rank output projection
        self.w_o1 = nn.Parameter(torch.randn(spec.num_heads * hd, spec.o_lora_rank))
        self.w_o2 = nn.ParameterList(
            nn.Parameter(torch.randn(spec.o_lora_rank, h)) for _ in range(spec.o_groups)
        )
        # indexer (CSA only)
        self.has_indexer = spec.compress_ratio == 4
        if self.has_indexer:
            self.w_wqb = nn.Parameter(torch.randn(spec.q_lora_rank, spec.index_n_heads * spec.index_head_dim))
            self.w_weights = nn.Parameter(torch.randn(h, spec.index_n_heads))
            self.w_kidx = nn.Parameter(torch.randn(h, spec.index_head_dim))

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``ratio`` consecutive tokens into one compressed latent."""
        ratio = self.spec.compress_ratio
        if not ratio:
            return x[:0]
        n = x.shape[0] // ratio
        if n == 0:
            return x[:0]
        return x[: n * ratio].view(n, ratio, -1).mean(dim=1)

    def forward(self, hidden: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Run one attention step.

        Args:
            hidden: query hidden states ``[T, h]`` (T = batch*seq for context,
                batch for generation).
            kv: KV source hidden states ``[S, h]`` from which the compressed /
                window / indexer caches are derived.
        """
        spec = self.spec
        hd = spec.head_dim
        rope = spec.qk_rope_head_dim
        t = hidden.shape[0]

        # ── projections ────────────────────────────────────────────────
        q_lora = hidden @ self.w_qa
        q = (q_lora @ self.w_qb).view(t, spec.num_heads, hd + rope)
        q_nope = q[:, :, :hd]
        q_rope = q[:, :, hd:]

        k_latent = kv @ self.w_kva  # [S, hd]  (MQA: single KV head)
        k_rope = kv @ self.w_kvr  # [S, rope]

        # ── compressor ─────────────────────────────────────────────────
        k_comp = k_latent
        kr_comp = k_rope
        for w in self.w_comp[: max(self.n_comp, 1)]:
            if self.n_comp == 0:
                break
            k_comp = self._compress(k_comp @ w)
            kr_comp = self._compress(kr_comp)
        if self.n_comp == 0:
            k_comp = k_latent[:0]
            kr_comp = k_rope[:0]

        # ── sliding window ─────────────────────────────────────────────
        w = min(k_latent.shape[0], spec.sliding_window)
        k_win = k_latent[-w:] if w else k_latent[:0]
        kr_win = k_rope[-w:] if w else k_rope[:0]

        keys = torch.cat([k_win, k_comp], dim=0)  # [K, hd]
        keys_rope = torch.cat([kr_win, kr_comp], dim=0)  # [K, rope]

        # ── CSA indexer: topk over the compressed KV ───────────────────
        if self.has_indexer and k_comp.shape[0] > spec.index_topk:
            idx_keys = k_comp @ self.w_kidx  # [Kc, index_head_dim]
            q_idx = (q_lora @ self.w_wqb).view(t, spec.index_n_heads, spec.index_head_dim)
            gate = (hidden @ self.w_weights).softmax(dim=-1)  # [T, index_n_heads]
            logits = torch.einsum("tid,kd->tik", q_idx, idx_keys)  # [T, iH, Kc]
            logits = (logits * gate.unsqueeze(-1)).sum(dim=1)  # [T, Kc]
            topk = min(spec.index_topk, logits.shape[-1])
            sel = logits.topk(topk, dim=-1).indices  # [T, topk]
            sel = sel + (keys.shape[0] - idx_keys.shape[0])  # offset past the window
            # Merge window + selected entries per query.
            win_idx = torch.arange(keys.shape[0] - idx_keys.shape[0], device=keys.device)
            win_idx = win_idx.unsqueeze(0).expand(t, -1)
            gather_idx = torch.cat([win_idx, sel], dim=1)  # [T, W+topk]
            keys = keys[gather_idx]  # [T, W+topk, hd]
            keys_rope = keys_rope[gather_idx]  # [T, W+topk, rope]
            scores = torch.einsum("thd,tkd->thk", q_nope, keys)
            scores = scores + torch.einsum("thr,tkr->thk", q_rope, keys_rope)
            scores = scores / ((hd + rope) ** 0.5)
            probs = scores.softmax(dim=-1)
            attn = torch.einsum("thk,tkd->thd", probs, keys)
        else:
            scores = torch.einsum("thd,kd->thk", q_nope, keys)
            scores = scores + torch.einsum("thr,kr->thk", q_rope, keys_rope)
            scores = scores / ((hd + rope) ** 0.5)
            probs = scores.softmax(dim=-1)
            attn = torch.einsum("thk,kd->thd", probs, keys)

        # ── two-stage grouped low-rank o_proj ──────────────────────────
        out = attn.reshape(t, spec.num_heads * hd) @ self.w_o1  # [T, o_lora]
        total = 0.0
        for w2 in self.w_o2:
            total = total + out @ w2
        return total


# ═══════════════════════════════════════════════════════════════════════════
# Framework module discovery
# ═══════════════════════════════════════════════════════════════════════════


def _import_class(dotted: str):
    module_path, _, cls_name = dotted.partition(":")
    if not module_path or not cls_name:
        raise ValueError(f"--attention-cls must look like 'module.path:ClassName', got {dotted!r}")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _discover_attention_cls():
    """Probe candidate modules for a DeepSeek-V4 attention class."""
    for module_path in _CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        for name in dir(module):
            if not any(hint in name for hint in _CLASS_NAME_HINTS):
                continue
            if "Attention" not in name and "attention" not in name:
                continue
            obj = getattr(module, name)
            if inspect.isclass(obj):
                logger.info("Discovered DSv4 attention class %s.%s", module_path, name)
                return obj
    return None


def _build_framework_module(spec: Dsv4ModuleSpec, device: str):
    """Instantiate vllm-ascend's DSv4 attention module.

    Reuses the VllmConfig plumbing from ``mla_module_factory`` so the ascend
    platform hooks (custom-op registration, ATB warmup, workspace manager) run
    the same way they do for the DSA/MLA collectors.
    """
    from mla_module_factory import _create_npu_vllm_config, _resolve_model_path
    from vllm.config import set_current_vllm_config

    cls = _import_class(spec.attention_cls) if spec.attention_cls else _discover_attention_cls()
    if cls is None:
        raise RuntimeError(
            "Could not locate a DeepSeek-V4 attention class in any candidate module "
            f"{_CANDIDATE_MODULES}. Pass --attention-cls module.path:ClassName or use "
            "--module-source reference."
        )

    local_model_path = _resolve_model_path(spec.model_path)
    is_context = spec.op_type == OP_CONTEXT
    max_tokens = spec.batch * spec.seq_len if is_context else spec.batch

    vllm_config = _create_npu_vllm_config(
        model_name=local_model_path,
        max_seq_len=spec.seq_len,
        max_batch_size=spec.batch,
        is_context=is_context,
        device=device,
    )
    hf_config = vllm_config.model_config.hf_config

    # Only forward the kwargs the class actually declares, so this works across
    # vllm-ascend releases with different constructor signatures.
    candidates = dict(
        vllm_config=vllm_config,
        config=hf_config,
        hidden_size=spec.hidden_size,
        num_heads=spec.num_heads,
        q_lora_rank=spec.q_lora_rank,
        head_dim=spec.head_dim,
        qk_rope_head_dim=spec.qk_rope_head_dim,
        index_n_heads=spec.index_n_heads,
        index_head_dim=spec.index_head_dim,
        index_topk=spec.index_topk,
        sliding_window=spec.sliding_window,
        max_position_embeddings=getattr(hf_config, "max_position_embeddings", 1048576),
        cache_config=vllm_config.cache_config,
        quant_config=vllm_config.quant_config,
        prefix="model.layers.0.self_attn",
    )
    params = inspect.signature(cls.__init__).parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = candidates if accepts_var_kw else {k: v for k, v in candidates.items() if k in params}
    logger.debug("Instantiating %s with kwargs %s", cls.__name__, sorted(kwargs))

    with set_current_vllm_config(vllm_config):
        module = cls(**kwargs)

    if any(p.is_meta for p in module.parameters()):
        module = module.to_empty(device=torch.device(device))
    else:
        module = module.to(device)
    module.eval()
    module.requires_grad_(False)
    return module, vllm_config, max_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def create_dsv4_module_func(
    spec: Dsv4ModuleSpec,
    device: str = "npu:0",
) -> tuple[Callable[[], None], dict]:
    """Build and return ``(forward_fn, meta)`` for one benchmark point.

    ``meta`` carries ``num_heads`` / ``attn_kind`` / ``compress_ratio`` so the
    collector can stamp them onto the output rows.
    """
    if spec.op_type not in SUPPORTED_OP_TYPES:
        raise ValueError(f"Unsupported op_type: {spec.op_type!r}")
    is_context = spec.op_type == OP_CONTEXT

    if spec.module_source == "reference":
        module = ReferenceDeepSeekV4Attention(spec).to(device=device, dtype=spec.dtype)
        module.eval()
        module.requires_grad_(False)

        num_tokens = spec.batch * spec.seq_len if is_context else spec.batch
        kv_len = spec.seq_len
        hidden = torch.randn(num_tokens, spec.hidden_size, dtype=spec.dtype, device=device)
        kv = torch.randn(kv_len, spec.hidden_size, dtype=spec.dtype, device=device)

        def forward_fn() -> None:
            module(hidden, kv)

    else:
        module, vllm_config, num_tokens = _build_framework_module(spec, device)
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            hidden = torch.randn(
                num_tokens, spec.hidden_size, dtype=spec.dtype, device=device
            )
            if is_context:
                positions = (
                    torch.arange(spec.seq_len, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(spec.batch, -1)
                    .reshape(-1)
                    .contiguous()
                )
            else:
                positions = torch.full((spec.batch,), spec.seq_len - 1, dtype=torch.long, device=device)

            def forward_fn() -> None:
                # vLLM attention modules take (positions, hidden_states); fall back
                # to hidden-only for modules with a different signature.
                try:
                    module.forward(positions, hidden, None)
                except TypeError:
                    module.forward(positions, hidden)

    # Dry run so construction/typing failures surface before the timed loop.
    with torch.inference_mode():
        forward_fn()
    torch.npu.synchronize()

    meta = {
        "num_heads": spec.num_heads,
        "attn_kind": spec.attn_kind,
        "compress_ratio": spec.compress_ratio,
    }
    return forward_fn, meta
