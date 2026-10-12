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


_DISTRIBUTED_READY = False
_TP_VLLM_CONFIG_CTX = None


def _ensure_distributed_env(vllm_config) -> None:
    """Initialise vllm's single-rank distributed world + TP groups (tp=1).

    DeepseekV4Attention.__init__ calls get_tensor_model_parallel_world_size(),
    which asserts "tensor model parallel group is not initialized" unless
    vllm's parallel state is up. Mirrors gemm_factory._init_vllm_context():
    RANK/WORLD_SIZE/MASTER env vars, init_distributed_environment(),
    ensure_model_parallel_initialized(1, 1). Idempotent per process.
    """
    global _DISTRIBUTED_READY, _TP_VLLM_CONFIG_CTX
    if _DISTRIBUTED_READY:
        return
    import os
    import socket

    from vllm.distributed import init_distributed_environment
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

    # Enable NZ (FRACTAL_NZ) format for weight tensors. In production vLLM
    # this is set by vllm_ascend/worker/model_runner_v1.py at import time;
    # benchmark scripts bypass the worker layer.
    if hasattr(torch, "npu"):
        torch.npu.config.allow_internal_format = True

    init_distributed_environment()

    # vllm's initialize_model_parallel reads get_current_vllm_config().
    # set_current_vllm_config is a @contextmanager: calling it without
    # entering never installs the config, so enter it and keep it installed
    # for the whole run (same trick as gemm_factory._ensure_vllm_config_ctx).
    if _TP_VLLM_CONFIG_CTX is None:
        from vllm.config import set_current_vllm_config

        _TP_VLLM_CONFIG_CTX = set_current_vllm_config(vllm_config)
        _TP_VLLM_CONFIG_CTX.__enter__()

    ensure_model_parallel_initialized(1, 1)
    _DISTRIBUTED_READY = True


def _reset_dsv4_rope_runtime_buffers() -> None:
    """Drop the process-global DSA RoPE cos/sin runtime buffers.

    vllm_ascend.ops.rope_dsv4 allocates the per-token cos/sin buffers once per
    (config_key, rope_group), sized by the *first* vllm_config that constructs
    a ComplexExpRotaryEmbedding, and never resizes them afterwards. In a sweep
    the first spec may be batch=1 (buffer [1, 1, 1, rotary_dim]); every later
    larger spec then fails in get_cos_and_sin_dsa with

        "shape of broadcast result should be [N, 1, 1, 64], but self is [1, 1, 1, 64]"

    and silently falls back to the profiling path. Clearing the buffers lets
    __init__ reallocate them against the current spec's max_num_batched_tokens.
    The precomputed position tables (full_rope_cache) are position-independent
    and are kept.
    """
    try:
        from vllm_ascend.ops.rope_dsv4 import _ROPE_STATE
    except ImportError:
        return
    _ROPE_STATE.runtime_buffer.clear()
    _ROPE_STATE.spec_runtime_buffer.clear()


def _reshape_dsv4_woa_weight_3d(module) -> None:
    """Give ``wo_a`` the 3-D grouped layout the DSA o_proj bmm requires.

    ``AscendDSAImpl._forward_o_proj`` (single-card ``else`` branch) feeds
    ``npu_transpose_batchmatmul(x1=[T, groups, gh], wo_a.weight,
    perm_x2=(0, 1, 2))``; the op requires a 3-D x2 of shape
    ``[n_local_groups, group_hidden_dim, o_lora_rank]``. vLLM's
    ``ColumnParallelLinear`` creates a 2-D ``[groups * o_lora, gh]``
    parameter, and the real DSv4-Pro deployment always runs finegrained
    o_lora TP (``olora_tp_enable()`` -> the ``wo_a(...)`` module-call
    branch), so the plain branch is never exercised in production and the
    2-D weight never reaches this op there. The collector benchmarks TP=1,
    so reshape here: dispatch only depends on shapes (weights are never
    loaded from a checkpoint), and the grouped bmm has the same FLOPs /
    memory traffic as the two 2-D GEMMs the SDK models.
    """
    wrapper = getattr(module, "dsa_attn", None)
    if wrapper is None:
        return
    wo_a = getattr(wrapper, "wo_a", None)
    if wo_a is None:
        return
    weight = getattr(wo_a, "weight", None)
    if weight is None or weight.dim() == 3:
        return
    groups = getattr(wrapper, "n_local_groups", None)
    o_lora = getattr(wrapper, "o_lora_rank", None)
    gh = weight.shape[1]
    if not groups or not o_lora or weight.shape[0] != groups * o_lora:
        logger.warning(
            "wo_a.weight shape %s incompatible with grouped reshape "
            "(groups=%s, o_lora=%s); leaving as-is",
            tuple(weight.shape), groups, o_lora,
        )
        return
    # [groups * o_lora, gh] -> [groups, gh, o_lora] (contiguous for the op).
    wo_a.weight.data = weight.view(groups, o_lora, gh).transpose(1, 2).contiguous()
    logger.info(
        "Reshaped wo_a.weight %s -> %s for npu_transpose_batchmatmul",
        tuple(weight.shape), tuple(wo_a.weight.shape),
    )


def _build_framework_module(spec: Dsv4ModuleSpec, device: str):
    """Instantiate vllm-ascend's DSv4 attention module.

    Reuses the VllmConfig plumbing from ``mla_module_factory`` so the ascend
    platform hooks (custom-op registration, ATB warmup, workspace manager) run
    the same way they do for the DSA/MLA collectors.
    """
    from mla_module_factory import (
        _init_triton_device_props,
        _prepare_npu_vllm_config,
        _setup_rope_caches,
    )
    from vllm.config import set_current_vllm_config

    cls = _import_class(spec.attention_cls) if spec.attention_cls else _discover_attention_cls()
    if cls is None:
        raise RuntimeError(
            "Could not locate a DeepSeek-V4 attention class in any candidate module "
            f"{_CANDIDATE_MODULES}. Pass --attention-cls module.path:ClassName or use "
            "--module-source reference."
        )

    is_context = spec.op_type == OP_CONTEXT
    max_tokens = spec.batch * spec.seq_len if is_context else spec.batch

    vllm_config = _prepare_npu_vllm_config(
        model_path=spec.model_path,
        max_seq_len=spec.seq_len,
        max_batch_size=spec.batch,
        is_context=is_context,
        device=device,
    )
    _ensure_distributed_env(vllm_config)
    hf_config = vllm_config.model_config.hf_config

    # DeepseekV4Attention reads compress_ratios[config_layer_idx] from hf_config
    # (vllm_ascend.utils.get_dsv4_compress_ratio) and ignores any constructor
    # kwarg, so the sweep's CSA(4)/HCA(128) choice must be written into the
    # config *before* the module is constructed. Without this every spec runs
    # the config's layer-0 ratio (128 for V4-Pro) and the CSA/HCA output columns
    # would silently measure the same thing.
    from vllm_ascend.utils import extract_dsv4_layer_index

    ratios = list(getattr(hf_config, "compress_ratios", None) or [])
    if ratios:
        layer_idx = extract_dsv4_layer_index(hf_config, "model.layers.0.self_attn")
        if 0 <= layer_idx < len(ratios):
            ratios[layer_idx] = spec.compress_ratio
            hf_config.compress_ratios = ratios
            logger.debug(
                "Forced hf_config.compress_ratios[%d]=%d (%s)",
                layer_idx, spec.compress_ratio, spec.attn_kind,
            )

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

    # Construct under the NPU default-device context, mirroring production
    # where the worker builds the model on-device. vllm-ascend's
    # ComplexExpRotaryEmbedding (rope_dsv4.py) mixes bare torch.arange (t)
    # with an explicitly-placed inv_freq in __init__ — under a CPU default
    # device the einsum crashes with "Expected all tensors to be on the same
    # device". bf16 default dtype matches the MLA collector path.
    from mla_module_factory import _set_default_dtype_ctx

    _reset_dsv4_rope_runtime_buffers()
    with set_current_vllm_config(vllm_config), torch.device(device), _set_default_dtype_ctx(spec.dtype):
        module = cls(**kwargs)

    if any(p.is_meta for p in module.parameters()):
        module = module.to_empty(device=torch.device(device))
    else:
        module = module.to(device)
    module.eval()
    module.requires_grad_(False)

    _reshape_dsv4_woa_weight_3d(module)

    # Same post-instantiation env setup as the MLA collector: rope cos/sin
    # caches live in module-level globals created on CPU, and the triton
    # device-property counters are normally filled by the vllm-ascend worker.
    _setup_rope_caches(vllm_config, device)
    _init_triton_device_props()

    return module, vllm_config, max_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def _call_get_kv_cache_spec(child, vllm_config):
    """Invoke a layer's get_kv_cache_spec, passing vllm_config when required.

    All vllm-ascend DSv4 cache layers declare
    ``get_kv_cache_spec(self, vllm_config)`` — calling without the argument
    raises TypeError, which must not be mistaken for "spec not available".
    """
    get_spec = getattr(child, "get_kv_cache_spec", None)
    if not callable(get_spec):
        return None
    try:
        n_params = len(inspect.signature(get_spec).parameters)
    except (TypeError, ValueError):
        n_params = 1
    try:
        return get_spec(vllm_config) if n_params >= 1 else get_spec()
    except Exception as e:
        logger.debug(
            "get_kv_cache_spec on %s failed: %s", type(child).__name__, e
        )
        return None


def _collect_dsv4_kv_layers(module, vllm_config, attn_layer_name):
    """Find sub-layers of the DSv4 module that expose a kv_cache_spec.

    ``ops/dsa.py::_build_kv_cache`` reads ``.kv_cache`` off these sub-layers
    (dsa_attn / swa_cache_layer / compressor.state_cache / indexer.*), so they
    are exactly the layers that need an allocated cache buffer.

    collected via named_modules() first (deduplicated by identity), then the
    explicit attribute paths _build_kv_cache indexes are probed as well, in
    case a build nests those modules differently.
    """
    layers = []
    seen: set[int] = set()

    def _try_add(full_name: str, child) -> None:
        if child is None or id(child) in seen:
            return
        # NOTE: do NOT gate on hasattr(child, "kv_cache") — vLLM assigns
        # .kv_cache externally (bind_kv_cache), so the attribute does not
        # exist yet on freshly constructed layers.
        kv_spec = _call_get_kv_cache_spec(child, vllm_config)
        if kv_spec is None:
            return
        seen.add(id(child))
        layers.append((full_name, child, kv_spec))

    for rel_name, child in module.named_modules():
        if not rel_name:
            continue
        _try_add(f"{attn_layer_name}.{rel_name}", child)

    # Explicit contract of ops/dsa.py::_build_kv_cache.
    for rel_path in (
        "compressor.state_cache",
        "dsa_attn.swa_cache_layer",
        "dsa_attn.dsa_attn",
        "swa_cache_layer",
        "dsa_attn",
        "indexer.compressor.state_cache",
        "indexer.k_cache",
    ):
        child = module
        for part in rel_path.split("."):
            child = getattr(child, part, None)
            if child is None:
                break
        if child is not None:
            _try_add(f"{attn_layer_name}.{rel_path}", child)

    return layers


def _alloc_dsv4_layer_cache(kv_spec, num_blocks, device):
    """Allocate one sub-layer's cache buffer, mirroring model_runner_v1's
    ``use_compress`` branch of _reshape_kv_cache_tensors (l.4382-4435).

    A spec with ``scale_dim != 0`` (the indexer k_cache) gets a ``(k, scale)``
    tuple so that ``_build_kv_cache``'s ``kv_cache[0][0] / kv_cache[0][1]``
    indexing works; every other layer gets a single 4-D tensor. The DSv4
    binding wraps each buffer in a 1-element list (model_runner_v1 l.3937-3948),
    which ``unfold_kvcache`` unwraps.
    """
    from vllm_ascend.attention.dsa_v1 import AscendDSABackend

    shape = AscendDSABackend.get_kv_cache_shape(
        num_blocks, kv_spec.block_size, kv_spec.num_kv_heads, kv_spec.head_size
    )
    scale_dim = getattr(kv_spec, "scale_dim", 0)
    if scale_dim:
        scale_shape = AscendDSABackend.get_kv_cache_shape(
            num_blocks, kv_spec.block_size, kv_spec.num_kv_heads, scale_dim
        )
        k = torch.zeros(shape, dtype=kv_spec.dtype, device=device)
        scale = torch.zeros(scale_shape, dtype=kv_spec.scale_dtype, device=device)
        return (k, scale)
    return torch.zeros(shape, dtype=kv_spec.dtype, device=device)


def _create_dsv4_kv_cache_and_metadata(
    module, vllm_config, spec, device, attn_layer_name
):
    """Build the DSv4 attention metadata (+ KV cache) for one forward pass.

    Replicates the per-KV-cache-group assembly of
    ``vllm_ascend/worker/model_runner_v1.py::_build_attention_metadata``:

      1. collect every sub-layer with ``get_kv_cache_spec()`` and group them by
         cache layout (the DSA groups have different block sizes: 64 / 4 / 16),
      2. allocate each sub-layer's ``.kv_cache`` buffer (the 6-tuple source
         that ``ops/dsa.py::_build_kv_cache`` reads),
      3. build one ``AscendCommonAttentionMetadata`` per group (per-group block
         table + slot mapping, matching the group's block size),
      4. run ``AscendDSABackend.get_builder_cls()``'s builder per group with
         the three shared ``ratio_to_sas_metadata`` dicts (cached across
         groups exactly like the model runner does),
      5. register each group's metadata under its full layer names and the
         main ``attn_layer_name`` that dsa_forward looks up.

    Returns None (profiling fallback) if anything in this version-specific
    chain fails; the traceback is logged so the gap is visible.
    """
    try:
        return _build_dsv4_real_metadata(module, vllm_config, spec, device, attn_layer_name)
    except Exception:
        logger.exception(
            "DSv4 real attn_metadata assembly failed; using dsa_forward profiling path"
        )
        return None


# named_modules path suffix -> construction-prefix suffix. Checked in order;
# the indexer entries must precede the bare ".compressor.state_cache" because
# they share that suffix.
_DSV4_KEY_SUFFIX_MAP = (
    (".indexer.compressor.state_cache", ".indexer.compressor.state_cache"),
    (".indexer.k_cache", ".indexer.k_cache"),
    (".compressor.state_cache", ".compressor.state_cache"),
    (".dsa_attn.swa_cache_layer", ".swa_cache"),
    (".dsa_attn.dsa_attn", ".attn"),
)


def _dsv4_metadata_key(child, full_name: str, attn_layer_name: str) -> str:
    """Forward-context attn_metadata dict key for one DSv4 kv sub-layer.

    ``vllm_ascend/ops/dsa.py::dsa_forward`` resolves the running layer, then
    ``filter_metadata`` keeps the dict entries whose key starts with the
    attention prefix and hands the *values* to
    ``AscendDSAImpl._forward_prefill``, which unpacks them positionally by
    ``compress_ratio``:

        ratio 4   -> [attn, compressor.state_cache, indexer.compressor.state_cache,
                      indexer.k_cache, swa_cache]
        ratio 128 -> [attn, compressor.state_cache, swa_cache]

    The keys that produce this order are the cache modules' *own*
    construction prefixes (``.../self_attn.attn``,
    ``.../self_attn.compressor.state_cache``, ``.../self_attn.indexer.*``,
    ``.../self_attn.swa_cache``) — NOT their ``named_modules`` paths, which
    would insert a spurious ``.../self_attn.dsa_attn.dsa_attn`` entry and
    break the unpack ("too many values to unpack (expected 3)").
    """
    for attr in ("layer_name", "prefix", "layername"):
        value = getattr(child, attr, None)
        if isinstance(value, str) and value:
            return value
    for path_suffix, key_suffix in _DSV4_KEY_SUFFIX_MAP:
        if full_name.endswith(path_suffix):
            return attn_layer_name + key_suffix
    return full_name


def _build_dsv4_real_metadata(module, vllm_config, spec, device, attn_layer_name):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.dsa_v1 import AscendDSABackend
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

    # The DSA metadata builders call vllm-ascend custom ops
    # (torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata et al.) whose aclnn
    # implementations live in the vendor custom-op package (libcust_opapi.so).
    # torch_npu's op loader (FunctionLoader) only resolves those symbols when
    # the vendor op_api/lib dir is on the loader search path. Plain
    # bootstrap_custom_op_env() (what platform init runs) only sets
    # ASCEND_CUSTOM_OPP_PATH; the vendor-lib variant is what the upstream
    # single-card tests (test_compressor_metadata.py) use before calling these
    # ops. Without it every op call fails with
    # "aclnnSparseAttnSharedkvMetadata ... not in libopapi.so".
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)

    is_context = spec.op_type == OP_CONTEXT
    batch, seq_len = spec.batch, spec.seq_len
    dev = torch.device(device)

    kv_layers = _collect_dsv4_kv_layers(module, vllm_config, attn_layer_name)
    if not kv_layers:
        candidates = [
            f"{rel}:{type(child).__name__}"
            for rel, child in module.named_modules()
            if rel and callable(getattr(child, "get_kv_cache_spec", None))
        ]
        raise RuntimeError(
            "no DSv4 sub-layer exposes get_kv_cache_spec(); cannot assemble DSA metadata. "
            f"Submodules with get_kv_cache_spec returning None: {candidates}"
        )
    logger.info(
        "DSv4 kv-cache sub-layers: %s",
        [f"{n}({type(s).__name__}, bs={s.block_size})" for n, _, s in kv_layers],
    )

    # Group sub-layers by cache layout; the main compressed-MLA group
    # (compress_ratio > 1, e.g. dsa_attn) must be built first because its
    # builder populates the shared ratio_to_sas_metadata caches.
    groups: dict[tuple, list] = {}
    for full_name, child, kv_spec in kv_layers:
        key = (type(kv_spec).__name__, kv_spec.block_size)
        groups.setdefault(key, []).append((full_name, child, kv_spec))
    ordered = sorted(
        groups.values(),
        key=lambda g: getattr(g[0][2], "compress_ratio", 0) <= 1,
    )

    num_reqs = batch
    num_tokens = batch * seq_len if is_context else batch
    max_query_len = seq_len if is_context else 1

    seq_lens_gpu = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
    seq_lens_cpu = torch.full((batch,), seq_len, dtype=torch.int32)
    if is_context:
        qsl = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32, device=dev)
        qsl_cpu = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32)
        num_computed_cpu = torch.zeros(batch, dtype=torch.int32)
        is_prefilling = torch.ones(batch, dtype=torch.bool, device=dev)
        attn_state = AscendAttentionState.PrefillNoCache
    else:
        qsl = torch.arange(0, batch + 1, dtype=torch.int32, device=dev)
        qsl_cpu = torch.arange(0, batch + 1, dtype=torch.int32)
        num_computed_cpu = torch.full((batch,), seq_len - 1, dtype=torch.int32)
        is_prefilling = torch.zeros(batch, dtype=torch.bool, device=dev)
        attn_state = AscendAttentionState.DecodeOnly

    builder_cls = AscendDSABackend.get_builder_cls()
    prefill_sas: dict = {}
    decode_sas: dict = {}
    common_sas: dict = {}

    metadata: dict = {}
    for group_layers in ordered:
        kv_spec = group_layers[0][2]
        bs = kv_spec.block_size
        blocks_per_req = (seq_len + bs - 1) // bs
        num_blocks = num_reqs * blocks_per_req + 1

        # Contiguous per-request block table in THIS group's block size.
        bt = (
            torch.arange(num_reqs, device=dev, dtype=torch.int32).unsqueeze(1)
            * blocks_per_req
            + torch.arange(blocks_per_req, device=dev, dtype=torch.int32)
        )
        if is_context:
            tok = torch.arange(seq_len, device=dev)
            slot_mapping = (
                bt[:, tok // bs] * bs + (tok % bs).unsqueeze(0)
            ).reshape(-1).to(torch.int64)
            positions = (
                torch.arange(seq_len, device=dev)
                .unsqueeze(0)
                .expand(batch, -1)
                .reshape(-1)
                .contiguous()
            )
        else:
            last = seq_len - 1
            slot_mapping = (bt[:, last // bs] * bs + (last % bs)).to(torch.int64)
            positions = torch.full((batch,), last, dtype=torch.int64, device=dev)

        cm = AscendCommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens_gpu,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_computed_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=seq_len,
            block_table_tensor=bt,
            slot_mapping=slot_mapping,
            causal=True,
            is_prefilling=is_prefilling,
            num_input_tokens=num_tokens,
            actual_seq_lengths_q=[],
            positions=positions,
            positions_cpu=None,
            attn_state=attn_state,
            decode_token_per_req=1,
            prefill_context_parallel_metadata=None,
        )

        builder = builder_cls(
            kv_cache_spec=kv_spec,
            layer_names=[
                _dsv4_metadata_key(child, full_name, attn_layer_name)
                for full_name, child, _ in group_layers
            ],
            vllm_config=vllm_config,
            device=dev,
        )
        group_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=cm,
            num_reqs_actual=num_reqs,
            prefill_ratio_to_sas_metadata=prefill_sas,
            decode_ratio_to_sas_metadata=decode_sas,
            common_ratio_to_sas_metadata=common_sas,
            block_size=bs,
        )

        for full_name, child, layer_spec in group_layers:
            child.kv_cache = [_alloc_dsv4_layer_cache(layer_spec, num_blocks, dev)]
            metadata[_dsv4_metadata_key(child, full_name, attn_layer_name)] = group_metadata

    logger.info(
        "DSv4 real attn_metadata built for %d groups, keys=%s",
        len(ordered),
        sorted(metadata),
    )
    return metadata


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

        attn_layer_name = "model.layers.0.self_attn"
        # DSv4 forward runs through torch.ops.vllm.dsa_forward, which requires a
        # live ForwardContext. set_ascend_forward_context auto-populates
        # no_compile_layers from static_forward_context (where the
        # AscendDeepseekSparseAttention was registered during __init__), and a
        # non-None attn_metadata selects the real compute path. The attn_metadata
        # + 6-tuple KV cache are produced by _create_dsv4_kv_cache_and_metadata.
        attn_metadata = _create_dsv4_kv_cache_and_metadata(
            module, vllm_config, spec, device, attn_layer_name
        )

        from contextlib import ExitStack
        from vllm.config import set_current_vllm_config

        # Everything below is constant for a given spec, so it is set up ONCE,
        # outside the timed closure. The benchmark loop calls forward_fn dozens
        # of times per shape; leaving any of it inside means
        #   * every iteration pays for vLLM forward-context construction, which
        #     dominates the measurement (latencies come out ~4 ms and flat in
        #     num_heads, i.e. pure host overhead instead of kernel time);
        #   * the host->device copy of num_tokens_across_dp runs *during* NPU
        #     graph capture, which is illegal — aclrtMemcpy fails with error
        #     107030 ("operation not permitted when a stream is capturing"), so
        #     capture can never succeed for this module.
        stack = ExitStack()
        stack.enter_context(set_current_vllm_config(vllm_config))
        try:
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context
            from vllm_ascend.ascend_config import WeightPrefetchConfig
            from vllm_ascend.utils import set_weight_prefetch_method

            set_weight_prefetch_method(WeightPrefetchConfig({}))
            num_tokens_across_dp = torch.tensor(
                [num_tokens], dtype=torch.int64, device=device
            )
            stack.enter_context(
                set_ascend_forward_context(
                    attn_metadata,
                    vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                )
            )
        except ImportError:
            from vllm.forward_context import set_forward_context

            stack.enter_context(
                set_forward_context(attn_metadata, vllm_config)
            )

        def forward_fn() -> None:
            # vLLM attention modules take (positions, hidden_states, llama_4_scaling).
            module.forward(positions, hidden, None)

        # The forward context has to stay live for the whole benchmark, so expose
        # a closer instead of using a `with` block inside forward_fn.
        forward_fn.close = stack.close  # type: ignore[attr-defined]

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
