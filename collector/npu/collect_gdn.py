"""Qwen3.5 / Qwen3.8 Gated DeltaNet (GDN) collector for Ascend NPU.

Times the four production GDN core kernels that vllm-ascend 0.23.0 dispatches
for the Qwen3.5/Qwen3.8 linear_attention layers, and writes a ``gdn_perf.txt``
consumed by ``perf_database.load_gdn_data``.

This mirrors the NVIDIA-side ``collector/vllm/collect_gdn.py`` geometry and the
NPU-side ``collect_mhc.py`` direct-output + checkpoint/resume layout. The kernel
*logical names* written to the CSV are the four that ``query_gdn`` keys on:

    context    : "causal_conv1d_fn"          (packed-QKV causal conv, run_mode=0)
               : "chunk_gated_delta_rule"    (triton chunked scan)
    generation : "causal_conv1d_update"      (single-step conv, run_mode=1)
               : "fused_sigmoid_gating_delta_rule_update"  (single-step recurrence)

On 910B these map to vllm-ascend's real kernels:

    causal conv      -> torch.ops._C_ascend.npu_causal_conv1d_custom (run_mode 0/1)
    chunked scan     -> vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule
    single-step recur-> torch.ops._C_ascend.npu_recurrent_gated_delta_rule

The g/beta gating (DeviceOperator.fused_gdn_gating) is *not* a GDN kernel in the
SDK model -- it is charged to the per-layer ElementWise norm op -- so g/beta are
precomputed outside the timed region (as random tensors) exactly like the NVIDIA
collector does.

Environment
-----------
Run inside the vllm-ascend 0.23.0 NPU docker image (the one that ships the
``_C_ascend`` custom-op package). The collector bootstraps that package via
``vllm_ascend.utils.bootstrap_custom_op_env`` + ``enable_custom_op`` before
touching ``torch.ops._C_ascend.*``.

Ascend 950DT (A5) caveat
------------------------
``enable_custom_op()`` deliberately short-circuits on A5 (vllm_ascend/utils.py,
"custom op compilation and execution are partially available in ASCEND950 chip,
we temporarily disable all custom ops", issue #7157), i.e. it never imports
``vllm_ascend.vllm_ascend_C`` and the whole ``torch.ops._C_ascend`` namespace
stays empty -> every point dies with
``AttributeError: '_OpNamespace' '_C_ascend' object has no attribute
'npu_causal_conv1d_custom'``. Since the GDN conv / recurrence kernels *are*
that package, ``_init_runtime`` imports the extension explicitly on A5 too (see
``_load_ascend_custom_ops``).

If the image's triton backend cannot build its ``npu_utils`` helper (missing
``/lib64/libm.so.6`` -> ``clang: error: linker command failed``), the triton
chunk scan additionally needs
``init_device_properties_triton``; ``_fallback_triton_device_props`` then fills
the core counters from torch_npu (or from ``AIC_TRITON_NUM_AICORE`` /
``AIC_TRITON_NUM_VECTORCORE``).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# Allow running from any cwd: sibling modules (bench_engine) live here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from bench_engine import BenchResult, benchmark_npu

logger = logging.getLogger(__name__)

# Op identifiers -- MUST match perf_database.query_gdn / load_gdn_data.
OP_NAME = "gdn"
KERNEL_CONV_CTX = "causal_conv1d_fn"
KERNEL_SCAN_CTX = "chunk_gated_delta_rule"
KERNEL_CONV_GEN = "causal_conv1d_update"
KERNEL_RECUR_GEN = "fused_sigmoid_gating_delta_rule_update"

# Default Qwen3.8-27B geometry (overridable for other Qwen3.5 models).
DEFAULT_D_MODEL = 5120
DEFAULT_D_CONV = 4
DEFAULT_NUM_K_HEADS = 16
DEFAULT_HEAD_K_DIM = 128
DEFAULT_NUM_V_HEADS = 48
DEFAULT_HEAD_V_DIM = 128
DEFAULT_TP_SIZES = [1, 2, 4, 8]

# Base sweep grids (mirrors collector/cases/base_ops/gdn.yaml).
CONTEXT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
CONTEXT_SEQ_LENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
GENERATION_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

OUTPUT_COLUMNS = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "phase",
    "batch_size",
    "seq_len",
    "num_tokens",
    "d_model",
    "d_conv",
    "num_k_heads",
    "head_k_dim",
    "num_v_heads",
    "head_v_dim",
    "model_name",
    "latency",
]

CHECKPOINT_FILE = "gdn_checkpoint.json"

# Skip context points whose packed token count exceeds this (HBM safety net).
# Individual kernels are also wrapped in try/except so an OOM on one shape does
# not abort the whole run.
DEFAULT_MAX_CONTEXT_TOKENS = 1_000_000

DTYPE = torch.bfloat16


@dataclass(frozen=True)
class GdnBenchSpec:
    """One (phase, kernel, tp, batch, seq) benchmark point."""

    phase: str
    kernel_source: str
    tp: int
    batch_size: int
    seq_len: int
    d_model: int
    d_conv: int
    num_k_heads: int  # TP-local
    head_k_dim: int
    num_v_heads: int  # TP-local
    head_v_dim: int
    model_name: str

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.seq_len if self.phase == "context" else self.batch_size

    @property
    def conv_dim(self) -> int:
        return 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim


def _spec_key(spec: GdnBenchSpec) -> str:
    return (
        f"{spec.phase}_{spec.kernel_source}_{spec.tp}_{spec.batch_size}_{spec.seq_len}_"
        f"{spec.num_k_heads}_{spec.num_v_heads}"
    )


def _load_checkpoint(output_dir: Path) -> set[str]:
    ckpt_path = output_dir / CHECKPOINT_FILE
    if not ckpt_path.exists():
        return set()
    with open(ckpt_path) as f:
        return set(json.load(f).get("completed", []))


def _save_checkpoint(output_dir: Path, completed: set[str]) -> None:
    ckpt_path = output_dir / CHECKPOINT_FILE
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    tmp_path.replace(ckpt_path)


# ---------------------------------------------------------------------------
# Runtime bootstrap (custom-op package + torch_npu device)
# ---------------------------------------------------------------------------
_runtime = {}


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _patch_pcp_group() -> None:
    """Make get_pcp_group() report world_size=1 for single-card benchmarking.

    vllm-ascend's chunk_gated_delta_rule_fwd does `if get_pcp_group().world_size
    > 1:`; the prefill-context-parallel group is only created on multi-card
    setups, so standalone (single card) it is None and the call asserts
    "prefill context parallel group is not initialized". We are not running
    context parallel, so world_size=1 is the correct value.
    """
    import types

    from vllm_ascend.ops.triton.fla import chunk as chunk_mod

    sentinel = types.SimpleNamespace(world_size=1, rank=0)

    def _dummy() -> types.SimpleNamespace:
        return sentinel

    # chunk.py binds the getter locally via `from ... import get_pcp_group`.
    if "get_pcp_group" in chunk_mod.__dict__:
        chunk_mod.get_pcp_group = _dummy
    # Also patch the source module in case it is referenced indirectly.
    try:
        import vllm.distributed.parallel_state as _ps

        _ps.get_pcp_group = _dummy
    except Exception:
        pass


_C_ASCEND_KERNELS = ("npu_causal_conv1d_custom", "npu_recurrent_gated_delta_rule")


def _load_ascend_custom_ops() -> None:
    """Force-register ``torch.ops._C_ascend.*`` -- on Ascend 950 (A5) too.

    ``vllm_ascend.utils.enable_custom_op()`` short-circuits on A5::

        # FIXME(linfeng): ... custom op compilation and execution are partially
        # available in ASCEND950 chip, we temporarily disable all custom ops.
        if envs.VLLM_BATCH_INVARIANT or get_ascend_device_type() == AscendDeviceType.A5:
            _CUSTOM_OP_ENABLED = False
            return _CUSTOM_OP_ENABLED

    i.e. it never imports ``vllm_ascend.vllm_ascend_C``, so the torch op library
    that *defines* ``npu_causal_conv1d_custom`` / ``npu_recurrent_gated_delta_rule``
    is never loaded and every GDN point dies with
    ``AttributeError: '_OpNamespace' '_C_ascend' object has no attribute ...``
    (on 910B the same collector works because enable_custom_op() does import it).

    Those two ops are exactly the GDN conv / recurrence kernels, so import the
    extension explicitly (the vendor op_api lib is already on LD_LIBRARY_PATH).
    If the aclnn implementation really is unavailable on this build, the call
    fails per point and is reported as a normal point error instead of taking
    the whole sweep down at attribute-lookup time.
    """
    try:
        import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
    except Exception as e:
        logger.warning(
            "could not import vllm_ascend.vllm_ascend_C (%s: %s); the "
            "torch.ops._C_ascend kernels %s stay unregistered",
            type(e).__name__, e, _C_ASCEND_KERNELS,
        )
        return

    missing = [n for n in _C_ASCEND_KERNELS if not hasattr(torch.ops._C_ascend, n)]
    if missing:
        logger.warning(
            "vllm_ascend_C loaded but %s are still not registered on this "
            "build; the corresponding GDN points will fail", missing,
        )
    else:
        logger.info("torch.ops._C_ascend kernels available: %s", list(_C_ASCEND_KERNELS))


def _fallback_triton_device_props() -> None:
    """Fill triton_utils' core counters when its helper .so cannot be built.

    triton's Ascend driver compiles a small ``npu_utils`` extension on first
    use; on images missing ``/lib64/libm.so.6`` that clang++ build always fails
    ("cannot find /lib64/libm.so.6" -> "linker command failed with exit code
    1"), so ``init_device_properties_triton()`` raises and every triton kernel
    (here ``chunk_gated_delta_rule``, via ``l2norm_fwd ->
    get_vectorcore_num()``) aborts with "AssertionError: Device properties not
    initialized."

    The counters only size launch grids, so an approximate value is enough to
    get real timings. Sources, in order:
      1. ``$AIC_TRITON_NUM_AICORE`` / ``$AIC_TRITON_NUM_VECTORCORE`` (override)
      2. ``torch.npu.get_device_properties()`` (cube_core_num / vector_core_num)
    """
    import vllm_ascend.ops.triton.triton_utils as triton_utils

    aic = os.environ.get("AIC_TRITON_NUM_AICORE")
    vec = os.environ.get("AIC_TRITON_NUM_VECTORCORE")
    if aic and vec:
        triton_utils._NUM_AICORE = int(aic)
        triton_utils._NUM_VECTORCORE = int(vec)
        logger.info(
            "triton device props from env override: aicore=%s vectorcore=%s", aic, vec
        )
        return

    try:
        props = torch.npu.get_device_properties(torch.npu.current_device())
    except Exception as e:
        logger.warning(
            "triton device props unavailable (%s: %s); chunk_gated_delta_rule "
            "will abort with 'Device properties not initialized'. Fix the "
            "image (/lib64/libm.so.6 symlink) or set AIC_TRITON_NUM_AICORE / "
            "AIC_TRITON_NUM_VECTORCORE.",
            type(e).__name__, e,
        )
        return

    def _pick(candidates: tuple[str, ...]) -> int:
        for name in candidates:
            value = getattr(props, name, None)
            if isinstance(value, int) and value > 0:
                return value
        return 0

    aicore = _pick(("ai_core_num", "aicore_num", "num_aicore", "cube_core_num"))
    vectorcore = _pick(("vector_core_num", "num_vectorcore", "vectorcore_num"))
    if not aicore or not vectorcore:
        logger.warning(
            "triton device props unavailable (cube/vector core counts not "
            "exposed by torch_npu); chunk_gated_delta_rule will abort with "
            "'Device properties not initialized'. Fix the image "
            "(/lib64/libm.so.6 symlink) or set AIC_TRITON_NUM_AICORE / "
            "AIC_TRITON_NUM_VECTORCORE."
        )
        return
    triton_utils._NUM_AICORE = aicore
    triton_utils._NUM_VECTORCORE = vectorcore
    logger.info(
        "triton device props from torch_npu: aicore=%d vectorcore=%d "
        "(approximate; only used to size launch grids)", aicore, vectorcore,
    )


def _init_runtime(device: str) -> None:
    """Bootstrap the vllm-ascend custom-op package and select the NPU device."""
    torch.npu.set_device(0)
    if hasattr(torch, "npu"):
        torch.npu.config.allow_internal_format = True

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm_ascend.ops.triton.fla import chunk as chunk_mod
    from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

    bootstrap_custom_op_env(include_vendor_lib=True)
    enable_custom_op()
    # A5 disables custom ops wholesale (see _load_ascend_custom_ops); the GDN
    # conv / recurrence kernels live in that very package, so load it anyway.
    _load_ascend_custom_ops()

    # Populate the triton module-level _NUM_AICORE / _NUM_VECTORCORE counters.
    # In production vllm-ascend's worker calls this during device init
    # (worker.py); we bypass the worker, so trigger it ourselves. Without it the
    # chunk scan dies in l2norm_fwd -> get_vectorcore_num() with
    # "AssertionError: Device properties not initialized."
    try:
        from vllm_ascend.ops.triton.triton_utils import (
            get_aicore_num,
            get_vectorcore_num,
            init_device_properties_triton,
        )

        init_device_properties_triton()
        get_aicore_num()
        get_vectorcore_num()
    except Exception as e:
        logger.warning(
            "init_device_properties_triton failed (%s: %s); falling back to "
            "approximate core counts",
            type(e).__name__,
            e,
        )
        _fallback_triton_device_props()

    # The triton chunk scan reads vLLM parallel-state groups (TP for collectives,
    # PCP for context-parallel). On a real worker these are created at model load;
    # standalone we set up a single-card (tp=1, pp=1) world and shim PCP.
    try:
        import os

        from vllm.distributed import init_distributed_environment
        from vllm.distributed.parallel_state import ensure_model_parallel_initialized

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))

        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)
    except Exception as e:
        logger.warning("vLLM distributed init failed (%s: %s)", type(e).__name__, e)

    _patch_pcp_group()

    _runtime.update(
        pad_slot_id=PAD_SLOT_ID,
        chunk_mod=chunk_mod,
        chunk_gated_delta_rule=chunk_gated_delta_rule,
        prepare_chunk_indices=prepare_chunk_indices,
        device=torch.device("npu"),
    )


@contextmanager
def _no_forward_context():
    """Make ``chunk_gated_delta_rule_fwd`` callable outside a vLLM forward pass.

    The 910B triton chunk reads ``get_forward_context().attn_metadata``
    unconditionally; with no forward context installed that raises. We shim the
    module-level import so it sees a context whose ``attn_metadata`` is ``None``
    (pure-prefill path, num_decodes=0).
    """
    chunk_mod = _runtime["chunk_mod"]
    original = chunk_mod.get_forward_context
    chunk_mod.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)  # type: ignore[assignment]
    try:
        yield
    finally:
        chunk_mod.get_forward_context = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Kernel builders (weights/state pre-materialised outside the timed region)
# ---------------------------------------------------------------------------
def _build_conv_ctx_func(spec: GdnBenchSpec) -> Callable[[], None]:
    dev = _runtime["device"]
    pad_slot_id = _runtime["pad_slot_id"]
    conv_dim = spec.conv_dim
    d_conv = spec.d_conv
    batch = spec.batch_size
    seq_len = spec.seq_len
    num_tokens = spec.num_tokens

    # causal_conv1d requires the channel dim (conv_dim) to be the LAST axis of
    # x / weight / conv_state; otherwise the aclnn tiling rejects it with
    # "dim must satisfy dim % 16 == 0" or "weight.shape[1] must equal dim".
    x = torch.randn(num_tokens, conv_dim, dtype=DTYPE, device=dev)  # [tokens, C]
    x_origin = x
    weight = torch.randn(d_conv, conv_dim, dtype=DTYPE, device=dev)  # [K, C]
    weight_t = weight
    conv_state = torch.zeros(batch, d_conv - 1, conv_dim, dtype=DTYPE, device=dev)  # [b, K-1, C]
    out = torch.empty_like(x)  # [tokens, C]
    query_start_loc = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int32, device=dev)
    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    has_initial_state = torch.zeros(batch, dtype=torch.bool, device=dev)

    def run() -> None:
        torch.ops._C_ascend.npu_causal_conv1d_custom(  # type: ignore[attr-defined]
            out,
            x_origin,
            weight_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=has_initial_state,
            num_accepted_tokens_opt=None,
            activation_mode=1,
            pad_slot_id=pad_slot_id,
            run_mode=0,
        )

    run()
    torch.npu.synchronize()
    return run


def _build_scan_ctx_func(spec: GdnBenchSpec) -> Callable[[], None]:
    dev = _runtime["device"]
    nk, hk = spec.num_k_heads, spec.head_k_dim
    nv, hv = spec.num_v_heads, spec.head_v_dim
    num_tokens = spec.num_tokens
    batch = spec.batch_size
    seq_len = spec.seq_len

    # vllm-ascend 0.23.0 chunk_gated_delta_rule (verified against source):
    #   - q/k/v must share DTYPE and NOT be float32; beta must be 3-D [B,T,H].
    #   - With cu_seqlens, B must be 1 and inputs flattened via
    #     rearrange('b t ... -> 1 (b t) ...'): T becomes total_tokens = batch*seq_len,
    #     and cu_seqlens (length batch+1) marks each sequence boundary.
    #   - initial_state first dim must equal n_seq = len(cu_seqlens)-1 = batch,
    #     and the doc example keeps it in DTYPE (bf16) like the other tensors.
    #   - g MUST be float32: chunk_local_cumsum defaults output_dtype to torch.float,
    #     so s(g).dtype must equal it or the kernel fails to compile.
    #   - The AscendC kernel chunk_gated_delta_rule_fwd_h processes tokens in fixed
    #     64-token chunks and reads a FULL block per chunk index (verified against
    #     csrc/.../gdn_fwd_h_kernel and the aclnn test which only uses T=128). It
    #     does NOT handle a partial last chunk, so each sequence length must be a
    #     multiple of 64 -- otherwise it reads out of bounds (aicore OOB at s=1).
    #     We pad every sequence up to the next multiple of 64 (matching how vLLM
    #     pads prefill to chunk boundaries); cu_seqlens marks the padded lengths.
    CHUNK = 64
    eff_seq = max(((seq_len + CHUNK - 1) // CHUNK) * CHUNK, CHUNK)
    total = batch * eff_seq
    q = torch.randn(1, total, nk, hk, dtype=DTYPE, device=dev)
    k = torch.randn(1, total, nk, hk, dtype=DTYPE, device=dev)
    v = torch.randn(1, total, nv, hv, dtype=DTYPE, device=dev)
    g = torch.randn(1, total, nv, dtype=torch.float32, device=dev)
    beta = torch.rand(1, total, nv, dtype=DTYPE, device=dev)
    initial_state = torch.zeros(batch, nv, hk, hv, dtype=DTYPE, device=dev)
    cu_seqlens = torch.arange(0, total + 1, eff_seq, dtype=torch.int32, device=dev)
    chunk_gated_delta_rule = _runtime["chunk_gated_delta_rule"]
    prepare_chunk_indices = _runtime["prepare_chunk_indices"]
    scale = 1.0 / math.sqrt(hk)

    # The AscendC kernel chunk_gated_delta_rule_fwd_h is varlen-only for its
    # chunk indexing: it reads per-chunk token offsets from `chunk_indices`,
    # which chunk_gated_delta_rule_fwd normally obtains from the attention
    # backend's prebuilt_meta. We don't run a real model, so we synthesize the
    # minimal prebuilt_meta the fwd function touches (every attribute it reads
    # must exist, even if None): cu_seqlens_host (a tuple, fed straight to the
    # kernel) and chunk_indices_chunk64 (a tensor, turned into the int64 tuple
    # the kernel expects). keep_meta=None makes cu_seqlens_kern fall back to
    # cu_seqlens_host, and the all-positive seq lengths need no compaction.
    class _PrebuiltMeta:
        def __init__(self, cu_seqlens_host, chunk_indices_chunk64):
            self.block_indices_cumsum = None
            self.cu_seqlens_host = cu_seqlens_host
            self.chunk_indices_chunk64 = chunk_indices_chunk64
            self.chunk_indices_chunk64_host = None
            self.chunk_offsets_chunk64 = None
            self.update_chunk_offsets_chunk64 = None
            self.final_chunk_indices_chunk64 = None
            self.chunk_indices_large_block = None
            self.keep_meta = None
            self.cu_seqlens_kern = None

    # NOTE: chunk_gated_delta_rule_fwd_h's aclnn signature declares
    # `cu_seqlens: int[]?` and `chunk_indices: int[]?` -- it accepts ONLY a host
    # Python list/tuple of ints (NOT a tensor). The op copies these to device at
    # call time, which Ascend permits during capture ONLY in RELAXED mode
    # ("operation not permitted when a stream is capturing and the specified
    # capture mode is not relaxed"). run_benchmark therefore captures this
    # kernel with graph_capture_error_mode="relaxed"; the tuple form is what the
    # op actually accepts and is also what the eager run needs.
    meta = _PrebuiltMeta(
        cu_seqlens_host=tuple(cu_seqlens.tolist()),
        chunk_indices_chunk64=prepare_chunk_indices(cu_seqlens, CHUNK),
    )

    def run() -> None:
        with _no_forward_context():
            chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                prebuilt_meta=meta,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )

    run()
    torch.npu.synchronize()
    return run


def _build_conv_gen_func(spec: GdnBenchSpec) -> Callable[[], None]:
    dev = _runtime["device"]
    pad_slot_id = _runtime["pad_slot_id"]
    conv_dim = spec.conv_dim
    d_conv = spec.d_conv
    batch = spec.batch_size

    # Same axis convention as the context conv: channel dim (conv_dim) last.
    x = torch.randn(batch, conv_dim, dtype=DTYPE, device=dev)  # [batch, C]
    x_origin = x
    weight = torch.randn(d_conv, conv_dim, dtype=DTYPE, device=dev)  # [K, C]
    weight_t = weight
    conv_state = torch.zeros(batch, d_conv - 1, conv_dim, dtype=DTYPE, device=dev)  # [b, K-1, C]
    out = torch.empty_like(x)
    # Each decode sequence contributes exactly one token.
    query_start_loc = torch.arange(0, batch + 1, dtype=torch.int32, device=dev)
    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)

    def run() -> None:
        torch.ops._C_ascend.npu_causal_conv1d_custom(  # type: ignore[attr-defined]
            out,
            x_origin,
            weight_t,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=query_start_loc,
            cache_indices_opt=cache_indices,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=None,
            activation_mode=1,
            pad_slot_id=pad_slot_id,
            run_mode=1,
        )

    run()
    torch.npu.synchronize()
    return run


def _build_recurrence_gen_func(spec: GdnBenchSpec) -> Callable[[], None]:
    dev = _runtime["device"]
    nk, hk = spec.num_k_heads, spec.head_k_dim
    nv, hv = spec.num_v_heads, spec.head_v_dim
    batch = spec.batch_size

    query = torch.nn.functional.normalize(
        torch.randn(batch, nk, hk, dtype=DTYPE, device=dev), dim=-1
    )
    key = torch.nn.functional.normalize(
        torch.randn(batch, nk, hk, dtype=DTYPE, device=dev), dim=-1
    )
    value = torch.randn(batch, nv, hv, dtype=DTYPE, device=dev)
    g = torch.randn(batch, nv, dtype=torch.float32, device=dev)  # log-space inside op
    beta = torch.rand(batch, nv, dtype=DTYPE, device=dev)
    # Recurrence state layout: [slots, nv, hv, hk] (production ssm_state layout).
    state = torch.zeros(batch, nv, hv, hk, dtype=torch.float32, device=dev)
    scale = hk**-0.5
    actual_seq_lengths = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    actual_seq_lengths[1:] = 1
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device=dev)

    def run() -> None:
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(  # type: ignore[attr-defined]
            query,
            key,
            value,
            state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            g=g,
        )

    run()
    torch.npu.synchronize()
    return run


_KERNEL_BUILDERS = {
    KERNEL_CONV_CTX: _build_conv_ctx_func,
    KERNEL_SCAN_CTX: _build_scan_ctx_func,
    KERNEL_CONV_GEN: _build_conv_gen_func,
    KERNEL_RECUR_GEN: _build_recurrence_gen_func,
}


# ---------------------------------------------------------------------------
# Spec grid
# ---------------------------------------------------------------------------
def build_specs(
    d_model: int,
    d_conv: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    tp_sizes: list[int],
    model_name: str,
    max_context_tokens: int,
) -> list[GdnBenchSpec]:
    specs: list[GdnBenchSpec] = []
    for tp in tp_sizes:
        assert num_k_heads % tp == 0 and num_v_heads % tp == 0, (
            f"GDN heads must divide tp: nk={num_k_heads}, nv={num_v_heads}, tp={tp}"
        )
        local_nk = num_k_heads // tp
        local_nv = num_v_heads // tp

        for batch_size in CONTEXT_BATCH_SIZES:
            for seq_len in CONTEXT_SEQ_LENS:
                if batch_size * seq_len > max_context_tokens:
                    continue
                for ks in (KERNEL_CONV_CTX, KERNEL_SCAN_CTX):
                    specs.append(
                        GdnBenchSpec(
                            phase="context",
                            kernel_source=ks,
                            tp=tp,
                            batch_size=batch_size,
                            seq_len=seq_len,
                            d_model=d_model,
                            d_conv=d_conv,
                            num_k_heads=local_nk,
                            head_k_dim=head_k_dim,
                            num_v_heads=local_nv,
                            head_v_dim=head_v_dim,
                            model_name=model_name,
                        )
                    )

        for batch_size in GENERATION_BATCH_SIZES:
            for ks in (KERNEL_CONV_GEN, KERNEL_RECUR_GEN):
                specs.append(
                    GdnBenchSpec(
                        phase="generation",
                        kernel_source=ks,
                        tp=tp,
                        batch_size=batch_size,
                        seq_len=1,
                        d_model=d_model,
                        d_conv=d_conv,
                        num_k_heads=local_nk,
                        head_k_dim=head_k_dim,
                        num_v_heads=local_nv,
                        head_v_dim=head_v_dim,
                        model_name=model_name,
                    )
                )
    return specs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _make_row(spec: GdnBenchSpec, result: BenchResult, framework: str, version: str, device: str) -> dict[str, str]:
    return {
        "framework": framework,
        "version": version,
        "device": device,
        "op_name": OP_NAME,
        "kernel_source": spec.kernel_source,
        "phase": spec.phase,
        "batch_size": str(spec.batch_size),
        "seq_len": str(spec.seq_len),
        "num_tokens": str(spec.num_tokens),
        "d_model": str(spec.d_model),
        "d_conv": str(spec.d_conv),
        "num_k_heads": str(spec.num_k_heads),
        "head_k_dim": str(spec.head_k_dim),
        "num_v_heads": str(spec.num_v_heads),
        "head_v_dim": str(spec.head_v_dim),
        "model_name": spec.model_name,
        "latency": f"{result.avg_us / 1000.0:.6f}",
    }


def run_benchmark(
    specs: list[GdnBenchSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
    timing_mode: str = "auto",
    framework: str = "vllm-ascend",
    version: str = "0.23.0",
    device: str = "Ascend 910B",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    csv_path = output_dir / "gdn_perf.txt"
    file_exists = csv_path.exists() and resume
    fh = open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    total = len(specs)
    skipped = errors = 0
    t_start = __import__("time").monotonic()

    try:
        for i, spec in enumerate(specs):
            key = _spec_key(spec)
            if key in completed:
                skipped += 1
                continue

            progress = f"[{i + 1}/{total}]"
            logger.info(
                "%s %s/%s tp=%d b=%d s=%d (nk=%d nv=%d)",
                progress, spec.phase, spec.kernel_source, spec.tp,
                spec.batch_size, spec.seq_len, spec.num_k_heads, spec.num_v_heads,
            )

            builder = _KERNEL_BUILDERS[spec.kernel_source]
            func = None
            failed = False
            try:
                func = builder(spec)
                # chunk_gated_delta_rule_fwd_h takes cu_seqlens as a HOST int
                # list and copies it H2D on every call, so it can only be
                # captured in RELAXED mode (GLOBAL/THREAD_LOCAL abort with
                # "operation not permitted when a stream is capturing and the
                # specified capture mode is not relaxed"). The _C_ascend conv /
                # recurrence kernels capture fine in the default GLOBAL mode.
                capture_mode = (
                    "relaxed" if spec.kernel_source == KERNEL_SCAN_CTX else None
                )
                result = benchmark_npu(
                    func,
                    warmup_iters=warmup_iters,
                    num_runs=bench_iters,
                    timing_mode=timing_mode,
                    graph_capture_error_mode=capture_mode,
                )
                writer.writerow(_make_row(spec, result, framework, version, device))
                logger.info("%s -> %.2f us (avg of %d runs)", progress, result.avg_us, result.num_runs)
            except Exception:
                logger.exception(
                    "%s FAILED %s/%s tp=%d b=%d s=%d",
                    progress, spec.phase, spec.kernel_source, spec.tp,
                    spec.batch_size, spec.seq_len,
                )
                errors += 1
                failed = True
            finally:
                del func
                try:
                    torch.npu.synchronize()
                except RuntimeError:
                    pass
                try:
                    torch.npu.empty_cache()
                except RuntimeError:
                    # Device may be in a bad state after a kernel fault; do not
                    # let cleanup abort the whole run (resume would lose progress).
                    pass

            if failed:
                # A runtime aicore fault poisons the NPU device context: every
                # subsequent synchronize fails with "ACL stream synchronize
                # failed", cascading into hundreds of false failures. Detect that
                # and halt so the *root* cause (the FAILED spec above) is visible
                # instead of being buried under cascade noise. Re-run with
                # --resume after fixing the root cause; completed specs are kept.
                try:
                    torch.npu.synchronize()
                except RuntimeError:
                    logger.error(
                        "NPU device is in a fatal error state after the failure "
                        "above; halting to avoid cascading false failures. Fix "
                        "the root cause (the FAILED spec shown above) and re-run "
                        "with --resume."
                    )
                    break
                continue

            completed.add(key)
            if len(completed) % 10 == 0:
                fh.flush()
                _save_checkpoint(output_dir, completed)
    finally:
        fh.flush()
        fh.close()
        _save_checkpoint(output_dir, completed)

    elapsed = __import__("time").monotonic() - t_start
    logger.info(
        "Done: %d benchmarked, %d skipped, %d errors in %.1fs",
        len(completed) - skipped, skipped, errors, elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5/3.8 GDN kernel collector for Ascend NPU")
    parser.add_argument("--output-dir", type=str, default="./gdn_data", help="Output dir for gdn_perf.txt")
    parser.add_argument("--model-name", type=str, default="qwen3_8_27B", help="model_name column value")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--d-conv", type=int, default=DEFAULT_D_CONV)
    parser.add_argument("--num-k-heads", type=int, default=DEFAULT_NUM_K_HEADS)
    parser.add_argument("--head-k-dim", type=int, default=DEFAULT_HEAD_K_DIM)
    parser.add_argument("--num-v-heads", type=int, default=DEFAULT_NUM_V_HEADS)
    parser.add_argument("--head-v-dim", type=int, default=DEFAULT_HEAD_V_DIM)
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=DEFAULT_TP_SIZES)
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--timing-mode", default="auto", choices=["auto", "graph", "eager"],
        help="Latency source: auto = min(graph, eager); "
             "graph = prefer NPU graph, fall back to eager on capture failure; "
             "eager = eager-only (no graph capture).",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--framework", default="vllm-ascend")
    parser.add_argument("--version", default="0.23.0")
    parser.add_argument("--device", default="Ascend 910B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    _init_runtime(args.device)
    logger.info("Runtime ready on %s", _runtime.get("device"))

    specs = build_specs(
        d_model=args.d_model,
        d_conv=args.d_conv,
        num_k_heads=args.num_k_heads,
        head_k_dim=args.head_k_dim,
        num_v_heads=args.num_v_heads,
        head_v_dim=args.head_v_dim,
        tp_sizes=args.tp_sizes,
        model_name=args.model_name,
        max_context_tokens=args.max_context_tokens,
    )
    logger.info("Total specs: %d", len(specs))

    run_benchmark(
        specs=specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        resume=args.resume,
        timing_mode=args.timing_mode,
        framework=args.framework,
        version=args.version,
        device=args.device,
    )


if __name__ == "__main__":
    main()
