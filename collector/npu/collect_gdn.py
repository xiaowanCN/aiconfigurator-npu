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


def _init_runtime(device: str) -> None:
    """Bootstrap the vllm-ascend custom-op package and select the NPU device."""
    torch.npu.set_device(0)
    if hasattr(torch, "npu"):
        torch.npu.config.allow_internal_format = True

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm_ascend.ops.triton.fla import chunk as chunk_mod
    from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
    from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

    bootstrap_custom_op_env(include_vendor_lib=True)
    enable_custom_op()

    _runtime.update(
        pad_slot_id=PAD_SLOT_ID,
        chunk_mod=chunk_mod,
        chunk_gated_delta_rule=chunk_gated_delta_rule,
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

    x = torch.randn(num_tokens, conv_dim, dtype=DTYPE, device=dev)
    x_origin = x.transpose(0, 1)  # [conv_dim, num_tokens]
    weight = torch.randn(conv_dim, d_conv, dtype=DTYPE, device=dev)
    weight_t = weight.transpose(0, 1)  # [d_conv, conv_dim]
    # SD-layout conv state (matches vllm-ascend standalone test).
    conv_state = torch.zeros(batch, d_conv - 1, conv_dim, dtype=DTYPE, device=dev).transpose(-1, -2)
    out = torch.empty_like(x_origin)
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

    q = torch.randn(1, num_tokens, nk, hk, dtype=DTYPE, device=dev)
    k = torch.randn(1, num_tokens, nk, hk, dtype=DTYPE, device=dev)
    v = torch.randn(1, num_tokens, nv, hv, dtype=DTYPE, device=dev)
    g = torch.randn(1, num_tokens, nv, dtype=torch.float32, device=dev)
    beta = torch.rand(1, num_tokens, nv, dtype=DTYPE, device=dev)
    initial_state = torch.zeros(batch, nv, hk, hv, dtype=torch.float32, device=dev)
    cu_seqlens = torch.arange(0, num_tokens + 1, seq_len, dtype=torch.int64, device=dev)
    chunk_gated_delta_rule = _runtime["chunk_gated_delta_rule"]
    scale = 1.0 / math.sqrt(hk)

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
                prebuilt_meta=None,
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

    x = torch.randn(batch, conv_dim, dtype=DTYPE, device=dev)
    x_origin = x.transpose(0, 1)  # [conv_dim, batch]
    weight = torch.randn(conv_dim, d_conv, dtype=DTYPE, device=dev)
    weight_t = weight.transpose(0, 1)  # [d_conv, conv_dim]
    conv_state = torch.zeros(batch, d_conv - 1, conv_dim, dtype=DTYPE, device=dev).transpose(-1, -2)
    out = torch.empty_like(x_origin)
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
            try:
                func = builder(spec)
                result = benchmark_npu(func, warmup_iters=warmup_iters, num_runs=bench_iters, timing_mode=timing_mode)
                writer.writerow(_make_row(spec, result, framework, version, device))
                logger.info("%s -> %.2f us (avg of %d runs)", progress, result.avg_us, result.num_runs)
            except Exception:
                logger.exception(
                    "%s FAILED %s/%s tp=%d b=%d s=%d",
                    progress, spec.phase, spec.kernel_source, spec.tp,
                    spec.batch_size, spec.seq_len,
                )
                errors += 1
                continue
            finally:
                del func
                try:
                    torch.npu.synchronize()
                except RuntimeError:
                    pass
                torch.npu.empty_cache()

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
