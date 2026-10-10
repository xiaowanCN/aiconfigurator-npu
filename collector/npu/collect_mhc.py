"""DeepSeek-V4 mHC module collector for Ascend NPU.

Collects the manifold-constrained hyper-connection (mHC) module that DeepSeek-V4
applies twice per layer (a ``pre`` site before attention and a ``post`` site after
it). aiconfigurator models it as ``sdk/operations.py:DeepSeekV4MHCModule``.

Op graph (mirrors aiconfigurator's ``operators/mhc.rs``)::

    hc_dim = hc_mult * hidden_size                 # expanded residual stream
    mix_hc = (2 + hc_mult) * hc_mult               # mixing coefficient matrix width
    sites  = 2                                     # attention mHC + FFN mHC

    pre :  x[T, h] --W_xpre--> [T, hc_dim]
                   --W_mix---> [T, mix_hc]         # 2*nt*hc_dim*mix_hc
                   --norm/gate/residual-->         # nt*hc_dim*3
                   --sinkhorn(hc, hc) x iters-->   # nt*(hc^2 + 2hc)*sinkhorn_iters
                   --W_down--> [T, h]              # 2*nt*hc*h

    post:  x[T, hc_dim] --W_up--> [T, h]           # 2*nt*hc*h*h (per-lane) + 2*nt*hc*h

Output:
    mhc_module_perf.txt -- consumed by perf_database.load_mhc_module_data(),
    indexed as ``data[op_name][hc_mult][hidden_size][num_tokens]``.
    ``num_tokens`` is the only interpolated axis; ``op_name`` / ``hc_mult`` /
    ``hidden_size`` are exact keys.

Without this table SILICON mode raises and HYBRID falls back to ``SOL / 0.5``.

NOTE: this is a *reference* implementation of the mHC math, not a call into
vllm-ascend's own module. It is self-contained (torch_npu only) so it can be
run as soon as a card is free; once vllm-ascend exposes a DSv4 mHC module,
swap in the real module and keep the same output schema.
"""

import argparse
import csv
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from bench_engine import BenchResult, benchmark_npu

logger = logging.getLogger(__name__)

# --- Op identifiers (must match perf_database.query_mhc_module) ---
OP_PRE = "pre"
OP_POST = "post"
SUPPORTED_OP_TYPES = (OP_PRE, OP_POST)

# Token counts aiconfigurator queries: prefill token counts and decode batches.
DEFAULT_NUM_TOKENS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

OUTPUT_COLUMNS = [
    "framework",
    "version",
    "device",
    "op_name",
    "num_tokens",
    "hc_mult",
    "hidden_size",
    "latency",
]

CHECKPOINT_FILE = "mhc_checkpoint.json"

# GEMMQuantMode.float16 is measured as bf16 on Ascend, matching the other
# collectors in this repo (see convert_to_aiconfigurator's bf16 -> float16 map).
DTYPE_MAP = {"float16": torch.bfloat16}


@dataclass(frozen=True)
class MhcSpec:
    """Immutable mHC benchmark specification (DeepSeek-V4-Pro defaults)."""

    op_type: str
    num_tokens: int
    hc_mult: int = 4
    hidden_size: int = 7168
    sinkhorn_iters: int = 20


def _spec_key(spec: MhcSpec) -> str:
    return f"{spec.op_type}_{spec.num_tokens}_{spec.hc_mult}_{spec.hidden_size}_{spec.sinkhorn_iters}"


def _load_checkpoint(output_dir: Path) -> set[str]:
    ckpt_path = output_dir / CHECKPOINT_FILE
    if not ckpt_path.exists():
        return set()
    with open(ckpt_path) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def _save_checkpoint(output_dir: Path, completed: set[str]) -> None:
    ckpt_path = output_dir / CHECKPOINT_FILE
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    tmp_path.replace(ckpt_path)


def _sinkhorn(log_alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Sinkhorn-Knopp normalisation of a [hc, hc] log matrix (reference impl)."""
    alpha = torch.exp(log_alpha)
    for _ in range(iters):
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        alpha = alpha / alpha.sum(dim=-2, keepdim=True).clamp_min(1e-9)
    return alpha


def create_mhc_func(
    spec: MhcSpec, device: torch.device, dtype: torch.dtype
) -> Callable[[], torch.Tensor]:
    """Build the mHC ``pre`` / ``post`` forward for one spec.

    Weights are pre-materialised outside the timed region so only the math that
    aiconfigurator charges to the module is measured.
    """
    nt = spec.num_tokens
    h = spec.hidden_size
    hc = spec.hc_mult
    hc_dim = hc * h
    mix_hc = (2 + hc) * hc
    sites = 2  # attention mHC + FFN mHC

    x = torch.randn(nt, h, dtype=dtype, device=device)
    # Per-site weights; loop over sites so FLOPs match the analytic model.
    w_up = [torch.randn(h, hc_dim, dtype=dtype, device=device) for _ in range(sites)]
    w_mix = [torch.randn(hc_dim, mix_hc, dtype=dtype, device=device) for _ in range(sites)]
    w_down = [torch.randn(mix_hc, hc_dim, dtype=dtype, device=device) for _ in range(sites)]
    w_out = [torch.randn(hc_dim, h, dtype=dtype, device=device) for _ in range(sites)]
    w_lane = [torch.randn(h, h, dtype=dtype, device=device) for _ in range(sites)]
    log_alpha = torch.randn(hc, hc, dtype=torch.float32, device=device)
    # Sinkhorn is data independent -> precompute once (matching upstream, which
    # charges the iteration cost to FLOPs but runs it on a tiny [hc, hc] matrix).
    alpha = _sinkhorn(log_alpha, spec.sinkhorn_iters).to(dtype)

    if spec.op_type == OP_PRE:
        def forward() -> torch.Tensor:
            out = x
            for i in range(sites):
                hyper = out @ w_up[i]                 # [T, hc_dim]
                mix = hyper @ w_mix[i]                # [T, mix_hc]
                mix = mix * alpha.mean() + mix        # gate (elementwise)
                hyper = hyper + mix @ w_down[i]       # residual back to hc_dim
                out = out + hyper @ w_out[i]          # [T, h]
            return out
    else:
        def forward() -> torch.Tensor:
            out = x
            for i in range(sites):
                hyper = out @ w_up[i]                 # [T, hc_dim]
                # per-lane projection back to the residual width
                lanes = hyper.view(nt, hc, h)
                mixed = lanes.sum(dim=1) @ w_lane[i]  # [T, h]
                out = out + mixed
            return out

    forward()
    torch.npu.synchronize()
    return forward


def _make_row(
    spec: MhcSpec,
    result: BenchResult,
    framework: str,
    version: str,
    device: str,
) -> dict[str, str]:
    return {
        "framework": framework,
        "version": version,
        "device": device,
        "op_name": spec.op_type,
        "num_tokens": str(spec.num_tokens),
        "hc_mult": str(spec.hc_mult),
        "hidden_size": str(spec.hidden_size),
        "latency": f"{result.avg_us / 1000.0:.6f}",
    }


def _build_spec_list(
    op_types: list[str],
    num_tokens_list: list[int],
    hc_mult: int,
    hidden_size: int,
    sinkhorn_iters: int,
) -> list[MhcSpec]:
    return [
        MhcSpec(
            op_type=op_type,
            num_tokens=num_tokens,
            hc_mult=hc_mult,
            hidden_size=hidden_size,
            sinkhorn_iters=sinkhorn_iters,
        )
        for op_type in op_types
        for num_tokens in num_tokens_list
    ]


def run_benchmark(
    specs: list[MhcSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
    dtype: torch.dtype,
    framework: str = "vllm-ascend",
    version: str = "0.23.0",
    device: str = "Ascend 910B",
) -> None:
    npu_device = torch.device("npu")
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    csv_path = output_dir / "mhc_module_perf.txt"
    file_exists = csv_path.exists() and resume
    fh = open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
    if not file_exists:
        writer.writeheader()

    total = len(specs)
    skipped = 0
    errors = 0
    t_start = time.monotonic()

    try:
        for i, spec in enumerate(specs):
            key = _spec_key(spec)
            if key in completed:
                skipped += 1
                continue

            progress = f"[{i + 1}/{total}]"
            logger.info("%s %s tokens=%d", progress, spec.op_type, spec.num_tokens)

            mhc_func = None
            try:
                mhc_func = create_mhc_func(spec, npu_device, dtype)
                result = benchmark_npu(mhc_func, warmup_iters=warmup_iters, num_runs=bench_iters)
                writer.writerow(_make_row(spec, result, framework, version, device))
                logger.info("%s -> %.2f us (avg of %d runs)", progress, result.avg_us, result.num_runs)
            except Exception:
                logger.exception("%s FAILED %s tokens=%d", progress, spec.op_type, spec.num_tokens)
                errors += 1
                continue
            finally:
                del mhc_func
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

    elapsed = time.monotonic() - t_start
    logger.info(
        "Done: %d benchmarked, %d skipped, %d errors in %.1fs",
        len(completed) - skipped,
        skipped,
        errors,
        elapsed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 mHC module collector for Ascend NPU"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./mhc_data",
        help="Output directory for mhc_module_perf.txt and checkpoint",
    )
    parser.add_argument(
        "--op-types", nargs="+", default=[OP_PRE, OP_POST], choices=[OP_PRE, OP_POST],
        help="Which mHC sites to benchmark",
    )
    parser.add_argument(
        "--num-tokens-list", nargs="+", type=int, default=DEFAULT_NUM_TOKENS_LIST,
        help="Token counts to sweep (the only interpolated axis)",
    )
    parser.add_argument(
        "--hc-mult", type=int, default=4,
        help="mHC expansion factor (DeepSeek-V4-Pro: 4)",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=7168,
        help="Model hidden size (DeepSeek-V4-Pro: 7168)",
    )
    parser.add_argument(
        "--sinkhorn-iters", type=int, default=20,
        help="Sinkhorn iterations used by the mixing constraint",
    )
    parser.add_argument(
        "--dtype", default="float16", choices=sorted(DTYPE_MAP),
        help="GEMMQuantMode name written to the output. float16 is measured as bf16.",
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
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

    specs = _build_spec_list(
        op_types=args.op_types,
        num_tokens_list=args.num_tokens_list,
        hc_mult=args.hc_mult,
        hidden_size=args.hidden_size,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    logger.info("Total specs: %d", len(specs))

    run_benchmark(
        specs=specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        resume=args.resume,
        dtype=DTYPE_MAP[args.dtype],
        framework=args.framework,
        version=args.version,
        device=args.device,
    )


if __name__ == "__main__":
    main()
