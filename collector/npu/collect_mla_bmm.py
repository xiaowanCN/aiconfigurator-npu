"""MLA batched-projection (BMM) microbenchmark collector for Ascend NPU (CANN + vLLM Ascend).

Collects the two non-absorbed MLA decode projections that aiconfigurator models
as ``MLABmm`` (see ``sdk/models.py`` -- DeepSeekModel's generation path):

    mla_gen_pre   kv_c     @ W_uk :  [T, kv_lora_rank] @ [H, kv_lora_rank, head_dim] -> [H, T, head_dim]
    mla_gen_post  attn_out @ W_uv :  [H, T, head_dim]  @ [H, head_dim, kv_lora_rank] -> [H, T, kv_lora_rank]

For DeepSeek-V3 / R1: kv_lora_rank=512, head_dim=128, H = 128 // tp_size.

Output:
    mla_bmm_perf.txt -- consumed by perf_database.load_mla_bmm_data(), indexed as
        data[GEMMQuantMode][op_name][num_heads][num_tokens]
    ``num_tokens`` is the decode batch (concurrency) and is the only axis that
    gets interpolated; ``num_heads`` must be an exact key.

Without this table, SILICON mode raises (``raise_if_not_loaded``) and HYBRID
falls back to ``SOL / 0.8``.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from bench_engine import BenchResult, benchmark_npu
from gemm_factory import _init_vllm_context

logger = logging.getLogger(__name__)

# --- Op identifiers (must match perf_database.query_mla_bmm) ---
OP_PRE = "mla_gen_pre"
OP_POST = "mla_gen_post"
SUPPORTED_OP_TYPES = (OP_PRE, OP_POST)

# Decode concurrency that aiconfigurator interpolates over.
DEFAULT_NUM_TOKENS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# 128 // tp_size for tp in {1, 2, 4, 8}
DEFAULT_NUM_HEADS_LIST = [128, 64, 32, 16]

OUTPUT_COLUMNS = [
    "framework", "version", "device", "op_name", "bmm_dtype",
    "num_heads", "num_tokens", "latency",
]

CHECKPOINT_FILE = "mla_bmm_checkpoint.json"

# GEMMQuantMode.float16 is benchmarked as bf16 on Ascend, matching the other
# collectors in this repo (see convert_to_aiconfigurator's bf16 -> float16 map).
DTYPE_MAP = {"float16": torch.bfloat16}


@dataclass(frozen=True)
class BmmSpec:
    """Immutable MLA BMM benchmark specification (DeepSeek-V3 / R1 defaults)."""

    op_type: str
    num_tokens: int
    num_heads: int
    kv_lora_rank: int = 512
    head_dim: int = 128


def _spec_key(spec: BmmSpec) -> str:
    return f"{spec.op_type}_{spec.num_tokens}_{spec.num_heads}_{spec.kv_lora_rank}_{spec.head_dim}"


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


def create_bmm_func(
    spec: BmmSpec, device: torch.device, dtype: torch.dtype
) -> Callable[[], torch.Tensor]:
    """Build the batched projection forward for one spec.

    The broadcast of ``kv_c`` (a single [T, kv_lora_rank] latent) to
    ``[num_heads, T, kv_lora_rank]`` is pre-materialised so the timed region
    contains only the batched GEMM itself.
    """
    t = spec.num_tokens
    h = spec.num_heads
    kv = spec.kv_lora_rank
    d = spec.head_dim

    if spec.op_type == OP_PRE:
        # [H, T, kv_lora_rank] @ [H, kv_lora_rank, head_dim] -> [H, T, head_dim]
        kv_c = torch.randn(h, t, kv, dtype=dtype, device=device)
        w_uk = torch.randn(h, kv, d, dtype=dtype, device=device)

        def forward() -> torch.Tensor:
            return torch.bmm(kv_c, w_uk)
    else:
        # [H, T, head_dim] @ [H, head_dim, kv_lora_rank] -> [H, T, kv_lora_rank]
        attn_out = torch.randn(h, t, d, dtype=dtype, device=device)
        w_uv = torch.randn(h, d, kv, dtype=dtype, device=device)

        def forward() -> torch.Tensor:
            return torch.bmm(attn_out, w_uv)

    # Fail fast on shape / tiling errors before the timed loop.
    forward()
    torch.npu.synchronize()
    return forward


def _make_row(
    spec: BmmSpec,
    result: BenchResult,
    framework: str,
    version: str,
    device: str,
    bmm_dtype: str,
) -> dict[str, str]:
    return {
        "framework": framework,
        "version": version,
        "device": device,
        "op_name": spec.op_type,
        "bmm_dtype": bmm_dtype,
        "num_heads": str(spec.num_heads),
        "num_tokens": str(spec.num_tokens),
        "latency": f"{result.avg_us / 1000.0:.6f}",
    }


def _build_spec_list(
    op_types: list[str],
    num_tokens_list: list[int],
    num_heads_list: list[int],
    kv_lora_rank: int,
    head_dim: int,
) -> list[BmmSpec]:
    specs: list[BmmSpec] = []
    for op_type in op_types:
        for num_tokens in num_tokens_list:
            for num_heads in num_heads_list:
                specs.append(
                    BmmSpec(
                        op_type=op_type,
                        num_tokens=num_tokens,
                        num_heads=num_heads,
                        kv_lora_rank=kv_lora_rank,
                        head_dim=head_dim,
                    )
                )
    return specs


def run_benchmark(
    specs: list[BmmSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
    dtype: torch.dtype,
    framework: str = "vllm-ascend",
    version: str = "0.23.0",
    device: str = "Ascend 910B",
    bmm_dtype: str = "float16",
) -> None:
    npu_device = torch.device("npu")
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    csv_path = output_dir / "mla_bmm_perf.txt"
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
            logger.info(
                "%s %s tokens=%d heads=%d", progress, spec.op_type, spec.num_tokens, spec.num_heads
            )

            try:
                bmm_func = create_bmm_func(spec, npu_device, dtype)
                result = benchmark_npu(
                    bmm_func,
                    warmup_iters=warmup_iters,
                    num_runs=bench_iters,
                )
                writer.writerow(
                    _make_row(spec, result, framework, version, device, bmm_dtype)
                )
                logger.info("%s -> %.2f us (avg of %d runs)", progress, result.avg_us, result.num_runs)
            except Exception:
                logger.exception(
                    "%s FAILED %s tokens=%d heads=%d",
                    progress, spec.op_type, spec.num_tokens, spec.num_heads,
                )
                errors += 1
                continue
            finally:
                try:
                    del bmm_func
                except NameError:
                    pass
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
        len(completed) - skipped, skipped, errors, elapsed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLA batched-projection (BMM) microbenchmark collector for Ascend NPU"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./mla_bmm_data",
        help="Output directory for mla_bmm_perf.txt and checkpoint",
    )
    parser.add_argument(
        "--op-types", nargs="+", default=["pre", "post"], choices=["pre", "post"],
        help="Which projections to benchmark",
    )
    parser.add_argument(
        "--num-tokens-list", nargs="+", type=int, default=DEFAULT_NUM_TOKENS_LIST,
        help="Decode batch sizes to sweep (the interpolated axis)",
    )
    parser.add_argument(
        "--num-heads-list", nargs="+", type=int, default=DEFAULT_NUM_HEADS_LIST,
        help="128 // tp_size values to sweep (exact keys, not interpolated)",
    )
    parser.add_argument(
        "--kv-lora-rank", type=int, default=512,
        help="MLA KV latent rank (DeepSeek-V3/R1: 512)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=128,
        help="Per-head nope dim / v_head_dim (DeepSeek-V3/R1: 128)",
    )
    parser.add_argument(
        "--bmm-dtype", default="float16", choices=sorted(DTYPE_MAP),
        help="bmm_dtype written to the output (GEMMQuantMode name). float16 is measured as bf16.",
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

    logger.info("Initializing vLLM context...")
    _init_vllm_context()

    op_type_map = {"pre": OP_PRE, "post": OP_POST}
    op_types = [op_type_map[o] for o in args.op_types]

    specs = _build_spec_list(
        op_types=op_types,
        num_tokens_list=args.num_tokens_list,
        num_heads_list=args.num_heads_list,
        kv_lora_rank=args.kv_lora_rank,
        head_dim=args.head_dim,
    )
    logger.info("Total specs: %d", len(specs))

    run_benchmark(
        specs=specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        resume=args.resume,
        dtype=DTYPE_MAP[args.bmm_dtype],
        framework=args.framework,
        version=args.version,
        device=args.device,
        bmm_dtype=args.bmm_dtype,
    )


if __name__ == "__main__":
    main()
