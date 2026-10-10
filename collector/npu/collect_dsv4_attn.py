"""DeepSeek-V4 (DSv4) module-level attention collector for Ascend NPU.

Produces the four tables consumed by
``perf_database.load_context_dsv4_module_data()`` /
``load_generation_dsv4_module_data()``::

    dsv4_csa_context_module_perf.txt     data[fmha][kv][gemm][num_heads][s][b]
    dsv4_hca_context_module_perf.txt     (same)
    dsv4_csa_generation_module_perf.txt  data[kv][gemm][num_heads][b][s]
    dsv4_hca_generation_module_perf.txt  (same, s = isl + step)

Context and generation must be collected in separate output directories: the
checkpoint file name is shared and one directory holds a single sweep.

Dtype columns use the enum-key names the loaders expect:
``mla_dtype`` -> FMHAQuantMode, ``kv_cache_dtype`` -> KVCacheQuantMode,
``gemm_type`` -> GEMMQuantMode (``bfloat16`` is normalised to ``float16``).

Without these tables SILICON mode raises and HYBRID falls back to ``SOL / 0.5``
for the DSv4 attention ops.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from bench_engine import benchmark_npu
from dsv4_factory import (
    CSA,
    HCA,
    OP_CONTEXT,
    OP_GENERATION,
    PRO_DEFAULTS,
    Dsv4ModuleSpec,
    _spec_key,
    create_dsv4_module_func,
)

logger = logging.getLogger(__name__)

# Prefill: keep batch x seq bounded — the reference module materialises the
# full [T, H, K] score tensor.
DEFAULT_CONTEXT_BATCH_LIST = [1, 2, 4, 8, 16, 32]
DEFAULT_CONTEXT_SEQ_LIST = [128, 256, 512, 1024, 2048, 4096]
# Decode: one query token per request, so batch can go much higher.
DEFAULT_GENERATION_BATCH_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_GENERATION_SEQ_LIST = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# 128 // tp_size for tp in {1, 2, 4, 8}
DEFAULT_NUM_HEADS_LIST = [128, 64, 32, 16]

CONTEXT_COLUMNS = [
    "framework", "version", "device", "op_name", "mla_dtype", "kv_cache_dtype",
    "gemm_type", "num_heads", "batch_size", "isl", "latency",
]
GENERATION_COLUMNS = [
    "framework", "version", "device", "op_name", "kv_cache_dtype",
    "gemm_type", "num_heads", "batch_size", "isl", "step", "latency",
]

CHECKPOINT_FILE = "dsv4_attn_checkpoint.json"

DTYPE_MAP = {"float16": torch.bfloat16}


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


def _output_filename(op_type: str, attn_kind: str) -> str:
    return f"dsv4_{attn_kind}_{'context' if op_type == OP_CONTEXT else 'generation'}_module_perf.txt"


def _make_row(
    spec: Dsv4ModuleSpec,
    result,
    framework: str,
    version: str,
    device: str,
    mla_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
) -> dict[str, str]:
    latency_ms = f"{result.avg_us / 1000.0:.6f}"
    if spec.op_type == OP_CONTEXT:
        return {
            "framework": framework,
            "version": version,
            "device": device,
            "op_name": f"dsv4_{spec.attn_kind}_context",
            "mla_dtype": mla_dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "gemm_type": gemm_type,
            "num_heads": str(spec.num_heads),
            "batch_size": str(spec.batch),
            "isl": str(spec.seq_len),
            "latency": latency_ms,
        }
    return {
        "framework": framework,
        "version": version,
        "device": device,
        "op_name": f"dsv4_{spec.attn_kind}_generation",
        "kv_cache_dtype": kv_cache_dtype,
        "gemm_type": gemm_type,
        "num_heads": str(spec.num_heads),
        "batch_size": str(spec.batch),
        "isl": str(spec.seq_len),
        "step": "0",  # decode: s = isl + step, step 0 means "KV length == isl"
        "latency": latency_ms,
    }


def run_benchmark(
    specs: list[Dsv4ModuleSpec],
    output_dir: Path,
    warmup_iters: int,
    bench_iters: int,
    resume: bool,
    framework: str,
    version: str,
    device: str,
    mla_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = _load_checkpoint(output_dir) if resume else set()
    if resume and completed:
        logger.info("Resuming: %d specs already completed", len(completed))

    # One file per (attn_kind, phase); a sweep is expected to cover a single
    # combination, so pick the file from the first spec and assert homogeneity.
    kinds = {(s.op_type, s.attn_kind) for s in specs}
    if len(kinds) != 1:
        raise ValueError(
            f"All specs must share one (op_type, attn_kind); got {sorted(kinds)}. "
            "Run each combination into its own --output-dir."
        )
    first = specs[0]
    csv_path = output_dir / _output_filename(first.op_type, first.attn_kind)
    columns = CONTEXT_COLUMNS if first.op_type == OP_CONTEXT else GENERATION_COLUMNS

    file_exists = csv_path.exists() and resume
    fh = open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=columns)
    if not file_exists:
        writer.writeheader()

    total = len(specs)
    skipped = errors = 0
    t_start = time.monotonic()

    try:
        for i, spec in enumerate(specs):
            key = _spec_key(spec)
            if key in completed:
                skipped += 1
                continue

            progress = f"[{i + 1}/{total}]"
            logger.info(
                "%s %s b=%d s=%d heads=%d", progress, spec.attn_kind, spec.batch, spec.seq_len, spec.num_heads
            )

            try:
                forward_fn, _meta = create_dsv4_module_func(spec, device="npu:0")
                result = benchmark_npu(forward_fn, warmup_iters=warmup_iters, num_runs=bench_iters)
                writer.writerow(
                    _make_row(spec, result, framework, version, device, mla_dtype, kv_cache_dtype, gemm_type)
                )
                logger.info("%s -> %.2f us (avg of %d runs)", progress, result.avg_us, result.num_runs)
            except Exception:
                logger.exception(
                    "%s FAILED %s b=%d s=%d heads=%d",
                    progress, spec.attn_kind, spec.batch, spec.seq_len, spec.num_heads,
                )
                errors += 1
                continue
            finally:
                try:
                    torch.npu.synchronize()
                except RuntimeError:
                    pass
                torch.npu.empty_cache()

            completed.add(key)
            if len(completed) % 5 == 0:
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
    p = argparse.ArgumentParser(description="DeepSeek-V4 attention module collector for Ascend NPU")
    p.add_argument("--output-dir", type=str, default="./dsv4_attn_data")
    p.add_argument("--op-types", nargs="+", default=[OP_CONTEXT], choices=[OP_CONTEXT, OP_GENERATION])
    p.add_argument(
        "--attn-kinds", nargs="+", default=[CSA, HCA], choices=[CSA, HCA],
        help="CSA = compress_ratio 4, HCA = compress_ratio 128 (SWA folds into HCA)",
    )
    p.add_argument("--batch-list", nargs="+", type=int, default=None)
    p.add_argument("--seq-len-list", nargs="+", type=int, default=None)
    p.add_argument("--num-heads-list", nargs="+", type=int, default=DEFAULT_NUM_HEADS_LIST)
    p.add_argument("--hidden-size", type=int, default=PRO_DEFAULTS["hidden_size"])
    p.add_argument("--q-lora-rank", type=int, default=PRO_DEFAULTS["q_lora_rank"])
    p.add_argument("--o-lora-rank", type=int, default=PRO_DEFAULTS["o_lora_rank"])
    p.add_argument("--o-groups", type=int, default=PRO_DEFAULTS["o_groups"])
    p.add_argument("--head-dim", type=int, default=PRO_DEFAULTS["head_dim"])
    p.add_argument("--qk-rope-head-dim", type=int, default=PRO_DEFAULTS["qk_rope_head_dim"])
    p.add_argument("--index-n-heads", type=int, default=PRO_DEFAULTS["index_n_heads"])
    p.add_argument("--index-head-dim", type=int, default=PRO_DEFAULTS["index_head_dim"])
    p.add_argument("--index-topk", type=int, default=PRO_DEFAULTS["index_topk"])
    p.add_argument("--sliding-window", type=int, default=PRO_DEFAULTS["sliding_window"])
    p.add_argument(
        "--module-source", default="framework", choices=["framework", "reference"],
        help="framework = vllm-ascend's own module; reference = self-contained torch impl",
    )
    p.add_argument(
        "--attention-cls", default=None,
        help="Override module discovery, e.g. vllm.model_executor.models.deepseek_v4:DeepseekV4Attention",
    )
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    p.add_argument("--mla-dtype", default="float16")
    p.add_argument("--kv-cache-dtype", default="float16")
    p.add_argument("--gemm-type", default="float16")
    p.add_argument("--dtype", default="float16", choices=sorted(DTYPE_MAP))
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--bench-iters", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--framework", default="vllm-ascend")
    p.add_argument("--version", default="0.23.0")
    p.add_argument("--device", default="Ascend 910B")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    is_context = args.op_types == [OP_CONTEXT]
    if args.batch_list is None:
        args.batch_list = DEFAULT_CONTEXT_BATCH_LIST if is_context else DEFAULT_GENERATION_BATCH_LIST
    if args.seq_len_list is None:
        args.seq_len_list = DEFAULT_CONTEXT_SEQ_LIST if is_context else DEFAULT_GENERATION_SEQ_LIST

    specs: list[Dsv4ModuleSpec] = []
    for op_type in args.op_types:
        for kind in args.attn_kinds:
            ratio = 4 if kind == CSA else 128
            for batch in args.batch_list:
                for seq_len in args.seq_len_list:
                    for num_heads in args.num_heads_list:
                        specs.append(
                            Dsv4ModuleSpec(
                                op_type=op_type,
                                attn_kind=kind,
                                compress_ratio=ratio,
                                batch=batch,
                                seq_len=seq_len,
                                num_heads=num_heads,
                                hidden_size=args.hidden_size,
                                q_lora_rank=args.q_lora_rank,
                                o_lora_rank=args.o_lora_rank,
                                o_groups=args.o_groups,
                                head_dim=args.head_dim,
                                qk_rope_head_dim=args.qk_rope_head_dim,
                                index_n_heads=args.index_n_heads,
                                index_head_dim=args.index_head_dim,
                                index_topk=args.index_topk,
                                sliding_window=args.sliding_window,
                                dtype=DTYPE_MAP[args.dtype],
                                module_source=args.module_source,
                                attention_cls=args.attention_cls,
                                model_path=args.model,
                            )
                        )
    logger.info("Total specs: %d", len(specs))

    if len(args.op_types) != 1 or len(args.attn_kinds) != 1:
        raise SystemExit(
            "Run one (op_type, attn_kind) per invocation into its own --output-dir; "
            "the CSV file name is derived from that combination."
        )

    run_benchmark(
        specs=specs,
        output_dir=Path(args.output_dir),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        resume=args.resume,
        framework=args.framework,
        version=args.version,
        device=args.device,
        mla_dtype=args.mla_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        gemm_type=args.gemm_type,
    )


if __name__ == "__main__":
    main()
