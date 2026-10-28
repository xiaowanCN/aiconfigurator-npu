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

from bench_engine import benchmark_npu, describe_mode
from dsv4_factory import (
    A5_SPARSE_NUM_HEADS,
    CSA,
    HCA,
    OP_CONTEXT,
    OP_GENERATION,
    PRO_DEFAULTS,
    Dsv4ModuleSpec,
    _is_a5_device,
    _spec_key,
    create_dsv4_module_func,
)

logger = logging.getLogger(__name__)

# Prefill: keep batch x seq bounded — the reference module materialises the
# full [T, H, K] score tensor.
DEFAULT_CONTEXT_BATCH_LIST = [1, 2, 4, 8, 16, 32]
DEFAULT_CONTEXT_SEQ_LIST = [128, 256, 512, 1024, 2048, 4096]
# Decode: one query token per request, so batch can go much higher.
# Must reach 512: agg sweeps up to max_batch_size=512 and disagg up to
# decode_max_num_tokens=512 (pareto_analysis.py / inference_session.py). Anything
# above the largest collected batch gets clamped to the grid edge by
# PerfDatabase._nearest_1d_point_helper, i.e. silently constant-extrapolated.
DEFAULT_GENERATION_BATCH_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512]
DEFAULT_GENERATION_SEQ_LIST = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# num_heads = num_attention_heads(128) // tp_size. tp is NOT capped at 8:
# TaskConfig._widen_gpu_sweep_lists (sdk/task.py) extends tp_list with 16/32/64/...
# up to --total-gpus, because a <=8-GPU worker cannot hold DeepSeek-V4-Pro
# (~2.9 TB bf16).
#
# IMPORTANT (Ascend 910B hardware constraint): the SparseAttnSharedkv kernel
# requires q_head_num to be a multiple of 4. num_heads=2 -> tp=64 therefore
# fails at the tiling stage ("head num should be multiple of 4, but got 2"),
# so the smallest rank-local head count we can collect is 4 (=> max tp=32).
DEFAULT_NUM_HEADS_LIST = [128, 64, 32, 16, 8, 4]

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
    timing_mode: str = "auto",
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

            forward_fn = None
            try:
                forward_fn, _meta = create_dsv4_module_func(spec, device="npu:0")
                result = benchmark_npu(
                    forward_fn,
                    warmup_iters=warmup_iters,
                    num_runs=bench_iters,
                    timing_mode=timing_mode,
                )
                writer.writerow(
                    _make_row(spec, result, framework, version, device, mla_dtype, kv_cache_dtype, gemm_type)
                )
                logger.info(
                    "%s -> %.2f us (avg of %d runs, %s)",
                    progress, result.avg_us, result.num_runs, describe_mode(result),
                )
            except Exception:
                logger.exception(
                    "%s FAILED %s b=%d s=%d heads=%d",
                    progress, spec.attn_kind, spec.batch, spec.seq_len, spec.num_heads,
                )
                errors += 1
                continue
            finally:
                # NPU kernels report failures asynchronously: the host-side
                # call returns fine and the error only shows up at the next
                # stream sync. Silently swallowing it here used to hide the
                # *real* first failure, so every later spec was then blamed for
                # "ACL stream synchronize failed, error code:507018" and the
                # run died in torch.npu.empty_cache() with no useful context.
                # Log it loudly (the failing spec above is the suspect) and
                # keep the sweep going.
                try:
                    torch.npu.synchronize()
                except RuntimeError as e:
                    logger.error(
                        "NPU synchronize failed after %s b=%d s=%d heads=%d: %s "
                        "-- the device is now in an error state; every later "
                        "spec will fail for THIS reason (507018), not its own.",
                        spec.attn_kind, spec.batch, spec.seq_len, spec.num_heads, e,
                    )
                # Release the vLLM forward context kept open by the framework
                # path (the reference path has no such closer).
                close = getattr(forward_fn, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        logger.debug("closing forward context failed", exc_info=True)
                try:
                    torch.npu.empty_cache()
                except RuntimeError as e:
                    logger.error("npu empty_cache failed: %s", e)

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
    p.add_argument(
        "--no-graph", action="store_true",
        help=(
            "Skip NPU-graph capture and time the eager path only. This is an "
            "escape hatch: graph mode is the DEFAULT and works for "
            "--module-source framework because the vLLM forward context is built "
            "once OUTSIDE the timed closure (a host->device copy during capture "
            "is illegal: aclrtMemcpy error 107030). Use --no-graph only if graph "
            "capture still fails for a specific shape."
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--timing-mode", default="auto", choices=["auto", "graph", "eager"],
        help="Latency source: auto = min(graph, eager); "
             "graph = prefer NPU graph, fall back to eager on capture failure; "
             "eager = eager-only (no graph capture).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--framework", default="vllm-ascend")
    p.add_argument("--version", default="0.23.0")
    p.add_argument("--device", default="Ascend 910B")
    return p.parse_args()


def _validate_num_heads_list(num_heads_list: list[int]) -> list[int]:
    """Drop head counts the Ascend attention kernels cannot run.

    910B (A2/A3)
        The SparseAttnSharedkv kernel requires ``q_head_num`` to be a multiple
        of 4, so ``num_heads=2`` (i.e. tp=64) fails at the tiling stage
        ("head num should be multiple of 4, but got 2").

    950DT (A5)
        The kv-quant sparse-attention metadata / forward kernels hard-check the
        query head count and accept **only 64 or 128**
        (``KvQuantSparseAttnSharedkvMetadataCpuKernel::CheckSingleParam`` ->
        "num_heads_q should be 64 or 128"; the forward op's
        ``CheckSingleParaNumHeads`` has the same check). The failure is
        *asynchronous*: the host-side build looks fine and the error only
        surfaces at the next stream sync as

            E39999 ... kernelName=KvQuantSparseAttnSharedkvMetadata,
            errorCode=0x2a ... rtStreamSynchronize execution failed

        which then poisons the device for every later spec (all of them report
        "ACL stream synchronize failed, error code:507018"). So these points
        must never be queued on A5 -- drop them up front.

    Warn and skip the unsupported counts so a full sweep does not abort on
    every such point.
    """
    kept, dropped = [], []
    for h in num_heads_list:
        if h % 4 == 0:
            kept.append(h)
        else:
            dropped.append(h)
    if dropped:
        logger.warning(
            "Dropping num_heads=%s: the Ascend 910B attention kernel requires "
            "q_head_num to be a multiple of 4 (num_heads=2 => tp=64 is "
            "unsupported). Collecting only %s.",
            dropped, kept,
        )

    if _is_a5_device():
        a5_kept = [h for h in kept if h in A5_SPARSE_NUM_HEADS]
        a5_dropped = [h for h in kept if h not in A5_SPARSE_NUM_HEADS]
        if a5_dropped:
            logger.warning(
                "Dropping num_heads=%s: on Ascend 950DT the kv-quant sparse "
                "attention kernels only accept num_heads_q in %s "
                "(num_heads = num_attention_heads // tp_size, so only tp=1/2 "
                "are measurable). Those tp configs fall back to SOL.",
                a5_dropped, list(A5_SPARSE_NUM_HEADS),
            )
        return a5_kept

    return kept


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

    args.num_heads_list = _validate_num_heads_list(args.num_heads_list)

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
        timing_mode="eager" if args.no_graph else args.timing_mode,
    )


if __name__ == "__main__":
    main()
