"""HCCL communication microbenchmark collector for Ascend NPU.

Measures all_reduce / all_gather / reduce_scatter latency across message sizes
and writes CSVs compatible with aiconfigurator-npu's perf_database:

  custom_allreduce_perf.txt : allreduce_dtype, num_gpus, message_size, latency, power
  nccl_perf.txt             : nccl_dtype, op_name, num_gpus, message_size, latency, power

message_size unit: ELEMENT COUNT (fp16 elements), matching query_custom_allreduce /
query_nccl which pass element counts (see "count, not size in bytes" in perf_database).

Usage (single node, 8 cards):
    torchrun --nproc_per_node=8 collector/npu/collect_hccl.py --output-dir ./hccl_data

Usage (multi node, e.g. 2x8):
    torchrun --nnodes=2 --nproc_per_node=8 \
        --master_addr=<ip> --master_port=29500 \
        collector/npu/collect_hccl.py --output-dir ./hccl_data

After collection, copy outputs into the database dir:
    cp hccl_data/custom_allreduce_perf.txt hccl_data/nccl_perf.txt \
       systems/data/ascend_910b/vllm-ascend/0.18.0/
"""

import argparse
import csv
import os

import torch

try:
    import torch_npu  # noqa: F401 — register NPU dispatch
except ImportError:
    pass

import torch.distributed as dist

# Message sizes to sweep, in fp16 ELEMENT counts.
# 1) Grid sweep: 1MB .. 1GB bytes  -> elems = bytes / 2
# 2) Business shapes: bs x hidden for common LLM hidden sizes (fp16 elems)
GRID_BYTES = [1 << 20, 4 << 20, 16 << 20, 64 << 20, 128 << 20, 256 << 20, 512 << 20, 1 << 30]
BIZ_HIDDENS = [4096, 5120, 6144, 7168]
BIZ_BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

WARMUP_ITERS = 10
MEASURE_ITERS = 30

NCCL_OPS = ["all_reduce", "all_gather", "reduce_scatter"]


def build_size_list() -> list[int]:
    sizes = {b // 2 for b in GRID_BYTES}
    for h in BIZ_HIDDENS:
        for bs in BIZ_BS:
            sizes.add(h * bs)
    return sorted(sizes)


def run_op(op: str, tensor: torch.Tensor, group) -> None:
    if op == "all_reduce":
        dist.all_reduce(tensor, group=group)
    elif op == "all_gather":
        out = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(out, tensor, group=group)
    elif op == "reduce_scatter":
        chunk = tensor.numel() // dist.get_world_size(group)
        inp = tensor.reshape(dist.get_world_size(group), chunk)
        out = torch.empty(chunk, dtype=tensor.dtype, device=tensor.device)
        dist.reduce_scatter(out, inp, group=group)
    else:
        raise ValueError(f"unknown op {op}")


def bench_op(op: str, num_elems: int, iters: int = MEASURE_ITERS) -> float:
    """Return average latency in milliseconds via NPU event timing."""
    world = dist.get_world_size()
    if op == "reduce_scatter" and num_elems % world != 0:
        num_elems = (num_elems // world + 1) * world  # must divide evenly

    tensor = torch.randn(num_elems, dtype=torch.float16, device="npu")
    for _ in range(WARMUP_ITERS):
        run_op(op, tensor, None)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_op(op, tensor, None)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="HCCL comm microbenchmark for aiconfigurator-npu")
    parser.add_argument("--output-dir", default="./hccl_data", help="output directory for the CSVs")
    parser.add_argument("--ops", nargs="+", default=NCCL_OPS, choices=NCCL_OPS)
    parser.add_argument("--iters", type=int, default=MEASURE_ITERS)
    parser.add_argument("--min-bytes", type=int, default=1 << 20, help="skip grid sizes below this (bytes)")
    args = parser.parse_args()

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank % torch.npu.device_count())

    sizes = [s for s in build_size_list() if s * 2 >= args.min_bytes]
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"world_size={world}, {len(sizes)} message sizes, ops={args.ops}")

    ar_rows, nccl_rows = [], []
    for op in args.ops:
        for size in sizes:
            try:
                latency_ms = bench_op(op, size, iters=args.iters)
            except Exception as exc:  # noqa: BLE001 — keep sweeping remaining sizes
                if rank == 0:
                    print(f"[skip] op={op} size={size}: {exc}")
                continue
            if rank == 0:
                print(f"op={op} world={world} elems={size} ({size * 2 / 2**20:.1f} MiB): {latency_ms:.3f} ms")
                # power: optional npu-smi sampling; 0.0 keeps legacy-compatible output
                nccl_rows.append(["half", op, world, size, f"{latency_ms:.6f}", "0.0"])
                if op == "all_reduce":
                    ar_rows.append(["half", world, size, f"{latency_ms:.6f}", "0.0"])

    if rank == 0:
        with open(os.path.join(args.output_dir, "nccl_perf.txt"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["nccl_dtype", "op_name", "num_gpus", "message_size", "latency", "power"])
            w.writerows(nccl_rows)
        with open(os.path.join(args.output_dir, "custom_allreduce_perf.txt"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["allreduce_dtype", "num_gpus", "message_size", "latency", "power"])
            w.writerows(ar_rows)
        print(f"Saved {len(nccl_rows)} nccl rows / {len(ar_rows)} allreduce rows to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
