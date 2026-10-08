#!/usr/bin/env python3
"""
parse_hccl_test_logs.py —— 解析 nccl-tests/hccl-test 系日志 → 框架 CSV

用法:
  python tools/parse_hccl_test_logs.py \
      --input /mnt/caikaiwei/projects/aic-collect-hccl/raw_comm \
      --output ./hccl_data_0923

支持三类日志:
  1. HCCL 管道分隔表格（本仓库采集脚本实际输出）:
       524288        | 258.65    | 2.02703    | NULL
  2. nccl-tests 空格分隔表格:
       1048576    524288    float16     sum     322.72    ...
  3. CSV 输出（size_bytes,latency_us）
"""

import argparse
import csv
import os
import re

# 框架字段
NCCL_HEADER = ["nccl_dtype", "op_name", "num_gpus", "message_size", "latency", "power"]
CAR_HEADER = ["allreduce_dtype", "num_gpus", "message_size", "latency", "power"]

DTYPE_BYTES = {"half": 2, "bf16": 2, "fp8": 1, "int8": 1}

# 文件名约定: {op}_np{N}_{seg}.txt / .log / .csv
LOG_PATTERN = re.compile(
    r"(?P<op>all_reduce|all_gather|reduce_scatter|alltoall)_np(?P<np>\d+)",
    re.IGNORECASE,
)

# 格式 1: HCCL 管道行  data_size(Bytes) | avg_time(us) | alg_bandwidth | check
PIPE_ROW_PATTERN = re.compile(
    r"^\s*(?P<size>\d+)\s*\|\s*(?P<time_us>[\d.]+)\s*\|"
)

# 格式 2: nccl-tests 表格行  size count type red time_us algbw busbw ...
ROW_PATTERN = re.compile(
    r"^\s*(?P<size>\d+)\s+(?P<count>\d+)\s+\S+\s+\S+\s+(?P<time_us>[\d.]+)"
)


def bytes_to_message_size(op: str, size_bytes: int, np_: int, dtype: str) -> int:
    """按原语语义把工具输出的字节数转换为框架 message_size（完整张量元素数）。

    换算表见 docs/HCCL_COMM_PERF_DATA_COLLECTION_GUIDE.md §7:
      - all_reduce     : data_size = 完整张量字节 → / dtype_bytes
      - all_gather     : data_size = 每 rank 输入字节 → * np / dtype_bytes
      - reduce_scatter : data_size 按完整输入字节处理 → / dtype_bytes
                         （若所用工具表示"每 rank 输出"，需改为 * np）
      - alltoall       : data_size = 每 rank 字节 → * np / dtype_bytes
    """
    elem = DTYPE_BYTES[dtype]
    if op in ("all_gather", "alltoall"):
        return (size_bytes * np_) // elem
    return size_bytes // elem


def count_to_message_size(op: str, count: int, np_: int) -> int:
    """nccl-tests 的 count（已是元素数）→ 框架 message_size。"""
    if op in ("all_gather", "alltoall"):
        return count * np_
    # all_reduce: 直用; reduce_scatter: 按"完整输入"语义直用
    return count


def parse_log(path: str, dtype: str = "half"):
    """返回 (op, num_gpus, [(message_size_elements, latency_ms), ...])"""
    m = LOG_PATTERN.search(os.path.basename(path))
    if not m:
        return None, None, []
    op = m.group("op").lower()
    np_ = int(m.group("np"))

    results = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            # --- 格式 1: HCCL 管道分隔 ---
            pm = PIPE_ROW_PATTERN.match(line)
            if pm:
                size_bytes = int(pm.group("size"))
                time_ms = float(pm.group("time_us")) / 1000.0
                results.append((bytes_to_message_size(op, size_bytes, np_, dtype), time_ms))
                continue

            # --- 格式 2: nccl-tests 空格表格 ---
            rm = ROW_PATTERN.match(line)
            if rm:
                count = int(rm.group("count"))
                time_ms = float(rm.group("time_us")) / 1000.0
                results.append((count_to_message_size(op, count, np_), time_ms))
                continue

            # --- 格式 3: CSV 回退 size_bytes,latency_us ---
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    size_b = int(float(parts[0]))
                    t_ms = float(parts[1]) / 1000.0
                    results.append((bytes_to_message_size(op, size_b, np_, dtype), t_ms))
                except ValueError:
                    pass
    return op, np_, results


def convert(input_dir: str, output_dir: str, dtype: str = "half") -> None:
    os.makedirs(output_dir, exist_ok=True)
    nccl_rows, car_rows = [], []
    parsed_files = 0
    skipped = []

    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith((".log", ".txt", ".csv")):
            continue

        op, np_, points = parse_log(path, dtype)
        if op is None:
            skipped.append(f"{name} (文件名不符合 {{op}}_np{{N}} 约定)")
            continue
        if not points:
            skipped.append(f"{name} (未解析到数据行)")
            continue

        parsed_files += 1
        print(f"parsed {name}: op={op} np={np_} points={len(points)}")

        for message_size, latency_ms in points:
            nccl_rows.append({
                "nccl_dtype": dtype,
                "op_name": op,
                "num_gpus": np_,
                "message_size": message_size,
                "latency": f"{latency_ms:.6f}",
                "power": "0.0",
            })
            if op == "all_reduce":
                car_rows.append({
                    "allreduce_dtype": dtype,
                    "num_gpus": np_,
                    "message_size": message_size,
                    "latency": f"{latency_ms:.6f}",
                    "power": "0.0",
                })

    if parsed_files == 0:
        print("ERROR: 未解析到任何有效日志文件，请检查 --input 目录与文件名约定")
        for s in skipped:
            print(f"  skipped: {s}")
        raise SystemExit(1)

    with open(os.path.join(output_dir, "nccl_perf.txt"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NCCL_HEADER)
        w.writeheader()
        w.writerows(nccl_rows)

    with open(os.path.join(output_dir, "custom_allreduce_perf.txt"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAR_HEADER)
        w.writeheader()
        w.writerows(car_rows)

    print(f"\nparsed files: {parsed_files}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"  skipped: {s}")
    print(f"nccl_perf.txt: {len(nccl_rows)} rows")
    print(f"custom_allreduce_perf.txt: {len(car_rows)} rows")
    print(f"output dir: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="解析 hccl-test/nccl-tests 日志为框架 CSV")
    ap.add_argument("--input", required=True, help="原始日志目录")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--dtype", default="half", choices=list(DTYPE_BYTES.keys()),
                    help="数据类型（用于字节→元素换算），默认 half")
    args = ap.parse_args()
    convert(args.input, args.output, args.dtype)

