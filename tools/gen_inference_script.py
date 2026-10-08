#!/usr/bin/env python3
"""Generate vllm-ascend deployment + benchmark scripts from aiconfigurator-npu predictions.

Reads a search result's ``best_config_topn.csv`` and emits ready-to-run shell
scripts for real-hardware validation:

  agg   -> N vllm serve replicas (GPU ranges split automatically)
  disagg-> 1 prefill instance + N decode instances (KV connector placeholders)

Usage:
    python tools/gen_inference_script.py \
        --results results/Qwen/Qwen3-32B_..._tpot30_419291 \
        --model Qwen/Qwen3-32B \
        --rank 1 \
        --output-dir ./deploy

Then review deploy/*.sh and run on the 910B node.
"""

import argparse
import csv
import os
from pathlib import Path

# TODO: confirm the exact connector name/params for your vllm-ascend build
# (see vllm_ascend package examples for PD disaggregation in v0.18).
def kv_transfer_config(role: str, port: int, gpu: int) -> str:
    return (
        '{"kv_connector":"AscendKVConnector",'
        f'"kv_role":"{role}","kv_port":{port},'
        f'"kv_connector_extra_params":{{"device_id":{gpu}}}}}'
    )


def gpu_range(start: int, count: int) -> str:
    return str(start) if count == 1 else f"{start}-{start + count - 1}"


def serve_cmd(model: str, tp: int, gpus: str, port: int, max_seqs: int | None,
              role: str | None, kv_port: int, extra: list[str]) -> str:
    first_gpu = int(gpus.split("-")[0])
    cmd = (
        f"ASCEND_RT_VISIBLE_DEVICES={gpus} vllm serve {model} "
        f"--tensor-parallel-size {tp} --port {port} "
        f"--dtype float16 "
    )
    if max_seqs is not None:
        cmd += f"--max-num-seqs {max_seqs} "
    if role is not None:
        cmd += f"--kv-transfer-config '{kv_transfer_config(role, kv_port, first_gpu)}' "
    for e in extra:
        cmd += f"{e} "
    return cmd.strip()


def gen_agg_scripts(row: dict, model: str, out_dir: Path, base_port: int) -> None:
    tp = int(row.get("tp", 1))
    bs = int(row.get("bs", 0))
    total = int(row.get("num_total_gpus", 0))
    replicas = max(1, total // tp)
    isl, osl = int(row.get("isl", 4000)), int(row.get("osl", 500))

    lines = ["#!/bin/bash", "# Aggregated serving: 1 vllm instance per replica", ""]
    for r in range(replicas):
        gpus = gpu_range(r * tp, tp)
        port = base_port + r
        lines.append(f"# replica {r + 1}: GPUs {gpus}, predicted bs={bs}")
        lines.append(serve_cmd(model, tp, gpus, port, bs, None, 0, []) + " &")
        lines.append("")
    lines.append("echo 'All agg replicas launched.'")

    (out_dir / "run_agg.sh").write_text("\n".join(lines), encoding="utf-8")
    gen_bench_script(out_dir, "bench_agg.sh", model, base_port, isl, osl, replicas)


def gen_disagg_scripts(row: dict, model: str, out_dir: Path, base_port: int) -> None:
    p_tp, p_workers = int(row.get("(p)tp", 4)), int(row.get("(p)workers", 1))
    d_tp, d_workers = int(row.get("(d)tp", 4)), int(row.get("(d)workers", 1))
    p_bs, d_bs = int(row.get("(p)bs", 1)), int(row.get("(d)bs", 0))
    isl, osl = int(row.get("isl", 4000)), int(row.get("osl", 500))

    lines = ["#!/bin/bash", "# PD disaggregated serving: prefill instances + decode instances", ""]
    gpu = 0
    port = base_port
    for w in range(p_workers):
        gpus = gpu_range(gpu, p_tp)
        lines.append(f"# prefill worker {w + 1}: GPUs {gpus} (bs={p_bs})")
        lines.append(serve_cmd(model, p_tp, gpus, port, p_bs, "kv_producer", 29000 + w, []) + " &")
        gpu += p_tp
        port += 1
        lines.append("")
    for w in range(d_workers):
        gpus = gpu_range(gpu, d_tp)
        lines.append(f"# decode worker {w + 1}: GPUs {gpus} (bs={d_bs})")
        lines.append(serve_cmd(model, d_tp, gpus, port, d_bs, "kv_consumer", 29100 + w, []) + " &")
        gpu += d_tp
        port += 1
        lines.append("")
    lines.append("echo 'All PD workers launched. Start the router to route P -> D.'")

    (out_dir / "run_disagg.sh").write_text("\n".join(lines), encoding="utf-8")
    gen_bench_script(out_dir, "bench_disagg.sh", model, base_port, isl, osl, 1)


def gen_bench_script(out_dir: Path, name: str, model: str, port: int,
                     isl: int, osl: int, replicas: int) -> None:
    rates = "0.5 1 2 4 8 16"
    lines = [
        "#!/bin/bash",
        "# Benchmark: SLA-aligned random dataset (matches search ISL/OSL)",
        f"# Point --port at the router / one replica. Replicas={replicas}.",
        "",
        "mkdir -p bench_results",
    ]
    for r in rates.split():
        lines += [
            f"vllm bench serve --model {model} --port {port} \\",
            f"  --dataset-name random --random-input-len {isl} --random-output-len {osl} \\",
            f"  --request-rate {r} --num-prompts 200 \\",
            f"  --percentile-metrics ttft,tpot,itl,throughput \\",
            f"  --save-result bench_results/req_rate_{r}.json",
            "",
        ]
    (out_dir / name).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="search results dir containing best_config_topn.csv")
    ap.add_argument("--model", required=True, help="model name/path passed to vllm serve")
    ap.add_argument("--rank", type=int, default=1, help="which predicted config to deploy (1-based)")
    ap.add_argument("--mode", choices=["agg", "disagg", "auto"], default="auto")
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--output-dir", default="./deploy")
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ["agg", "disagg"]:
        csv_path = results_dir / mode / "best_config_topn.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if args.rank > len(rows):
            print(f"[skip] {mode}: only {len(rows)} config(s) available (rank {args.rank} requested)")
            continue
        row = rows[args.rank - 1]
        mode_out = out_dir / mode
        mode_out.mkdir(exist_ok=True)
        if mode == "agg" or args.mode == "agg":
            gen_agg_scripts(row, args.model, mode_out, args.base_port)
        else:
            gen_disagg_scripts(row, args.model, mode_out, args.base_port)
        print(f"[ok] {mode} scripts -> {mode_out}/ (rank {args.rank})")

    print(f"\nReview scripts in {out_dir}/, then run on the 910B node:")
    print("  bash deploy/agg/run_agg.sh && bash deploy/agg/bench_agg.sh")


if __name__ == "__main__":
    main()

