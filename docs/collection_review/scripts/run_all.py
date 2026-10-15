#!/usr/bin/env python3
"""Run aic-npu default across models x database-mode and capture logs."""
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\aiconfigurator-npu"
PY = r"D:\Production\Misc\Env\MiniConda\miniconda3\python.exe"
OUTDIR = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\staging-1014\1014\runs"
os.makedirs(OUTDIR, exist_ok=True)

MODELS = {
    "R1": "model_configs/deepseek-ai--DeepSeek-R1_config.json",
    "V4": "model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json",
    "Qwen": "model_configs/qwen3_8_27B_config.json",
}
MODES = ["SILICON", "HYBRID", "EMPIRICAL", "SOL"]

COMMON = [
    "--total-gpus", "128", "--system", "ascend_910b", "--backend", "vllm-ascend",
    "--backend-version", "0.23.0", "--isl", "2048", "--osl", "8192",
    "--ttft", "30000", "--tpot", "3000", "--top-n", "10",
]


def run(model_key, mode):
    model = MODELS[model_key]
    save = os.path.join(OUTDIR, f"results_{model_key}_{mode}")
    log = os.path.join(OUTDIR, f"{model_key}_{mode}.log")
    cmd = [PY, "-m", "aiconfigurator_npu.cli.main", "default",
           "--model-path", model, "--database-mode", mode,
           "--save-dir", save] + COMMON
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    return model_key, mode, p.returncode, log


def main():
    jobs = [(mk, m) for mk in MODELS for m in MODES]
    # skip R1 SILICON (already captured)
    jobs = [j for j in jobs if not (j == ("R1", "SILICON"))]
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(run, mk, m) for mk, m in jobs]
        for f in futs:
            results.append(f.result())
    print("DONE")
    for mk, m, rc, log in sorted(results):
        # extract last summary line
        lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
        tail = [l for l in lines if "successful experiment" in l or "blocking" in l.lower()
                or "No results" in l or "model does not fit" in l or "top configurations" in l.lower()]
        print(f"  {mk:5s} {m:10s} rc={rc}  -> {log}")
        for t in tail[-3:]:
            print("       |", t.strip()[:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())
