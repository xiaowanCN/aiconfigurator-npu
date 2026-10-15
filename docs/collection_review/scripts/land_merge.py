#!/usr/bin/env python3
"""Back up 0.23.0 then merge staging/out tables into it (documented landing step)."""
import os
import shutil
import subprocess
import sys
from datetime import datetime

REPO = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\aiconfigurator-npu"
DATA_DIR = os.path.join(REPO, "src", "aiconfigurator_npu", "systems", "data", "ascend_910b", "vllm-ascend", "0.23.0")
STAGING_OUT = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\staging-1014\1014\staging\out"
MERGE = os.path.join(REPO, "tools", "merge_perf_tables.py")
PY = r"D:\Production\Misc\Env\MiniConda\miniconda3\python.exe"

TABLES = [
    "gemm_perf", "moe_perf",
    "context_mla_perf", "generation_mla_perf", "mla_bmm_perf",
    "dsv4_csa_context_module_perf", "dsv4_csa_generation_module_perf",
    "dsv4_hca_context_module_perf", "dsv4_hca_generation_module_perf",
    "mhc_module_perf", "context_attention_perf", "generation_attention_perf", "gdn_perf",
]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(os.path.dirname(DATA_DIR), "0.23.0.bak." + ts)
    print(f"Backing up {DATA_DIR} -> {bak}")
    shutil.copytree(DATA_DIR, bak)
    print("Backup done.\n")

    for t in TABLES:
        src = os.path.join(STAGING_OUT, t + ".txt")
        dst = os.path.join(DATA_DIR, t + ".txt")
        if not os.path.isfile(src):
            print(f"  skip (no staging): {t}")
            continue
        cmd = [PY, MERGE, dst, src, "--in-place"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  merge {t}: rc={r.returncode} {r.stderr.strip() or r.stdout.strip()}")
    print("\nLanding complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
