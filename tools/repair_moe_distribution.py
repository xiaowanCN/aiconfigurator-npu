#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add missing MoE ``distribution`` keys to a shipped moe_perf.txt.

Why this exists
---------------
``distribution`` is an EXACT query key in ``PerfDatabase._resolve_moe_distribution``:
it is not interpolated and not fuzzy-matched. Each model family queries only its own
``_power_law_alpha`` key (``sdk/models.py``):

    MOEModel (Qwen2/3-MoE, Mixtral, V2-Lite) -> power_law_1.2
    DeepSeekModel / DeepSeekV32 / DeepSeekV4 -> power_law_1.01

``tools/convert_to_aiconfigurator.py`` defaults to ``--moe-distribution power_law_1.2``,
so a table converted with defaults serves MOEModel only. Every expert shape the
DeepSeek families need then resolves to an EMPTY slice -- the failure documented in
``docs/MODEL_ADAPTATION_AUDIT.md`` section 3-B.

The fix needs NO hardware as long as the raw TensorCast CSVs are still around: the
same measurements are simply re-written under the missing distribution key(s).

What it does
------------
Reads ``GroupedMatmul_MoE_{BF16,W8A8}.csv`` from ``--input-dir``, converts them exactly
the way ``convert_to_aiconfigurator.convert_moe`` does, and appends ONLY the rows whose
key is not already present in the target table. Existing rows are never modified, so
deepseek-v4-pro rows (whose raw CSVs may not live in --input-dir) survive untouched.

Usage
-----
    # dry run: show what would be added
    python tools/repair_moe_distribution.py --input-dir systems/data/ascend_910b/vllm-ascend/0.18.0

    # apply (prints row counts)
    python tools/repair_moe_distribution.py --input-dir <raw csv dir> --apply
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from typing import Iterable
from typing import Iterable

DEFAULT_INPUT_DIR = "systems/data/ascend_910b/vllm-ascend/0.18.0"
DEFAULT_OUTPUT = "src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0/moe_perf.txt"

MOE_QUANT_MAP = {"bf16": "float16", "int8": "int8", "w8a8_dynamic": "int8"}
RAW_CSVS = ("GroupedMatmul_MoE_BF16.csv", "GroupedMatmul_MoE_W8A8.csv")
ALL_DISTRIBUTIONS = ("power_law_1.2", "power_law_1.01")

FIELDNAMES = [
    "framework", "version", "device", "op_name", "kernel_source",
    "moe_dtype", "num_tokens", "hidden_size", "inter_size", "topk",
    "num_experts", "moe_tp_size", "moe_ep_size", "distribution", "latency",
]
# Every column except the measurement itself identifies a cell in the table.
KEY_FIELDS = [f for f in FIELDNAMES if f != "latency"]

DEVICE = "Ascend 910B"
FRAMEWORK = "vllm-ascend"
VERSION = "0.23.0"


def _cells(rows: Iterable[dict]) -> dict[tuple, list[float]]:
    """Group rows into (all dimensions except latency and distribution) -> latencies."""
    out: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in KEY_FIELDS if field != "distribution")
        out.setdefault(key, []).append(float(row["latency"]))
    return out


def check_same_campaign(existing: list[dict], candidates: list[dict], rtol: float) -> tuple[int, int, float]:
    """Detect whether the raw CSVs come from the SAME measurement campaign.

    Each (shape, ep, token) cell holds several repeated measurements. If the raw
    source was produced by a different session than the shipped table, the values
    differ even though the grid matches -- and silently mixing two campaigns into
    one table would fabricate precision nobody measured. Returns how many cells
    overlapped, how many disagreed, and the worst relative gap.
    """
    existing_cells = _cells(existing)
    candidate_cells = _cells(candidates)
    overlap = set(existing_cells) & set(candidate_cells)
    mismatched = 0
    worst = 0.0
    for key in overlap:
        left = sorted(existing_cells[key])
        right = sorted(candidate_cells[key])
        if len(left) != len(right):
            mismatched += 1
            continue
        worst = max(worst, max(abs(a - b) / max(abs(b), 1e-9) for a, b in zip(left, right)))
        if not all(abs(a - b) <= rtol * max(abs(b), 1e-9) for a, b in zip(left, right)):
            mismatched += 1
    return len(overlap), mismatched, worst


def _us_to_ms(value: float) -> float:
    return value / 1000.0


def convert_raw(input_dir: str, distributions: Iterable[str]) -> list[dict]:
    """Convert the raw TensorCast MoE CSVs into moe_perf.txt rows."""
    rows: list[dict] = []
    for filename in RAW_CSVS:
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            print(f"  [skip] {filename} not found")
            continue
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                quant = MOE_QUANT_MAP.get(row.get("Quant Type", "bf16").strip().lower())
                if quant is None:
                    print(f"  [warn] unknown quant '{row.get('Quant Type')}' in {filename}, skipped")
                    continue
                num_experts = int(row["Num Experts"])
                topk = int(row["TopK"])
                ep_size = int(row["EP Size"])
                local_experts = int(row["Local Experts"])
                moe_tp_size = max(1, num_experts // local_experts // ep_size) if local_experts > 0 else 1
                for distribution in distributions:
                    rows.append({
                        "framework": FRAMEWORK,
                        "version": VERSION,
                        "device": DEVICE,
                        "op_name": "moe",
                        "kernel_source": "vllm_ascend_fused_moe",
                        "moe_dtype": quant,
                        "num_tokens": int(row["Num Tokens"]),
                        "hidden_size": int(row["Hidden Size"]),
                        "inter_size": int(row["Intermediate Size"]),
                        "topk": topk,
                        "num_experts": num_experts,
                        "moe_tp_size": moe_tp_size,
                        "moe_ep_size": ep_size,
                        "distribution": distribution,
                        "latency": _us_to_ms(float(row["Average Duration(us)"])),
                    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="directory holding the raw TensorCast MoE CSVs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="target moe_perf.txt")
    parser.add_argument("--apply", action="store_true", help="write the change (default is a dry run)")
    parser.add_argument("--no-backup", action="store_true", help="skip writing <output>.bak")
    parser.add_argument(
        "--force",
        action="store_true",
        help="write even when --input-dir looks like a different measurement campaign (see warning below)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.01,
        help="relative tolerance when comparing repeated measurements of overlapping cells (default 0.01)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"input dir not found: {args.input_dir}")
        return 2
    if not os.path.isfile(args.output):
        print(f"target table not found: {args.output}")
        return 2

    with open(args.output, newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    if not existing:
        print(f"target table is empty: {args.output}")
        return 2

    def _key(row: dict) -> tuple:
        # Values are read from CSV on one side and built as python objects on the
        # other, so normalize to str before comparing: 1 != "1" would otherwise
        # make every row look new.
        return tuple(str(row[field]) for field in KEY_FIELDS)

    existing_keys = {_key(r) for r in existing}
    candidates = convert_raw(args.input_dir, ALL_DISTRIBUTIONS)
    missing = [r for r in candidates if _key(r) not in existing_keys]

    print(f"target      : {args.output} ({len(existing)} rows, {len(existing_keys)} unique keys)")
    print(f"raw source  : {args.input_dir}")
    print(f"converted   : {len(candidates)} rows across {ALL_DISTRIBUTIONS}")
    print(f"missing keys: {len(missing)} rows would be added")

    if not missing:
        print("nothing to do - every distribution key is already present")
        return 0

    added = {}
    for row in missing:
        key = (row["distribution"], row["num_experts"], row["topk"], row["inter_size"])
        added[key] = added.get(key, 0) + 1
    for key in sorted(added):
        print(f"  + {added[key]:>4} rows  distribution={key[0]} experts={key[1]} topk={key[2]} inter={key[3]}")

    if not args.apply:
        print("\ndry run - re-run with --apply to write")
        return 0

    overlap, mismatched, worst = check_same_campaign(existing, candidates, args.rtol)
    if overlap:
        print(f"campaign check: {overlap} overlapping cells, {mismatched} differ (worst rel gap {worst:.1%})")
    if overlap and mismatched and not args.force:
        print(
            "\nREFUSED: --input-dir does NOT belong to the same measurement campaign as the target\n"
            "table. Every overlapping cell would carry different numbers, so appending would mix\n"
            "two sessions under one '0.23.0' label. Re-run pointing --input-dir at the raw CSVs\n"
            "that produced this table, or re-collect. Use --force only if you have confirmed the\n"
            "two sources are comparable (e.g. same hardware/git revision) and accept the mix."
        )
        return 1
    if overlap == 0:
        print("campaign check: no overlapping cells - nothing to cross-check, proceed with care")

    if not args.no_backup:
        backup = args.output + ".bak"
        shutil.copy2(args.output, backup)
        print(f"backup written: {backup}")

    with open(args.output, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writerows(missing)
    print(f"appended {len(missing)} rows -> {len(existing) + len(missing)} rows total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
