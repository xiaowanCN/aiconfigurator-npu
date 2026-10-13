#!/usr/bin/env python3
"""Merge two aiconfigurator perf tables by key (new overrides existing, other rows kept).

Problem this solves
-------------------
The shared perf tables (``gemm_perf.txt``, ``moe_perf.txt``, ``*_attention_perf.txt``,
``*_mla_perf.txt``, ``dsv4_*_module_perf.txt``, ``mhc_module_perf.txt``, ``gdn_perf.txt`` …)
are keyed by geometry (hidden / heads / ep / …), **not by model name**, and are consumed by
many models at once. Historically, landing new data with ``convert_to_aiconfigurator.py
--append`` was non-idempotent (duplicate rows) and a full ``cp`` would wipe every other
model's rows. This script merges a freshly-collected "clean" table into the existing table
**in place**:

  * rows whose key already exists in the existing table are REPLACED by the new row
    (key = all columns except ``latency``);
  * rows present only in the existing table are PRESERVED (so other models' data survives);
  * brand-new rows from the new table are appended.

The operation is idempotent: re-running it with the same inputs yields the same file, so it
is safe to repeat after a re-collection. It never duplicates rows.

Usage
-----
    # write merged result back into the existing file
    python tools/merge_perf_tables.py <existing.txt> <new.txt> --in-place

    # or print to stdout (redirect to land)
    python tools/merge_perf_tables.py <existing.txt> <new.txt> > merged.txt

If ``existing`` does not exist, the new table is simply copied through.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys


def _latency_col(header: list[str]) -> int | None:
    for i, c in enumerate(header):
        if c.strip().lower() == "latency":
            return i
    return None


def load(path: str) -> tuple[list[str], list[list[str]]]:
    if not path or not os.path.isfile(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def merge(existing_path: str, new_path: str) -> tuple[list[str], list[list[str]]]:
    e_head, e_rows = load(existing_path)
    n_head, n_rows = load(new_path)
    head = n_head or e_head
    lat = _latency_col(head)
    if lat is None:
        raise SystemExit(f"no 'latency' column in header: {head}")

    def key(row: list[str]) -> tuple[str, ...]:
        return tuple(v for i, v in enumerate(row) if i != lat)

    merged = {key(r): r for r in e_rows}
    for r in n_rows:
        merged[key(r)] = r

    # Preserve existing row order; only append truly-new keys at the end.
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for r in e_rows:
        k = key(r)
        if k in seen:
            continue
        out.append(merged.pop(k, r))
        seen.add(k)
    for k, r in merged.items():
        if k not in seen:
            out.append(r)
            seen.add(k)
    return head, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("existing", help="existing perf table (e.g. 0.23.0/gemm_perf.txt)")
    ap.add_argument("new", help="freshly collected clean table to merge in")
    ap.add_argument("--in-place", action="store_true", help="write result back into <existing>")
    args = ap.parse_args()

    head, out = merge(args.existing, args.new)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    w.writerows(out)
    text = buf.getvalue()

    if args.in_place:
        with open(args.existing, "w", newline="", encoding="utf-8") as f:
            f.write(text)
        print(f"merged -> {args.existing} ({len(out)} rows)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
