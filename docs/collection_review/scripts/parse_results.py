#!/usr/bin/env python3
"""Recover clean Final Results tables from run logs (undo escaped \\n)."""
import os
import re

OUT = r"D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\staging-1014\1014\runs"
order = [
    ("R1", "HYBRID"), ("R1", "EMPIRICAL"), ("R1", "SOL"),
    ("V4", "SILICON"), ("V4", "HYBRID"), ("V4", "EMPIRICAL"), ("V4", "SOL"),
    ("Qwen", "SILICON"), ("Qwen", "HYBRID"), ("Qwen", "EMPIRICAL"), ("Qwen", "SOL"),
]


def extract(log):
    txt = open(os.path.join(OUT, log), encoding="utf-8", errors="replace").read()
    # The table is embedded with literal \n in a 'Message: ...' line; recover.
    if "\\n" in txt:
        txt = txt.replace("\\n", "\n")
    idx = txt.rfind("Final Results")
    if idx < 0:
        return None
    seg = txt[idx:]
    # keep from the header separator to the closing ****
    m = re.search(r"\+[-+]+\+", seg)
    start = m.start() if m else 0
    end = seg.rfind("****")
    if end < 0:
        end = len(seg)
    return seg[start:end]


def main():
    os.makedirs(os.path.join(OUT, "clean"), exist_ok=True)
    for mk, m in order:
        log = f"{mk}_{m}.log"
        clean = extract(log)
        if clean is None:
            print(f"!! no Final Results in {log}")
            continue
        path = os.path.join(OUT, "clean", f"{mk}_{m}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(clean)
        print(f"wrote {path} ({len(clean)} chars)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
