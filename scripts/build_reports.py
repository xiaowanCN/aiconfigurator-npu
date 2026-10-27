#!/usr/bin/env python3
"""为**已有的**运行补生成单文件静态报告 ``report.html``。

任务成功时服务会自动产出报告（``api/tasks.py``），但以下两类没有：

- 本次改动之前就已经跑完的任务；
- 直接用 CLI（``infer-strategy-npu --recommend``）跑出来的任务。

本脚本扫描结果根目录，为每个含 ``run.json`` 的任务目录补写 ``report.html``。

用法：
    python scripts/build_reports.py                          # 默认 results/infer-strategy-npu
    python scripts/build_reports.py _verify_results          # 指定结果根目录
    python scripts/build_reports.py D:/workspace/results/infer-strategy-npu
    python scripts/build_reports.py <结果根>/<run_id>         # 只处理单个任务
    python scripts/build_reports.py <结果根> --force          # 覆盖已存在的 report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer_strategy_npu.reporting import write_report_from_run_json  # noqa: E402

DEFAULT_ROOT = "results/infer-strategy-npu"


def collect(root: Path) -> list[Path]:
    """返回待处理的任务目录：根目录扫描子目录，或直接就是单个任务目录。"""
    if (root / "run.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if (d / "run.json").is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description="为已有运行补生成 report.html")
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="结果根目录，或单个任务目录")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 report.html")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    dirs = collect(root)
    if not dirs:
        print(f"[!] 未找到任何含 run.json 的目录: {root}")
        return 1

    ok = skipped = failed = 0
    for d in dirs:
        target = d / "report.html"
        if target.is_file() and not args.force:
            skipped += 1
            print(f"[-] 已存在，跳过（加 --force 覆盖）: {target}")
            continue
        try:
            out = write_report_from_run_json(d / "run.json")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[!] 失败 {d.name}: {type(exc).__name__}: {exc}")
            continue
        if out is None:
            failed += 1
            print(f"[!] 失败 {d.name}: run.json 不可读")
            continue
        ok += 1
        print(f"[ok] {out}（{out.stat().st_size / 1024:.1f} KB）")

    print(f"\n完成：新增 {ok} / 跳过 {skipped} / 失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
