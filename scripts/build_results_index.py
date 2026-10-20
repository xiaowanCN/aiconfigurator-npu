#!/usr/bin/env python3
"""扫描 results/infer-strategy-npu/*/run.json，生成 results/infer-strategy-npu/index.json 清单。

前端静态界面通过 fetch 该清单渲染任务列表；任务详情再按需 fetch 对应 run.json。
后续切换为接口读取时，仅需替换 ui/app.js 的 DataSource，本清单可废弃。

用法：
    python scripts/build_results_index.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("results/infer-strategy-npu")
OUT = ROOT / "index.json"
RUN_ID_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")


def created_from_name(name: str) -> str:
    m = RUN_ID_RE.match(name)
    if not m:
        return ""
    y, mo, d, hh, mm, ss = m.groups()
    return f"{y}-{mo}-{d} {hh}:{mm}:{ss}"


def summarize(run_dir: Path) -> dict:
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    req = data.get("request", {}) or {}
    routing = data.get("routing", {}) or {}
    rec = (data.get("recommendation") or {}) or {}
    rec_best = rec.get("best") or {}
    return {
        "task_id": run_dir.name,  # 以目录名作为权威 task_id（与 HTTP 服务一致）
        "status": data.get("status", "unknown"),
        "created_at": created_from_name(run_dir.name) or data.get("generated_at", ""),
        "model": req.get("model"),
        "device": req.get("device"),
        "num_devices": req.get("num_devices"),
        "quantize": req.get("quantize"),
        "isl": req.get("input_length"),
        "osl": req.get("output_length"),
        "ttft_limit": req.get("ttft_limit"),
        "tpot_limit": req.get("tpot_limit"),
        "framework": routing.get("framework"),
        "framework_reason": routing.get("reason"),
        "recommendation_mode": rec.get("serving_mode"),
        "best_throughput": rec_best.get("throughput_tokens_per_s"),
    }


def main() -> int:
    if not ROOT.is_dir():
        print(f"[!] 未找到目录: {ROOT}")
        return 1
    tasks: list[dict] = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or not (d / "run.json").is_file():
            continue
        try:
            tasks.append(summarize(d))
        except Exception as exc:  # noqa: BLE001
            print(f"[!] 跳过 {d.name}: {exc}")
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(tasks),
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 生成 {OUT}（{len(tasks)} 个任务）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
