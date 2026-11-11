#!/usr/bin/env python3
"""模型 × 设备两两交叉测试（真实调用 infer-strategy-npu-task-enter.py）。

参考既有两套测试方法：
* ``demo_infer_strategy_task_enter_responses.py``：动态 import 入口脚本，从
  ``MODEL_PLANS`` / ``DEVICE_SPECS`` 读取当前白名单，配置驱动、自动适配后续扩充；
* ``test_infer_strategy_task_enter_container.sh``：逐个子进程执行真实命令，
  落盘 json/log/stdout/stderr，统一生成报告。

与演示脚本不同的是**不做任何模拟注入**：每个组合都真实启动一次入口脚本，
用于观察「哪些模型与设备的配对无法出结果」。

固定参数（接口文档 4.1.2 / 4.1.3，仅 ``--model`` / ``--device`` 变化）：
    --num_devices 64 --input-length 4096 --output-length 1024
    --tpot-limit 100 --ttft-limit 20 --serving_modes <默认 1>

用法：
    python scripts/test_model_device_matrix.py --dry-run
    python scripts/test_model_device_matrix.py --out-dir /apps/suyantest/outputs/infer/infer_strategy_npu/matrix
    python scripts/test_model_device_matrix.py --models DeepSeek-R1-0528-W8A8 --devices 950DT
    python scripts/test_model_device_matrix.py --resume
    python scripts/test_model_device_matrix.py --increment

产物（--out-dir 下）：
    matrix.md        矩阵报告（汇总、结果矩阵、无结果配对清单、全部明细、判定规则）
    matrix.json      机器可读汇总（含每个组合的完整记录与等价命令）
    no_result.txt    无结果配对清单
    cases/<model>__<device>.json/.log/.stdout/.stderr

退出码：0=全部组合都有结果；1=存在无结果组合；2=用法/环境错误。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTRY = ROOT / "infer-strategy-npu-task-enter.py"

FIXED_ARGS: tuple[tuple[str, str], ...] = (
    ("--input-length", "4096"),
    ("--output-length", "1024"),
    ("--tpot-limit", "100"),
    ("--ttft-limit", "20"),
)

REQUEST_KEY_ORDER = (
    "model", "device", "num_devices", "num_nodes", "input_length", "output_length",
    "ttft_limit", "tpot_limit", "serving_modes", "quantize-linear-action",
    "quantize-non-expert-linear-action", "mxfp4-group-size",
    "quantize-attention-action", "topk",
)
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00$")

STATUS_ORDER = ("OK", "EMPTY", "FAILED", "TIMEOUT", "ERROR")
NO_RESULT = ("EMPTY", "FAILED", "TIMEOUT", "ERROR")


# --------------------------------------------------------------------------- 入口白名单

def load_entry(entry: Path):
    """动态 import 入口脚本，读取当前 MODEL_PLANS / DEVICE_SPECS（不执行 main）。"""
    spec = importlib.util.spec_from_file_location("infer_strategy_task_enter_matrix", entry)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载入口脚本：{entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 解析字符串注解时需要
    spec.loader.exec_module(module)
    return module


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def next_run_tag(out_dir: Path) -> str:
    """无标记旧报告视为第 1 轮；自动递增 r02/r03...，保留历史结果。"""
    used: set[int] = {1} if (out_dir / "matrix.json").exists() else set()
    for path in out_dir.glob("r[0-9][0-9]/matrix.json"):
        used.add(int(path.parent.name[1:]))
    return f"r{max(used, default=0) + 1:02d}"


# --------------------------------------------------------------------------- 单组合执行

def build_command(opts: argparse.Namespace, model: str, device: str,
                  log: Path, result: Path) -> list[str]:
    cmd = [
        opts.python, str(opts.entry),
        "--model", model, "--device", device,
        "--num_devices", str(opts.num_devices),
    ]
    for key, value in FIXED_ARGS:
        cmd += [key, value]
    cmd += ["--serving_modes", str(opts.serving_modes),
            "--log-file", str(log), "--result-file", str(result)]
    return cmd


def contract_warnings(payload) -> list[str]:
    """轻量契约校验：字段、状态映射、result 形态、时间戳；只告警，不改变判定。"""
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ["结果顶层不是对象"]
    g_code, g_status = payload.get("code"), payload.get("status")
    if (g_code, g_status) not in ((0, "success"), (1, "failed")):
        warnings.append(f"外层 code/status 组合非法：{g_code}/{g_status}")
    common = payload.get("common")
    inner = common.get("infer_strategy_npu") if isinstance(common, dict) else None
    if not isinstance(inner, dict):
        return warnings + ["缺少 common.infer_strategy_npu 对象"]
    i_code, i_status = inner.get("code"), inner.get("status")
    if (i_code, i_status) not in ((0, "success"), (1, "failed"), (3, "empty")):
        warnings.append(f"内层 code/status 组合非法：{i_code}/{i_status}")
    data = inner.get("data")
    if not isinstance(data, dict):
        return warnings + ["内层 data 不是对象"]

    request = data.get("request")
    if isinstance(request, dict):
        missing = sorted(set(REQUEST_KEY_ORDER) - set(request))
        if missing:
            warnings.append(f"request 缺少字段：{missing}")
        if request.get("mxfp4-group-size") != "32":
            warnings.append(f"request.mxfp4-group-size={request.get('mxfp4-group-size')!r}，应为 '32'")
        if request.get("topk") != 1:
            warnings.append(f"request.topk={request.get('topk')!r}，应为 1")
    else:
        warnings.append("request 不是对象")

    result = data.get("result")
    if isinstance(result, dict):
        agg, disagg = result.get("agg"), result.get("disagg")
        if g_code == 1 or i_code in (1, 3):
            if agg or disagg:
                warnings.append("失败/无结果时 result 不是双空字典")
        for key in ("created_at", "started_at", "finished_at"):
            stamp = data.get(key)
            if not isinstance(stamp, str) or not TIME_RE.match(stamp):
                warnings.append(f"{key} 不是 ISO8601 秒精度 +08:00：{stamp!r}")
    else:
        warnings.append("result 不是对象")
    return warnings


def classify(record: dict) -> str:
    """OK=有候选；EMPTY=跑完无结果；FAILED=执行失败；TIMEOUT/ERROR=异常。"""
    if record["timed_out"]:
        return "TIMEOUT"
    payload = record["payload"]
    if payload is None:
        return "ERROR"
    common = payload.get("common") or {}
    inner = common.get("infer_strategy_npu") or {}
    data = inner.get("data") or {}
    result = data.get("result") or {}
    g_code, i_code = payload.get("code"), inner.get("code")
    if g_code == 1 or i_code == 1:
        return "FAILED"
    if i_code == 3:
        return "EMPTY"
    if i_code == 0 and isinstance(result, dict):
        n_agg = len((result.get("agg") or {}).get("candidates") or [])
        n_disagg = len((result.get("disagg") or {}).get("candidates") or [])
        if n_agg + n_disagg > 0:
            return "OK"
    return "ERROR"


def make_record(model: str, device: str, cmd: list[str], paths: dict) -> dict:
    return {
        "model": model,
        "device": device,
        "slug": f"{slug(model)}__{slug(device)}",
        "command": shlex.join(cmd) if hasattr(shlex, "join") else " ".join(map(str, cmd)),
        "result_file": str(paths["result"]),
        "log_file": str(paths["log"]),
        "stdout_file": str(paths["stdout"]),
        "stderr_file": str(paths["stderr"]),
        "exit_code": None,
        "timed_out": False,
        "elapsed_s": 0.0,
        "resumed": False,
        "payload": None,
        "global_code": None,
        "global_status": None,
        "global_message": None,
        "inner_code": None,
        "inner_status": None,
        "inner_message": None,
        "agg_candidates": None,
        "disagg_candidates": None,
        "warnings": [],
        "status": "ERROR",
    }


def fill_record(record: dict) -> None:
    """从结果 JSON 回填状态字段、候选数与契约告警，并给出最终判定。"""
    result_path = Path(record["result_file"])
    if result_path.is_file():
        try:
            record["payload"] = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["warnings"].append(f"结果文件不是合法 JSON：{exc}")
    else:
        record["warnings"].append("结果文件不存在")

    payload = record["payload"]
    if isinstance(payload, dict):
        common = payload.get("common") or {}
        inner = common.get("infer_strategy_npu") or {}
        data = inner.get("data") or {}
        result = data.get("result") or {}
        record.update({
            "global_code": payload.get("code"),
            "global_status": payload.get("status"),
            "global_message": payload.get("message"),
            "inner_code": inner.get("code"),
            "inner_status": inner.get("status"),
            "inner_message": inner.get("message"),
            "agg_candidates": len((result.get("agg") or {}).get("candidates") or [])
            if isinstance(result, dict) else None,
            "disagg_candidates": len((result.get("disagg") or {}).get("candidates") or [])
            if isinstance(result, dict) else None,
        })
        record["warnings"].extend(contract_warnings(payload))
    record["status"] = classify(record)


def run_pair(model: str, device: str, opts: argparse.Namespace, case_dir: Path) -> dict:
    stem = f"{slug(model)}__{slug(device)}"
    paths = {
        "result": case_dir / f"{stem}.json",
        "log": case_dir / f"{stem}.log",
        "stdout": case_dir / f"{stem}.stdout",
        "stderr": case_dir / f"{stem}.stderr",
    }
    cmd = build_command(opts, model, device, paths["log"], paths["result"])
    record = make_record(model, device, cmd, paths)

    if opts.dry_run:
        record["status"] = "DRYRUN"
        return record

    if opts.resume and paths["result"].is_file():
        record["resumed"] = True
        fill_record(record)
        return record

    for key in ("result", "log", "stdout", "stderr"):
        if paths[key].is_file():
            paths[key].unlink()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    started = time.monotonic()
    stdout_text, stderr_text = "", ""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=opts.timeout, env=env,
        )
        record["exit_code"] = proc.returncode
        stdout_text, stderr_text = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        stdout_text = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr_text = exc.stderr if isinstance(exc.stderr, str) else ""
    except OSError as exc:
        stderr_text = f"无法启动子进程：{exc}\n"
    record["elapsed_s"] = round(time.monotonic() - started, 1)
    paths["stdout"].write_text(stdout_text, encoding="utf-8")
    paths["stderr"].write_text(stderr_text, encoding="utf-8")

    fill_record(record)
    return record


# --------------------------------------------------------------------------- 报告

def _cell(record: dict) -> str:
    status = record["status"]
    if status == "OK":
        return f"OK (agg={record['agg_candidates']}, dis={record['disagg_candidates']})"
    if status == "EMPTY":
        return "EMPTY"
    if status == "FAILED":
        return f"FAILED (inner={record['inner_code']})"
    return status


def _tail(path: str, lines: int = 8) -> list[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def write_report(records: list[dict], opts: argparse.Namespace, out_dir: Path,
                 started_at: str, total_elapsed: float) -> None:
    models = list(dict.fromkeys(r["model"] for r in records))
    devices = list(dict.fromkeys(r["device"] for r in records))
    by_pair = {(r["model"], r["device"]): r for r in records}
    counts = {status: sum(1 for r in records if r["status"] == status)
              for status in STATUS_ORDER + ("DRYRUN",)}
    ok = counts["OK"]
    no_result = [r for r in records if r["status"] in NO_RESULT]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "config": {
            "entry": str(opts.entry), "python_bin": opts.python,
            "out_dir": str(out_dir), "started_at": started_at,
            "num_devices": opts.num_devices, "serving_modes": opts.serving_modes,
            "fixed_args": dict(FIXED_ARGS), "timeout_s": opts.timeout,
            "models": models, "devices": devices,
        },
        "totals": {
            "pairs": len(records), "has_result": ok,
            "no_result": len(no_result), "counts": counts,
            "elapsed_s": round(total_elapsed, 1),
        },
        "no_result_pairs": [
            {"model": r["model"], "device": r["device"], "status": r["status"],
             "message": r["global_message"], "log_file": r["log_file"]}
            for r in no_result
        ],
        "records": [{k: v for k, v in r.items() if k != "payload"} for r in records],
    }
    (out_dir / "matrix.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "no_result.txt").write_text(
        "".join(
            f"{r['status']} | {r['model']} | {r['device']} | "
            f"{r['global_message'] or '-'} | {r['result_file']} | {r['log_file']}\n"
            for r in no_result
        ) or "无无结果配对\n",
        encoding="utf-8",
    )

    fixed_text = ", ".join(f"`{k} {v}`" for k, v in FIXED_ARGS)
    report: list[str] = [
        "# 模型 × 设备交叉测试报告（infer-strategy-npu-task-enter.py）",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 环境：{summary['environment']['platform']} / Python {summary['environment']['python']}",
        f"- 入口脚本：`{opts.entry}`；解释器：`{opts.python}`",
        f"- 产物目录：`{out_dir}`；启动时间：`{started_at}`",
        f"- 变化维度：`--model`（{len(models)}）× `--device`（{len(devices)}）= **{len(records)}** 个组合",
        f"- 固定参数：`--num_devices {opts.num_devices}`、`--serving_modes {opts.serving_modes}`、{fixed_text}；单组合超时 `{opts.timeout}s`",
        "",
        "## 1. 汇总",
        "",
        f"**有结果 {ok} / {len(records)}；无结果 {len(no_result)}；累计耗时 {total_elapsed:.1f}s。**",
        "",
        "| 判定 | 数量 | 含义 |",
        "| --- | ---: | --- |",
        "| OK | %d | 内层 `0/success`，agg 或 disagg 至少 1 条候选 |" % counts["OK"],
        "| EMPTY | %d | 内层 `3/empty`，跑完但无可行解 |" % counts["EMPTY"],
        "| FAILED | %d | 全局 `1/failed`，执行失败 |" % counts["FAILED"],
        "| TIMEOUT | %d | 单组合超过 `--timeout` |" % counts["TIMEOUT"],
        "| ERROR | %d | 结果缺失/非法或契约异常 |" % counts["ERROR"],
        "",
        "## 2. 结果矩阵",
        "",
        "| 模型 \\ 设备 | " + " | ".join(f"`{d}`" for d in devices) + " |",
        "| --- | " + " | ".join("---" for _ in devices) + " |",
    ]
    for model in models:
        cells = []
        for device in devices:
            record = by_pair.get((model, device))
            cells.append(f"`{_cell(record)}`" if record else "-")
        report.append(f"| `{model}` | " + " | ".join(cells) + " |")

    report += ["", "## 3. 无结果配对清单", ""]
    if not no_result:
        report.append("全部组合均有结果。")
    else:
        for status in NO_RESULT:
            group = [r for r in no_result if r["status"] == status]
            if not group:
                continue
            report += [f"### {status}（{len(group)}）", ""]
            for record in group:
                report += [
                    f"#### `{record['model']}` × `{record['device']}`",
                    "",
                    f"- 全局/内层：`{record['global_code']}/{record['global_status']}` + "
                    f"`{record['inner_code']}/{record['inner_status']}`；"
                    f"退出码：{record['exit_code']}；耗时：{record['elapsed_s']}s",
                    f"- message：`{record['global_message'] or '-'}` / `{record['inner_message'] or '-'}`",
                    f"- 命令：`{record['command']}`",
                    f"- 结果文件：`{record['result_file']}`；日志：`{record['log_file']}`",
                ]
                if record["warnings"]:
                    report.append(f"- 契约告警：{' ; '.join(record['warnings'])}")
                tail = _tail(record["log_file"]) or _tail(record["stderr_file"])
                if tail:
                    report += ["- 日志/stderr 尾部：", "", "```text"] + tail + ["```", ""]
                else:
                    report.append("")

    report += [
        "## 4. 全部组合明细",
        "",
        "| # | 模型 | 设备 | 判定 | 全局 + 内层 | agg 候选 | disagg 候选 | 耗时(s) | 退出码 | 契约告警 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for index, record in enumerate(records, 1):
        report.append(
            f"| {index} | `{record['model']}` | `{record['device']}` | `{record['status']}` | "
            f"{record['global_code']}/{record['global_status']} + "
            f"{record['inner_code']}/{record['inner_status']} | "
            f"{record['agg_candidates'] if record['agg_candidates'] is not None else '-'} | "
            f"{record['disagg_candidates'] if record['disagg_candidates'] is not None else '-'} | "
            f"{record['elapsed_s']} | {record['exit_code']} | "
            f"{len(record['warnings'])} |"
        )
    report += [
        "",
        "## 5. 判定规则与覆盖说明",
        "",
        "- 白名单从入口脚本 `MODEL_PLANS` / `DEVICE_SPECS` 动态读取，新增模型或设备会自动进入矩阵。",
        "- 每个组合**真实启动一次**入口脚本（不注入模拟 recommend），仅 `--model` / `--device` 变化。",
        "- `OK`：内层 `0/success` 且至少一侧有候选；`EMPTY`：内层 `3/empty`（双空字典）；"
        "`FAILED`：全局 `1/failed`；`TIMEOUT`：单组合超时；`ERROR`：结果文件缺失/非法或状态异常。",
        "- 内层 `2/running` 按设计只出现在日志阶段记录，不作为终态落盘。",
        "- 轻量契约校验（14 个 request 字段、`mxfp4-group-size=\"32\"`、`topk=1`、"
        "状态映射、result 形态、秒精度时间戳）只产生告警，不改变判定。",
        "",
        "## 6. 产物",
        "",
        "- `matrix.md`：本报告；`matrix.json`：机器可读汇总；`no_result.txt`：无结果清单",
        "- `cases/<model>__<device>.json/.log/.stdout/.stderr`：单组合产物",
        "",
    ]
    (out_dir / "matrix.md").write_text("\n".join(report), encoding="utf-8")


# --------------------------------------------------------------------------- 入口

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模型 × 设备两两交叉测试（真实调用入口脚本，固定 64 卡）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir", default="model_device_matrix_results",
                        help="产物目录（默认 ./model_device_matrix_results）")
    parser.add_argument("--python", default=sys.executable, help="Python 解释器（默认当前解释器）")
    parser.add_argument("--entry", default=str(DEFAULT_ENTRY), help="入口脚本路径")
    parser.add_argument("--num-devices", type=int, default=64, help="固定卡数（默认 64）")
    parser.add_argument("--serving-modes", type=int, default=1, choices=(0, 1, 2),
                        help="固定服务部署模式（默认 1，仅 disagg，对设备/卡数最敏感）")
    parser.add_argument("--timeout", type=int, default=900, help="单组合超时秒数（默认 900）")
    parser.add_argument("--models", nargs="*", default=None, help="只测指定模型（默认全部）")
    parser.add_argument("--devices", nargs="*", default=None, help="只测指定设备（默认全部）")
    parser.add_argument("--resume", action="store_true", help="跳过已有结果 JSON 的组合（断点续跑）")
    parser.add_argument("--increment", action="store_true",
                        help="产物写入 rNN 子目录并自动递增，保留历史轮次")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不实际运行")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    opts.entry = Path(opts.entry).resolve()
    if not opts.entry.is_file():
        print(f"入口脚本不存在：{opts.entry}", file=sys.stderr)
        return 2
    if opts.num_devices <= 0 or opts.timeout <= 0:
        print("--num-devices / --timeout 必须为正数", file=sys.stderr)
        return 2

    module = load_entry(opts.entry)
    all_models = list(module.MODEL_PLANS)
    all_devices = list(module.DEVICE_SPECS)
    models = opts.models or all_models
    devices = opts.devices or all_devices
    unknown_m = [m for m in models if m not in all_models]
    unknown_d = [d for d in devices if d not in all_devices]
    if unknown_m or unknown_d:
        print(f"未知模型：{unknown_m}；未知设备：{unknown_d}", file=sys.stderr)
        print(f"可用模型：{all_models}", file=sys.stderr)
        print(f"可用设备：{all_devices}", file=sys.stderr)
        return 2

    out_dir = Path(opts.out_dir).resolve()
    if opts.increment:
        out_dir = out_dir / next_run_tag(out_dir)
    case_dir = out_dir / "cases"
    if not opts.dry_run:
        case_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    total_started = time.monotonic()
    print("=" * 78)
    print(f"入口脚本 : {opts.entry}")
    print(f"解释器   : {opts.python}")
    print(f"产物目录 : {out_dir}")
    print(f"矩阵     : {len(models)} 模型 × {len(devices)} 设备 = {len(models) * len(devices)} 组合")
    print(f"固定参数 : num_devices={opts.num_devices}, serving_modes={opts.serving_modes}, "
          f"input=4096, output=1024, tpot=100, ttft=20")
    print(f"超时     : {opts.timeout}s / 组合；模式: {'dry-run' if opts.dry_run else '真实调用'}")
    print("=" * 78)

    records: list[dict] = []
    for model in models:
        for device in devices:
            record = run_pair(model, device, opts, case_dir)
            records.append(record)
            if opts.dry_run:
                print(record["command"])
            else:
                suffix = " (resume)" if record["resumed"] else ""
                print(f"  [{record['status']:<7}] {model:<32} × {device:<6} "
                      f"{record['elapsed_s']:>6.1f}s{suffix}")

    if opts.dry_run:
        print(f"\ndry-run 完成：共 {len(records)} 条命令，未执行、未写报告。")
        return 0

    total_elapsed = time.monotonic() - total_started
    write_report(records, opts, out_dir, started_at, total_elapsed)
    no_result = [r for r in records if r["status"] in NO_RESULT]
    print("=" * 78)
    print(f"报告：{out_dir / 'matrix.md'}")
    print(f"汇总：{out_dir / 'matrix.json'}；无结果清单：{out_dir / 'no_result.txt'}")
    print(f"结果：{len(records)} 组合，有结果 {len(records) - len(no_result)}，无结果 {len(no_result)}，"
          f"耗时 {total_elapsed:.1f}s")
    if no_result:
        print("无结果配对：")
        for record in no_result:
            print(f"  {record['status']:<7} {record['model']} × {record['device']} "
                  f"— {record['global_message'] or record['warnings']}")
    print("=" * 78)
    return 1 if no_result else 0


if __name__ == "__main__":
    sys.exit(main())
