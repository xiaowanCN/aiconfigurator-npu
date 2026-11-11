#!/usr/bin/env python3
"""容器端黑盒测试的校验与报告生成器。

由 ``test_infer_strategy_task_enter_container.sh`` 调用，不单独使用：

* ``check``  ：校验单个用例的结果 JSON / 日志，向 stdout 输出一行 JSON 记录；
* ``skip``   ：输出一行 SKIP 记录（模式过滤掉的用例也要进报告）；
* ``report`` ：读 results.jsonl，生成 report.md / summary.json / failures.txt。

校验依据：接口文档 4.1.2 / 4.1.3 与
docs/plans/INFER_STRATEGY_NPU_TASK_ENTER_DESIGN.md（v3 定稿）。
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

REQUEST_KEY_ORDER = (
    "model", "device", "num_devices", "num_nodes", "input_length", "output_length",
    "ttft_limit", "tpot_limit", "serving_modes", "quantize-linear-action",
    "quantize-non-expert-linear-action", "mxfp4-group-size",
    "quantize-attention-action", "topk",
)
REQUEST_INT_KEYS = (
    "num_devices", "num_nodes", "input_length", "output_length",
    "tpot_limit", "serving_modes", "topk",
)
REQUEST_FLOAT_KEYS = ("ttft_limit",)
REQUEST_STR_KEYS = (
    "model", "device", "quantize-linear-action",
    "quantize-non-expert-linear-action", "mxfp4-group-size", "quantize-attention-action",
)
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00$")

GLOBAL_OK = ((0, "success"), (1, "failed"))
INNER_OK = ((0, "success"), (1, "failed"), (3, "empty"))


def _codes(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _key_values(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in spec.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _iter_floats(value):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_floats(item)


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _validate_contract(payload: dict, log_text: str, errors: list[str]) -> None:
    """结果 JSON 的契约校验（字段、类型、状态、result 形态、时间戳、指标精度）。"""
    for key in ("code", "status", "message", "common"):
        if key not in payload:
            errors.append(f"外层缺少字段 {key}")
    if errors:
        return

    g_code, g_status = payload.get("code"), payload.get("status")
    if not isinstance(g_code, int) or isinstance(g_code, bool):
        errors.append("外层 code 不是 integer")
    if not isinstance(g_status, str):
        errors.append("外层 status 不是 string")
    if (g_code, g_status) not in GLOBAL_OK:
        errors.append(f"外层 code/status 组合非法：{g_code}/{g_status}")
    if not isinstance(payload.get("message"), str) or not payload.get("message"):
        errors.append("外层 message 为空或非 string")

    common = payload.get("common")
    if not isinstance(common, dict) or not isinstance(common.get("infer_strategy_npu"), dict):
        errors.append("缺少 common.infer_strategy_npu 对象")
        return
    inner = common["infer_strategy_npu"]
    for key in ("code", "status", "message", "data"):
        if key not in inner:
            errors.append(f"内层缺少字段 {key}")
    if any(key not in inner for key in ("code", "status", "message", "data")):
        return

    i_code, i_status = inner.get("code"), inner.get("status")
    if (i_code, i_status) not in INNER_OK:
        errors.append(f"内层 code/status 组合非法（含 2/running 不应落盘）：{i_code}/{i_status}")
    if not isinstance(inner.get("message"), str) or not inner.get("message"):
        errors.append("内层 message 为空或非 string")

    data = inner.get("data")
    if not isinstance(data, dict):
        errors.append("内层 data 不是对象")
        return
    for key in ("created_at", "started_at", "finished_at", "request", "result"):
        if key not in data:
            errors.append(f"data 缺少字段 {key}")
    if any(key not in data for key in ("created_at", "started_at", "finished_at", "request", "result")):
        return

    stamps = [data.get(key) for key in ("created_at", "started_at", "finished_at")]
    for stamp in stamps:
        if not isinstance(stamp, str) or not TIME_RE.match(stamp):
            errors.append(f"时间戳不是 ISO8601 秒精度 +08:00：{stamp!r}")
    if all(isinstance(stamp, str) and TIME_RE.match(stamp) for stamp in stamps):
        if stamps != sorted(stamps):
            errors.append("三个时间戳不是非递减")

    request = data.get("request")
    if not isinstance(request, dict):
        errors.append("request 不是对象")
    else:
        if set(request) != set(REQUEST_KEY_ORDER):
            errors.append(f"request 字段不全/有多余：{sorted(request)}")
        if list(request) != list(REQUEST_KEY_ORDER):
            errors.append("request 字段顺序与 4.1.3 参数说明表不一致")
        # 失败态允许「尽力回显」为 null（模型/设备非法、数值无法解析、argparse 失败）；
        # 成功态必须输出完整且类型正确的值。
        allow_null = g_code == 1
        for key in REQUEST_INT_KEYS:
            value = request.get(key)
            if value is None and allow_null:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"request.{key} 应为 integer，实际 {value!r}")
        for key in REQUEST_FLOAT_KEYS:
            value = request.get(key)
            if value is None and allow_null:
                continue
            if not isinstance(value, float):
                errors.append(f"request.{key} 应为 float，实际 {value!r}")
        for key in REQUEST_STR_KEYS:
            value = request.get(key)
            if value is None and allow_null:
                continue
            if not isinstance(value, str):
                errors.append(f"request.{key} 应为 string，实际 {value!r}")
        if request.get("topk") != 1 and not (allow_null and request.get("topk") is None):
            errors.append(f"request.topk 应为 1，实际 {request.get('topk')!r}")
        if request.get("mxfp4-group-size") != "32":
            errors.append(f"request.mxfp4-group-size 应为 '32'，实际 {request.get('mxfp4-group-size')!r}")

    result = data.get("result")
    if not isinstance(result, dict):
        errors.append("result 不是对象")
        return
    agg, disagg = result.get("agg"), result.get("disagg")
    if not isinstance(agg, dict) or not isinstance(disagg, dict):
        errors.append("result.agg / result.disagg 应为对象")
        return
    if g_code == 1 or i_code in (1, 3):
        if agg or disagg:
            errors.append(f"失败(1)/无结果(3)时 result 应双空字典，实际 {result}")
    else:
        modes = request.get("serving_modes") if isinstance(request, dict) else None
        if modes == 0 and disagg:
            errors.append("serving_modes=0 时 disagg 应为空字典")
        if modes == 1 and agg:
            errors.append("serving_modes=1 时 agg 应为空字典")
        n_agg = len(agg.get("candidates") or [])
        n_disagg = len(disagg.get("candidates") or [])
        if n_agg + n_disagg == 0:
            errors.append("内层 0/success 但两侧都没有候选")
    for label, side in (("agg", agg), ("disagg", disagg)):
        candidates = side.get("candidates")
        if candidates is not None and not isinstance(candidates, list):
            errors.append(f"result.{label}.candidates 应为数组")
        elif isinstance(candidates, list):
            for value in _iter_floats(candidates):
                if abs(value - round(value, 2)) > 1e-9:
                    errors.append(f"result.{label} 指标 {value} 未保留 2 位小数")
                    break

    if "[done]" not in log_text:
        errors.append("日志缺少 [done] 终态记录")


def _record(args: argparse.Namespace, status: str, actual: dict, errors: list[str],
            checks: dict, notes: str) -> dict:
    return {
        "status": status,
        "id": args.id,
        "name": args.name,
        "category": args.category,
        "group": args.group,
        "command": args.command,
        "expected": {
            "global_code": args.expect_global,
            "inner_code": args.expect_inner,
            "exit_code": args.expect_exit,
            "message": args.expect_message,
            "result": args.expect_result,
            "request": args.expect_request,
        },
        "actual": actual,
        "elapsed_s": args.elapsed,
        "exit_code": args.exit_code,
        "timed_out": bool(args.timed_out),
        "result_file": args.result,
        "log_file": args.log,
        "stdout_file": args.stdout_file,
        "stderr_file": args.stderr_file,
        "checks": checks,
        "errors": errors,
        "notes": notes,
    }


def cmd_check(args: argparse.Namespace) -> int:
    errors: list[str] = []
    checks: dict = {}
    result_path = Path(args.result)
    # 日志可能因 log-file 非法而只落在 stdout；合并三处文本后再做阶段断言
    log_text = "\n".join(
        text for text in (
            _read(args.log), _read(args.stdout_file), _read(args.stderr_file),
        ) if text
    )
    payload = None

    if args.expect_no_result:
        if result_path.exists():
            errors.append("该用例期望不生成结果文件，但文件存在")
    elif not result_path.exists():
        errors.append("结果文件不存在")
    else:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"结果文件不是合法 JSON：{exc}")

    actual: dict = {}
    if isinstance(payload, dict):
        inner = payload.get("common", {}).get("infer_strategy_npu", {}) \
            if isinstance(payload.get("common"), dict) else {}
        data = inner.get("data", {}) if isinstance(inner, dict) else {}
        data = data if isinstance(data, dict) else {}
        result = data.get("result", {})
        actual = {
            "global_code": payload.get("code"),
            "global_status": payload.get("status"),
            "global_message": payload.get("message"),
            "inner_code": inner.get("code") if isinstance(inner, dict) else None,
            "inner_status": inner.get("status") if isinstance(inner, dict) else None,
            "inner_message": inner.get("message") if isinstance(inner, dict) else None,
            "result": result,
            "request": data.get("request", {}),
            "times": [data.get(k) for k in ("created_at", "started_at", "finished_at")],
        }

    if args.timed_out:
        errors.append(f"用例超时（限制 {args.timeout}s）")

    if args.expect_no_result:
        checks["contract"] = "skipped（该用例不产出结果文件）"
        checks["exit_code"] = f"{args.exit_code} ∈ {{{args.expect_exit}}}"
        if args.exit_code not in _codes(args.expect_exit):
            errors.append(f"退出码应为 {args.expect_exit}，实际 {args.exit_code}")
    else:
        if isinstance(payload, dict):
            if args.check_contract:
                _validate_contract(payload, log_text, errors)
                checks["contract"] = "checked"
            else:
                checks["contract"] = "skipped"
            if actual["global_code"] not in _codes(args.expect_global):
                errors.append(
                    f"全局 code 应为 {args.expect_global}，实际 {actual['global_code']}"
                )
            if actual["inner_code"] not in _codes(args.expect_inner):
                errors.append(
                    f"内层 code 应为 {args.expect_inner}，实际 {actual['inner_code']}"
                )
            if args.expect_message and args.expect_message not in str(actual["global_message"]):
                errors.append(
                    f"外层 message 应包含 {args.expect_message!r}，实际 {actual['global_message']!r}"
                )
            if args.expect_result == "empty" and actual["result"] != {"agg": {}, "disagg": {}}:
                errors.append(f"result 应为双空字典，实际 {actual['result']}")
            for key, value in _key_values(args.expect_request).items():
                got = actual["request"].get(key) if isinstance(actual["request"], dict) else None
                if str(got) != value:
                    errors.append(f"request.{key} 应为 {value}，实际 {got!r}")
            if args.expect_running and "2/running" not in log_text:
                errors.append("日志缺少运行期 2/running 阶段记录")
            if args.expect_validate and "[validate]" not in log_text:
                errors.append("日志缺少 [validate] 记录")
        if not args.timed_out and args.exit_code not in _codes(args.expect_exit):
            errors.append(f"退出码应为 {args.expect_exit}，实际 {args.exit_code}")

    if args.runtime_observed == "0":
        errors.append("运行期检测到 --result-file 提前出现（应只在终态写一次）")
    elif args.runtime_observed == "1":
        checks["runtime_no_result_file"] = "checked"

    notes = args.notes
    if args.runtime_observed in ("0", "1"):
        observation = (
            "运行期轮询：结果文件未提前出现" if args.runtime_observed == "1"
            else "运行期轮询：结果文件提前出现"
        )
        notes = f"{notes}；{observation}" if notes else observation

    print(json.dumps(_record(args, "FAIL" if errors else "PASS", actual, errors, checks, notes),
                     ensure_ascii=False))
    return 0


def cmd_skip(args: argparse.Namespace) -> int:
    print(json.dumps({
        "status": "SKIP", "id": args.id, "name": args.name, "category": args.category,
        "group": args.group, "command": args.command, "expected": {}, "actual": {},
        "elapsed_s": 0, "exit_code": None, "timed_out": False,
        "result_file": "", "log_file": "", "stdout_file": "", "stderr_file": "",
        "checks": {}, "errors": [], "notes": args.reason,
    }, ensure_ascii=False))
    return 0


CATEGORY_DESC = (
    ("smoke", "三种 serving_modes 的基础寻优，覆盖 0/1/2 与全局/内层状态组合"),
    ("defaults", "省略 ttft/tpot/serving_modes 时的默认值与回显"),
    ("alias", "下划线 / 连字符双 option string 别名"),
    ("matrix", "9 个模型方案 × 3 个设备 × 4 种卡数的 PD 组合过滤矩阵"),
    ("boundary", "num_devices / input / output / ttft / tpot 的值域上下界与极端 SLO"),
    ("invalid", "白名单、值域越界、非整数、路径、缺必填参数等全部失败分支"),
    ("contract", "运行期不写结果文件、失败仍落盘完整 JSON 等落盘契约"),
)


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def cmd_report(args: argparse.Namespace) -> int:
    lines = [line for line in Path(args.results).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for r in records if r["status"] == "PASS")
    failed = sum(1 for r in records if r["status"] == "FAIL")
    skipped = sum(1 for r in records if r["status"] == "SKIP")
    elapsed = sum(float(r.get("elapsed_s") or 0) for r in records)

    by_category: dict[str, dict[str, int]] = {}
    by_combo: dict[str, int] = {}
    for record in records:
        cat = record.get("category", "-")
        bucket = by_category.setdefault(cat, {"total": 0, "PASS": 0, "FAIL": 0, "SKIP": 0})
        bucket["total"] += 1
        bucket[record["status"]] += 1
        actual = record.get("actual") or {}
        combo = (f"{actual.get('global_code', '-')}/{actual.get('global_status', '-')}"
                 f" + {actual.get('inner_code', '-')}/{actual.get('inner_status', '-')}")
        if any(k in actual for k in ("global_code", "inner_code")):
            by_combo[combo] = by_combo.get(combo, 0) + 1

    fatal_real = [
        r for r in records
        if r.get("category") in ("smoke", "defaults", "alias", "matrix", "boundary")
        and (r.get("actual") or {}).get("inner_code") == 1
    ]

    env = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    config = {
        "entry": args.entry, "python_bin": args.python_bin, "out_dir": str(out_dir),
        "mode": args.mode, "per_case_timeout_s": args.timeout,
        "started_at": args.started_at, "base_args": args.base_args,
    }
    summary = {
        "generated_at": env["generated_at"], "environment": env, "config": config,
        "totals": {
            "total": len(records), "passed": passed, "failed": failed,
            "skipped": skipped, "elapsed_s": round(elapsed, 1),
        },
        "by_category": by_category, "by_status_combo": by_combo, "records": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report: list[str] = [
        "# infer-strategy-npu-task-enter.py 容器端测试报告",
        "",
        f"- 生成时间：{env['generated_at']}",
        f"- 环境：{env['platform']} / Python {env['python']}",
        f"- 入口脚本：`{args.entry}`",
        f"- 解释器：`{args.python_bin}`",
        f"- 输出目录：`{out_dir}`",
        f"- 测试模式：`{args.mode}`；单用例超时：`{args.timeout}s`；启动时间：`{args.started_at}`",
        f"- 基准参数：`{args.base_args}`",
        "",
        "## 1. 汇总",
        "",
        f"**总计 {len(records)} 个用例：通过 {passed}，失败 {failed}，跳过 {skipped}；累计耗时 {elapsed:.1f}s。**",
        "",
        "| 类别 | 用例数 | 通过 | 失败 | 跳过 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, _ in CATEGORY_DESC:
        bucket = by_category.get(category)
        if bucket:
            report.append(
                f"| {category} | {bucket['total']} | {bucket['PASS']} | {bucket['FAIL']} | {bucket['SKIP']} |"
            )
    for category in sorted(set(by_category) - {c for c, _ in CATEGORY_DESC}):
        bucket = by_category[category]
        report.append(
            f"| {category} | {bucket['total']} | {bucket['PASS']} | {bucket['FAIL']} | {bucket['SKIP']} |"
        )
    report += [
        "",
        "实际出现的内外层状态组合（仅统计产出了结果 JSON 的用例）：",
        "",
        "| 全局 + 内层 | 用例数 |",
        "| --- | ---: |",
    ]
    for combo, count in sorted(by_combo.items()):
        report.append(f"| {_md_cell(combo)} | {count} |")
    report += [
        "",
        "> 说明：内层 `2/running` 按设计只出现在日志阶段记录，不作为终态写入结果文件；"
        "真实寻优用例允许 `0/success + 0/success` 或 `0/success + 3/empty`（取决于 NPU 环境是否产生候选）。",
        "",
    ]
    if fatal_real:
        report += [
            f"> **环境提示**：{len(fatal_real)} 个真实寻优用例返回 `1/failed`，"
            "请检查 msmodeling whl、NPU 驱动与 `model_configs/msmodeling/<dir>/config.json`，"
            "并查看对应用例日志的异常栈。",
            "",
        ]

    report += [
        "## 2. 用例明细",
        "",
        "| # | ID | 类别 | 名称 | 预期 | 实际 | 耗时(s) | 结论 |",
        "| ---: | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for index, record in enumerate(records, 1):
        expected = record.get("expected") or {}
        actual = record.get("actual") or {}
        exp_text = "-"
        if expected:
            exp_text = (f"g∈{{{expected.get('global_code')}}}, "
                        f"i∈{{{expected.get('inner_code')}}}, "
                        f"rc∈{{{expected.get('exit_code')}}}")
        act_text = "-"
        if actual:
            act_text = (f"{actual.get('global_code')}/{actual.get('global_status')} + "
                        f"{actual.get('inner_code')}/{actual.get('inner_status')}, "
                        f"rc={record.get('exit_code')}")
        elif record.get("exit_code") is not None:
            act_text = f"无结果 JSON，rc={record.get('exit_code')}"
        report.append(
            f"| {index} | `{_md_cell(record['id'])}` | {record.get('category', '-')} | "
            f"{_md_cell(record['name'])} | {_md_cell(exp_text)} | {_md_cell(act_text)} | "
            f"{record.get('elapsed_s', 0)} | {record['status']} |"
        )

    report += ["", "## 3. 失败明细", ""]
    failures = [r for r in records if r["status"] == "FAIL"]
    if not failures:
        report.append("无失败用例。")
    else:
        for record in failures:
            report += [
                f"### `{record['id']}` {record['name']}",
                "",
                f"- 命令：`{_md_cell(record.get('command', ''))}`",
                f"- 结果文件：`{record.get('result_file', '')}`",
                f"- 日志：`{record.get('log_file', '')}`",
                f"- 退出码：{record.get('exit_code')}；耗时：{record.get('elapsed_s')}s；"
                f"超时：{'是' if record.get('timed_out') else '否'}",
                "- 断言失败：",
            ]
            for error in record.get("errors", []):
                report.append(f"  - {_md_cell(error)}")
            if record.get("notes"):
                report.append(f"- 备注：{_md_cell(record['notes'])}")
            log_tail = _read(record.get("log_file", "")).splitlines()[-15:]
            stderr_tail = _read(record.get("stderr_file", "")).splitlines()[-10:]
            if log_tail or stderr_tail:
                report += ["- 日志 / stderr 尾部：", "", "```text"]
                report += log_tail
                if stderr_tail:
                    report.append("---- stderr ----")
                    report += stderr_tail
                report += ["```", ""]
            else:
                report.append("")

    report += [
        "## 4. 覆盖说明",
        "",
        "| 类别 | 覆盖点 |",
        "| --- | --- |",
    ]
    for category, desc in CATEGORY_DESC:
        report.append(f"| {category} | {_md_cell(desc)} |")
    report += [
        "",
        "所有用例统一校验：结果 JSON 字段全集与类型（含 `request` 14 字段、"
        "`mxfp4-group-size=\"32\"`、`topk=1`）、ISO8601 秒精度时间戳非递减、"
        "内外层 `code/status` 映射、`result` 形态（失败/无结果双空字典）、"
        "指标 2 位小数、退出码、日志 `[done]` 终态记录。",
        "",
        "## 5. 产物",
        "",
        "- `report.md`：本报告",
        "- `summary.json`：机器可读汇总（含每个用例的完整记录）",
        "- `failures.txt`：失败清单",
        "- `results.jsonl`：逐用例原始记录",
        "- `cases/<id>.json` / `.log` / `.stdout` / `.stderr`：单用例产物",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "failures.txt").write_text(
        "".join(
            f"{r['id']} | {r['name']} | {' ; '.join(r.get('errors', []))}\n" for r in failures
        ) or "无失败用例\n",
        encoding="utf-8",
    )
    print(f"报告：{out_dir / 'report.md'}")
    print(f"汇总：{out_dir / 'summary.json'}")
    print(f"结果：总计 {len(records)}，通过 {passed}，失败 {failed}，跳过 {skipped}")
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--group", default="all")
    parser.add_argument("--command", default="")
    parser.add_argument("--result", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--stdout-file", default="")
    parser.add_argument("--stderr-file", default="")
    parser.add_argument("--elapsed", type=float, default=0)
    parser.add_argument("--exit-code", type=int, default=-1)
    parser.add_argument("--timed-out", type=int, default=0)
    parser.add_argument("--expect-global", default="")
    parser.add_argument("--expect-inner", default="")
    parser.add_argument("--expect-exit", default="")
    parser.add_argument("--expect-message", default="")
    parser.add_argument("--expect-result", default="auto", choices=("auto", "empty", "skip"))
    parser.add_argument("--expect-request", default="")
    parser.add_argument("--expect-running", type=int, default=0)
    parser.add_argument("--expect-validate", type=int, default=0)
    parser.add_argument("--expect-no-result", type=int, default=0)
    parser.add_argument("--check-contract", type=int, default=1)
    parser.add_argument("--runtime-observed", default="", choices=("", "0", "1"))
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--notes", default="")


def main(argv: list[str] | None = None) -> int:
    # 记录经 stdout 重定向写入 results.jsonl，统一 UTF-8，避免非 UTF-8 locale 下乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    check = sub.add_parser("check", help="校验单个用例")
    _add_common(check)
    check.set_defaults(func=cmd_check)

    skip = sub.add_parser("skip", help="记录一个被模式过滤掉的用例")
    skip.add_argument("--id", required=True)
    skip.add_argument("--name", required=True)
    skip.add_argument("--category", required=True)
    skip.add_argument("--group", default="all")
    skip.add_argument("--command", default="")
    skip.add_argument("--reason", default="被当前模式过滤")
    skip.set_defaults(func=cmd_skip)

    report = sub.add_parser("report", help="生成测试报告")
    report.add_argument("--results", required=True)
    report.add_argument("--out-dir", required=True)
    report.add_argument("--entry", required=True)
    report.add_argument("--python-bin", required=True)
    report.add_argument("--mode", required=True)
    report.add_argument("--timeout", required=True)
    report.add_argument("--started-at", required=True)
    report.add_argument("--base-args", default="")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
