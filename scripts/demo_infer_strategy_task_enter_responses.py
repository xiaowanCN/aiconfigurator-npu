#!/usr/bin/env python3
"""本地演示：依次自动运行 infer-strategy-npu-task-enter.py 的各类案例，输出响应示例。

本机没有 msmodeling / NPU，因此脚本把入口的 ``load_recommend`` 换成模拟实现，
让「成功 / 无结果 / 入参失败 / 执行失败」四类响应都能在本地真实跑出来。
每个案例都会：
- 打印等价的本地命令与容器命令（容器内换成真实路径即可直接执行）；
- 调用入口的 main()，写独立的 --log-file 与 --result-file；
- 打印该案例 JSON 里的 code / status / message（全局与内层）与 result 形态。

只覆盖 serving_modes=0 / 1，不测 2。

运行：
    python scripts/demo_infer_strategy_task_enter_responses.py
    python scripts/demo_infer_strategy_task_enter_responses.py --out-dir D:/tmp/responses
    python scripts/demo_infer_strategy_task_enter_responses.py --out-dir D:/tmp/responses --increment
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import shlex
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "infer-strategy-npu-task-enter.py"
_spec = importlib.util.spec_from_file_location("infer_strategy_task_enter_demo", ENTRY)
mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = mod  # dataclass 解析字符串注解需要
_spec.loader.exec_module(mod)

CONTAINER_DIR = "/apps/suyantest/outputs/infer/infer_strategy_npu"


# --------------------------------------------------------------------------- 模拟 msmodeling 结果

def _side(size: int) -> dict:
    return {
        "instances": 2,
        "devices_per_instance": size,
        "devices_total": size * 2,
        "parallel": {"tp": size},
        "parallel_str": f"tp{size}",
        "batch_size": 16,
        "concurrency": 64,
        "qps_per_instance": 1.234,
        "qps_allocated": 2.468,
        "ttft_ms": 1234.567,
        "tpot_ms": 45.678,
        "tokens_per_s_per_instance": 1234.567,
        "tokens_per_s_per_device": 154.321,
        "tokens_per_s_cluster": 2469.135,
    }


def _agg_candidate() -> dict:
    return {
        "framework": "msmodeling",
        "throughput_tokens_per_s": 12345.678,
        "throughput_tokens_per_s_per_device": 385.802,
        "throughput_req_per_s": 12.345,
        "ttft_ms": 1234.567,
        "tpot_ms": 45.678,
        "meets_slo": True,
        "concurrency": 64,
        "batch_size": 16,
        "num_devices": 32,
        "parallel": {"tp": 8},
        "raw": {"internal": 1},
    }


def _pd_candidate(size: int) -> dict:
    return {
        "framework": "msmodeling",
        "label": f"P{size}D{size}",
        "decode_tps_per_instance": 100.0 + size,
        "throughput_tokens_per_s": 9876.543,
        "prefill": _side(size),
        "decode": _side(size),
        "raw": {"internal": 1},
    }


def _framework_run(*, agg: list | None, pd: list, grid: bool) -> dict:
    return {
        "framework": "msmodeling",
        "status": "ok",
        "message": None,
        "commands": [
            "python -m cli.inference.throughput_optimizer models/GLM5.2 "
            "--device ATLAS_950_486T_96G --num-devices 32 --tpot-limit 100 --ttft-limit 20000"
        ],
        "artifacts": {"pd_ratio_grid": [{"label": "P8D8"}] if grid else []},
        "agg": {"candidates": agg} if agg is not None else None,
        "disagg": None,
        "disagg_variants": [],
        "pd_ratio_candidates": pd,
    }


def virtual_clock(elapsed_seconds: int):
    """演示专用虚拟时钟：模拟十几秒的任务耗时，避免真的等待；瞬时报错耗时为 0。"""
    base = datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0)
    last = base + timedelta(seconds=elapsed_seconds)
    stamps = iter([
        base.isoformat(timespec="seconds"),
        base.isoformat(timespec="seconds"),
        last.isoformat(timespec="seconds"),
    ])
    return lambda: next(stamps, last.isoformat(timespec="seconds"))


def make_recommend(scenario: str, run_dir: Path):
    """返回模拟的 recommend；success / empty / fatal 三种行为，耗时由虚拟时钟模拟。"""

    def recommend(request, *, log=None, run_id=None, on_progress=None):
        if scenario == "fatal":
            raise RuntimeError("failed to import msmodeling: No module named 'msmodeling'")
        if scenario == "empty":
            fw = _framework_run(agg=[], pd=[], grid=False)
        elif request.serving_modes == ["agg"]:
            fw = _framework_run(agg=[_agg_candidate()], pd=[], grid=True)
        else:
            size = request.pd_ratio_sizes[0]
            fw = _framework_run(agg=None, pd=[_pd_candidate(size)], grid=True)
        data = {
            "run_id": "demo-run",
            "status": "ok",
            "routing": {"framework": "msmodeling", "commands": []},
            "frameworks": [fw],
            "errors": [],
            "artifacts": {"run_dir": str(run_dir)},
        }
        return SimpleNamespace(to_dict=lambda: data)

    return recommend


# --------------------------------------------------------------------------- 案例定义

def base_args(overrides: dict) -> list[str]:
    params = {
        "--model": "DeepSeek-V4.1-Flash-MXFP4",
        "--device": "950DT",
        "--num_devices": "32",
        "--input-length": "4096",
        "--output-length": "1024",
        "--tpot-limit": "100",
        "--ttft-limit": "20",
        "--serving_modes": "0",
    }
    params.update(overrides)
    argv: list[str] = []
    for key, value in params.items():
        if value is None:  # 不传该参数，用于验证默认值
            continue
        argv.extend([key, str(value)])
    return argv


CASES = [
    {
        "name": "agg 成功（serving_modes=0）",
        "scenario": "success",
        "overrides": {"--serving_modes": "0"},
        "note": "全局 0/success，内层 0/success；agg 1 条候选，disagg 为 {}",
    },
    {
        "name": "disagg 成功（serving_modes=1）",
        "scenario": "success",
        "overrides": {"--model": "GLM-5.2-w4a4c8-mxfp4", "--serving_modes": "1"},
        "note": "三种对称 PD 组合各跑一次；agg 为 {}，disagg 取 D 侧吞吐最高的 1 条",
    },
    {
        "name": "不传 ttft/tpot（serving_modes=0）",
        "scenario": "success",
        "overrides": {
            "--model": "GLM-5.2-BF16", "--device": "910B2", "--num_devices": "64",
            "--ttft-limit": None, "--tpot-limit": None, "--serving_modes": "0",
        },
        "note": "默认 20s / 100ms；BF16 方案不下发 group-size，但 request 回显默认 32",
    },
    {
        "name": "非 MXFP4 + disagg（serving_modes=1）",
        "scenario": "success",
        "overrides": {"--model": "DeepSeek-R1-FP8", "--device": "910B4", "--serving_modes": "1"},
        "note": "request 回显默认 mxfp4-group-size=32（不下发）；量化动作 FP8/FP8/FP8",
    },
    {
        "name": "无结果 empty（serving_modes=0）",
        "scenario": "empty",
        "overrides": {"--ttft-limit": "0.001", "--tpot-limit": "1", "--serving_modes": "0"},
        "note": "极端 SLO 无可行解：全局 0/success，内层 3/empty；result 两侧统一 {}",
    },
    {
        "name": "无结果 empty（serving_modes=1）",
        "scenario": "empty",
        "overrides": {"--num_devices": "8", "--serving_modes": "1"},
        "note": "8 卡时 P8D8 也被框架过滤：全局 0/success，内层 3/empty；result 两侧统一 {}",
    },
    {
        "name": "失败：模型不在白名单",
        "scenario": "success",
        "overrides": {"--model": "Unknown-Model", "--serving_modes": "0"},
        "note": "入参校验失败：全局 1/failed，内层 1/failed",
    },
    {
        "name": "失败：设备代号非法",
        "scenario": "success",
        "overrides": {"--device": "950X", "--serving_modes": "1"},
        "note": "只支持 910B2 / 910B4 / 950DT",
    },
    {
        "name": "失败：卡数非 8 倍数",
        "scenario": "success",
        "overrides": {"--num_devices": "12", "--serving_modes": "0"},
        "note": "num_devices 必须是 8 的倍数且 [8,256]",
    },
    {
        "name": "失败：TTFT 越界",
        "scenario": "success",
        "overrides": {"--ttft-limit": "0", "--serving_modes": "0"},
        "note": "ttft_limit 取值范围 [0.001,60] 秒",
    },
    {
        "name": "失败：输出长度越界",
        "scenario": "success",
        "overrides": {"--output-length": "131073", "--serving_modes": "1"},
        "note": "input/output_length 取值范围 [128,131072]",
    },
    {
        "name": "失败：result-file 非 .json",
        "scenario": "success",
        "overrides": {"--serving_modes": "1"},
        "result_suffix": ".txt",
        "note": "--result-file 必须是 .json；失败时仍写出完整 failed JSON",
    },
    {
        "name": "失败：msmodeling 执行异常",
        "scenario": "fatal",
        "overrides": {"--serving_modes": "0"},
        "note": "执行期异常：全局 1/failed，内层 1/failed，异常栈写入日志",
    },
]


# --------------------------------------------------------------------------- 运行与展示

class _Tee:
    """同时写控制台与 UTF-8 报告文件（避免终端编码影响示例内容）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except (UnicodeEncodeError, OSError, ValueError):
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except (OSError, ValueError):
                pass


def describe_result(value) -> str:
    if not isinstance(value, dict) or not value:
        return "{}"
    candidates = value.get("candidates")
    return f"{len(candidates)} candidate(s)" if isinstance(candidates, list) else str(value)


def _next_run_tag(out_dir: Path) -> str:
    """无标记的旧示例视为第 1 轮；自动递增出 r02/r03...，保留历史结果。"""
    used: set[int] = {1} if (out_dir / "case_01.json").exists() else set()
    for path in out_dir.glob("case_01_r[0-9][0-9].json"):
        used.add(int(path.stem[len("case_01_r"):]))
    return f"r{max(used, default=0) + 1:02d}"


def _tagged_name(stem: str, suffix: str, tag: str) -> str:
    return f"{stem}_{tag}{suffix}" if tag else f"{stem}{suffix}"


def run_case(index: int, case: dict, out_dir: Path, tag: str) -> dict:
    suffix = case.get("result_suffix", ".json")
    label = f"case_{index:02d}" + (f"_{tag}" if tag else "")
    log_path = out_dir / f"{label}.log"
    result_path = out_dir / f"{label}{suffix}"
    run_dir = (
        out_dir / "artifacts" / tag / f"case_{index:02d}"
        if tag else out_dir / "artifacts" / f"case_{index:02d}"
    )
    base = base_args(case["overrides"])
    argv = base + ["--log-file", str(log_path), "--result-file", str(result_path)]
    container_argv = base + [
        "--log-file", f"{CONTAINER_DIR}/{label}.log",
        "--result-file", f"{CONTAINER_DIR}/{label}.json",
    ]

    elapsed = 0 if case["name"].startswith("失败") else 13
    with (
        patch.object(mod, "load_recommend", return_value=make_recommend(case["scenario"], run_dir)),
        patch.object(mod, "_now_iso", side_effect=virtual_clock(elapsed)),
    ):
        exit_code = mod.main(argv)

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    inner = payload.get("common", {}).get("infer_strategy_npu", {})
    data = inner.get("data", {})
    result = data.get("result", {})
    request = data.get("request", {})

    print("=" * 100)
    print(f"[{index:02d}] {case['name']}")
    print(f"  说明: {case['note']}")
    print(f"  本地命令: python infer-strategy-npu-task-enter.py {shlex.join(argv)}")
    print(f"  容器命令: python infer-strategy-npu-task-enter.py {shlex.join(container_argv)}")
    print(f"  结果文件: {result_path}")
    print(f"  日志: {log_path}")
    print(f"  全局  code={payload.get('code')}  status={payload.get('status')}  message={payload.get('message')}")
    print(f"  内层  code={inner.get('code')}  status={inner.get('status')}  message={inner.get('message')}")
    print(f"  result.agg={describe_result(result.get('agg'))}  result.disagg={describe_result(result.get('disagg'))}")
    print(
        "  request: "
        f"model={request.get('model')}, device={request.get('device')}, "
        f"num_devices={request.get('num_devices')}, num_nodes={request.get('num_nodes')}, "
        f"ttft_limit={request.get('ttft_limit')}, tpot_limit={request.get('tpot_limit')}, "
        f"serving_modes={request.get('serving_modes')}, topk={request.get('topk')}"
    )
    print(f"  退出码: {exit_code}")

    return {
        "index": index,
        "name": case["name"],
        "note": case["note"],
        "local_command": f"python infer-strategy-npu-task-enter.py {shlex.join(argv)}",
        "container_command": f"python infer-strategy-npu-task-enter.py {shlex.join(container_argv)}",
        "result_file": str(result_path),
        "log_file": str(log_path),
        "exit_code": exit_code,
        "global": {
            "code": payload.get("code"),
            "status": payload.get("status"),
            "message": payload.get("message"),
        },
        "inner": {
            "code": inner.get("code"),
            "status": inner.get("status"),
            "message": inner.get("message"),
        },
        "result": {
            "agg": describe_result(result.get("agg")),
            "disagg": describe_result(result.get("disagg")),
        },
        "request": request,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="依次运行一次性任务入口的响应示例")
    parser.add_argument("--out-dir", default=None, help="结果/日志输出目录（默认系统临时目录）")
    parser.add_argument(
        "--increment", action="store_true", help="产物加 rNN 标记并自动递增，保留历史示例"
    )
    options = parser.parse_args(argv)
    out_dir = (
        Path(options.out_dir).resolve()
        if options.out_dir
        else Path(tempfile.mkdtemp(prefix="infer_strategy_task_enter_examples_"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _next_run_tag(out_dir) if options.increment else ""
    report_path = out_dir / _tagged_name("responses", ".txt", tag)
    summary_path = out_dir / _tagged_name("summary", ".json", tag)

    with open(report_path, "w", encoding="utf-8") as report:
        tee = _Tee(sys.stdout, report)
        with contextlib.redirect_stdout(tee):
            print("说明：本机无 msmodeling/NPU，load_recommend 已注入模拟结果；容器命令可直接真实执行。")
            print("仅覆盖 serving_modes=0/1，不测 2。")
            summaries = [run_case(index, case, out_dir, tag) for index, case in enumerate(CASES, 1)]

            summary_path.write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("=" * 100)
            print(f"{'#':>2} | {'案例':<32} | {'全局':<14} | {'内层':<14} | 退出码")
            print("-" * 100)
            for item in summaries:
                g, i = item["global"], item["inner"]
                print(
                    f"{item['index']:>2} | {item['name']:<32} | "
                    f"{g['code']}/{str(g['status']):<12} | "
                    f"{i['code']}/{str(i['status']):<12} | {item['exit_code']}"
                )
            pattern = f"case_*_{tag}.json" if tag else "case_*.json"
            print(f"\n本轮标记：{tag or '无（覆盖同名文件）'}")
            print(f"完整响应示例：{out_dir}/{pattern}")
            print(f"UTF-8 文本报告：{report_path}")
            print(f"汇总 JSON：{summary_path}")

    import logging

    logger = logging.getLogger(mod._LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
