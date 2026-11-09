#!/usr/bin/env python3
"""推理策略寻优·容器一次性任务入口（接口文档 V1.0.0 第 4 章）。

调用方式（容器内 CWD=/workspace）::

    python infer-strategy-npu-task-enter.py --model DeepSeek-V4.1-Flash-MXFP4 \
        --device 950DT --num_devices 32 --input-length 4096 --output-length 1024 \
        --tpot-limit 100 --ttft-limit 20 --serving_modes 2 \
        --log-file /apps/suyantest/outputs/infer/infer_strategy_npu/01.log \
        --result-file /apps/suyantest/outputs/infer/infer_strategy_npu/01.json

约束（见 docs/plans/INFER_STRATEGY_NPU_TASK_ENTER_DESIGN.md v3）：
- 结果 JSON 只在全局 code 确定为 0/1 后写一次；运行期只写日志。
- request 字段类型严格按 4.1.3 参数说明表；示例中的字符串数值不作准。
- 默认 PD 搜索空间只有 P8D8 / P16D16 / P32D32 三种对称组合，内核零改动。
- PD 分离按 D 节点吞吐 decode_tps_per_instance 排序，跨组合取 topk=1。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Sequence

# 镜像里已 pip install -e .；本地直接跑脚本时补上 src，保证 import 可用
_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_LOGGER_NAME = "infer_strategy_npu.task_enter"

# --------------------------------------------------------------------------- 配置表

@dataclass(frozen=True)
class ModelPlan:
    """平台 --model 白名单 -> 仓库模型 ID + 固定量化方案。"""

    model_id: str
    local_dir: str
    linear: str
    non_expert_linear: str
    attention: str
    mxfp4_group_size: str | None = None


MODEL_PLANS: dict[str, ModelPlan] = {
    "GLM-5.2-w4a4c8-mxfp4": ModelPlan("ZhipuAI/GLM-5.2", "GLM5.2", "MXFP4", "FP8", "FP8", "32"),
    "GLM-5.2-BF16": ModelPlan("ZhipuAI/GLM-5.2", "GLM5.2", "DISABLED", "DISABLED", "DISABLED"),
    "GLM-5.3-BF16": ModelPlan("ZhipuAI/GLM-5.3", "GLM5.3", "DISABLED", "DISABLED", "DISABLED"),
    "GLM-5.2-FP8": ModelPlan("ZhipuAI/GLM-5.2", "GLM5.2", "FP8", "DISABLED", "FP8"),
    "GLM-5.3-FP8": ModelPlan("ZhipuAI/GLM-5.3", "GLM5.3", "FP8", "DISABLED", "FP8"),
    "DeepSeek-R1-FP8": ModelPlan("deepseek-ai/DeepSeek-R1", "DeepSeek-R1", "FP8", "FP8", "FP8"),
    "DeepSeek-V4.1-Flash-MXFP4": ModelPlan(
        "deepseek-ai/DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash", "MXFP4", "FP8", "DISABLED", "32"
    ),
    "DeepSeek-V4-Pro-W8A8": ModelPlan(
        "deepseek-ai/DeepSeek-V4-Pro", "DeepSeek-V4-Pro", "W8A8_DYNAMIC", "W8A8_DYNAMIC", "INT8"
    ),
}

# 文档 4.1.3 预留值域（用于方案表自检，不扩大下发范围）
_LINEAR_VALUE_DOMAIN = frozenset(
    {
        "DISABLED", "W8A16_STATIC", "W8A8_STATIC", "W4A8_STATIC", "W8A16_DYNAMIC",
        "W8A8_DYNAMIC", "W4A8_DYNAMIC", "FP8", "MXFP4", "HIF8",
    }
)
_ATTENTION_VALUE_DOMAIN = frozenset({"DISABLED", "INT8", "FP8", "HIF8"})


@dataclass(frozen=True)
class DeviceSpec:
    """平台 --device 代号 -> msmodeling 设备名 + 单机卡数。"""

    ms_device: str
    gpus_per_node: int


DEVICE_SPECS: dict[str, DeviceSpec] = {
    "910B2": DeviceSpec("ATLAS_800_A2_376T_64G", 8),
    "910B4": DeviceSpec("ATLAS_800_A2_280T_64G", 8),
    "950DT": DeviceSpec("ATLAS_950_486T_96G", 8),
}

# 默认且唯一的 PD 搜索空间：按原始 P8D8 方式逐组单值调用，天然只有对称组合
PD_RATIO_SIZES: tuple[int, ...] = (8, 16, 32)
TOPK = 1
FIXED_JOBS = 32
FIXED_MAX_BATCHED_TOKENS = 8192
FIXED_BATCH_RANGE = (4, 512)
DEFAULT_TTFT_LIMIT_S = 20.0
DEFAULT_TPOT_LIMIT_MS = 100

AGG_FIELDS = (
    "throughput_tokens_per_s", "throughput_tokens_per_s_per_device", "throughput_req_per_s",
    "ttft_ms", "tpot_ms", "meets_slo", "concurrency", "batch_size", "num_devices",
)
SIDE_FIELDS = (
    "instances", "devices_per_instance", "devices_total", "batch_size", "concurrency",
    "ttft_ms", "tpot_ms", "tokens_per_s_per_instance", "tokens_per_s_per_device",
    "tokens_per_s_cluster",
)
PARALLEL_KEYS = ("tp", "pp", "dp", "ep")

# 这些失败属于「执行失败」，其余（卡数不足 / SLO 无解）属于「无结果」
_FATAL_MARKERS = (
    "failed to import msmodeling",
    "rejected by msmodeling device check",
    "argument parsing exited",
    "unsupported --model",
    "unsupported --framework",
    "No module named",
)


# --------------------------------------------------------------------------- 工具函数

def _now_iso() -> str:
    """ISO8601 带时区偏移（+08:00），不依赖容器 tzdata。"""
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _clean(value: Any) -> Any:
    """numpy 标量 -> 内置类型；NaN/Inf -> None；其余原样（bool 保持 bool）。"""
    if value is None or isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:  # pragma: no cover - 防御性
            return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, str)):
        return value
    return None


def _num_key(value: Any) -> float:
    item = _clean(value)
    return float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else float("-inf")


def _option_value(argv: Sequence[str], names: Sequence[str]) -> str | None:
    """从原始 argv 里尽力取某个选项的值（用于校验失败时的回显/落盘）。"""
    names = tuple(names)
    for index, token in enumerate(argv):
        if token in names and index + 1 < len(argv):
            return argv[index + 1]
        for name in names:
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
    return None


def _best_effort_path(argv: Sequence[str], names: Sequence[str]) -> Path | None:
    raw = _option_value(argv, names)
    if not raw or "\x00" in raw:
        return None
    try:
        path = Path(raw).expanduser()
        return None if path.exists() and path.is_dir() else path.resolve()
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- 参数解析与校验

class _Parser(argparse.ArgumentParser):
    """把 argparse 的错误转成 ValueError，保证非法入参也走统一失败落盘。"""

    def error(self, message: str) -> None:
        raise ValueError(f"参数解析失败：{message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="infer-strategy-npu-task-enter.py",
        description="推理策略寻优任务创建（接口文档 V1.0.0 第 4 章）",
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, help="模型白名单，见 4.1.2")
    parser.add_argument("--device", required=True, help="910B2 / 910B4 / 950DT")
    parser.add_argument("--num_devices", "--num-devices", dest="num_devices", required=True)
    parser.add_argument("--input_length", "--input-length", dest="input_length", required=True)
    parser.add_argument("--output_length", "--output-length", dest="output_length", required=True)
    parser.add_argument(
        "--ttft_limit", "--ttft-limit", dest="ttft_limit", default=str(DEFAULT_TTFT_LIMIT_S),
        help=f"TTFT 上限（秒），默认 {DEFAULT_TTFT_LIMIT_S}",
    )
    parser.add_argument(
        "--tpot_limit", "--tpot-limit", dest="tpot_limit", default=str(DEFAULT_TPOT_LIMIT_MS),
        help=f"TPOT 上限（毫秒），默认 {DEFAULT_TPOT_LIMIT_MS}",
    )
    parser.add_argument(
        "--serving_modes", "--serving-modes", dest="serving_modes", default="1",
        help="0=agg，1=disagg，2=agg+disagg，默认 1",
    )
    parser.add_argument("--log-file", "--log_file", dest="log_file", required=True)
    parser.add_argument("--result-file", "--result_file", dest="result_file", required=True)
    return parser


def _to_int(name: str, raw: Any, *, minimum: int, maximum: int | None = None,
            multiple_of: int | None = None) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数，实际为 {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name}={value} 超出取值范围 {bound}")
    if multiple_of and value % multiple_of:
        raise ValueError(f"{name}={value} 必须是 {multiple_of} 的倍数")
    return value


def _to_float(name: str, raw: Any, *, minimum: float, maximum: float) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是浮点数，实际为 {raw!r}") from exc
    if not math.isfinite(value) or not (minimum <= value <= maximum):
        raise ValueError(f"{name}={value} 超出取值范围 [{minimum}, {maximum}]")
    return value


def _check_file_path(name: str, raw: Any, *, must_be_json: bool) -> Path:
    if not str(raw).strip() or "\x00" in str(raw):
        raise ValueError(f"{name} 不是合法的文件路径")
    path = Path(str(raw)).expanduser()
    if path.exists() and path.is_dir():
        raise ValueError(f"{name} 指向了目录：{path}")
    if must_be_json and path.suffix.lower() != ".json":
        raise ValueError(f"{name} 必须是 .json 文件：{path}")
    return path.resolve()


@dataclass(frozen=True)
class TaskConfig:
    model: str
    device: str
    num_devices: int
    input_length: int
    output_length: int
    ttft_limit_s: float
    tpot_limit_ms: int
    ttft_limit_ms: int
    serving_modes: int
    log_file: Path
    result_file: Path
    plan: ModelPlan
    spec: DeviceSpec
    num_nodes: int
    ttft_defaulted: bool
    tpot_defaulted: bool
    raw_args: dict[str, str]


def parse_and_validate(argv: Sequence[str]) -> TaskConfig:
    args = build_parser().parse_args(list(argv))

    plan = MODEL_PLANS.get(args.model)
    if plan is None:
        raise ValueError(f"--model={args.model!r} 不在白名单内；支持：{', '.join(MODEL_PLANS)}")
    if plan.linear not in _LINEAR_VALUE_DOMAIN or plan.non_expert_linear not in _LINEAR_VALUE_DOMAIN:
        raise ValueError(f"模型 {args.model} 的线性层量化动作超出 4.1.3 预留值域")
    if plan.attention not in _ATTENTION_VALUE_DOMAIN:
        raise ValueError(f"模型 {args.model} 的注意力量化动作超出 4.1.3 预留值域")

    spec = DEVICE_SPECS.get(args.device)
    if spec is None:
        raise ValueError(f"--device={args.device!r} 不在支持范围内；支持：{', '.join(DEVICE_SPECS)}")

    num_devices = _to_int("--num_devices", args.num_devices, minimum=8, maximum=256, multiple_of=8)
    input_length = _to_int("--input_length", args.input_length, minimum=128, maximum=131072)
    output_length = _to_int("--output_length", args.output_length, minimum=128, maximum=131072)
    ttft_limit_s = _to_float("--ttft_limit", args.ttft_limit, minimum=0.001, maximum=60.0)
    tpot_limit_ms = _to_int("--tpot_limit", args.tpot_limit, minimum=1, maximum=1000)
    serving_modes = _to_int("--serving_modes", args.serving_modes, minimum=0, maximum=2)
    if serving_modes not in (0, 1, 2):  # 防御：_to_int 已限制，保留显式范围约束
        raise ValueError(f"--serving_modes={serving_modes} 只允许 0/1/2")

    log_file = _check_file_path("--log-file", args.log_file, must_be_json=False)
    result_file = _check_file_path("--result-file", args.result_file, must_be_json=True)

    num_nodes = num_devices // spec.gpus_per_node
    if num_nodes <= 0 or num_devices % spec.gpus_per_node:
        raise ValueError(f"num_nodes 必须为正整数（num_devices={num_devices}, 单机 {spec.gpus_per_node} 卡）")

    # 秒 -> 毫秒，向下取整；用 Decimal 避免 0.001 * 1000 的浮点误差
    ttft_limit_ms = int((Decimal(str(ttft_limit_s)) * 1000).to_integral_value(rounding=ROUND_FLOOR))

    return TaskConfig(
        model=args.model,
        device=args.device,
        num_devices=num_devices,
        input_length=input_length,
        output_length=output_length,
        ttft_limit_s=ttft_limit_s,
        tpot_limit_ms=tpot_limit_ms,
        ttft_limit_ms=ttft_limit_ms,
        serving_modes=serving_modes,
        log_file=log_file,
        result_file=result_file,
        plan=plan,
        spec=spec,
        num_nodes=num_nodes,
        ttft_defaulted=_option_value(argv, ("--ttft_limit", "--ttft-limit")) is None,
        tpot_defaulted=_option_value(argv, ("--tpot_limit", "--tpot-limit")) is None,
        raw_args={str(k): str(v) for k, v in vars(args).items()},
    )


# --------------------------------------------------------------------------- request 回显

def _empty_echo() -> dict[str, Any]:
    """校验失败时也要保证 4.1.3 的 request 字段齐全（值尽力填充，无法确定则为 null）。"""
    keys = (
        "model", "device", "num_devices", "num_nodes", "input_length", "output_length",
        "ttft_limit", "tpot_limit", "serving_modes", "quantize-linear-action",
        "quantize-non-expert-linear-action", "quantize-attention-action", "topk",
    )
    return {key: None for key in keys}


def build_request_echo(config: TaskConfig) -> dict[str, Any]:
    """严格按 4.1.3 参数说明表的类型输出。"""
    echo: dict[str, Any] = {
        "model": str(config.model),
        "device": str(config.device),
        "num_devices": int(config.num_devices),
        "num_nodes": int(config.num_nodes),
        "input_length": int(config.input_length),
        "output_length": int(config.output_length),
        "ttft_limit": float(config.ttft_limit_s),
        "tpot_limit": int(config.tpot_limit_ms),
        "serving_modes": int(config.serving_modes),
        "quantize-linear-action": str(config.plan.linear),
        "quantize-non-expert-linear-action": str(config.plan.non_expert_linear),
    }
    if config.plan.mxfp4_group_size:
        echo["mxfp4-group-size"] = str(config.plan.mxfp4_group_size)
    echo["quantize-attention-action"] = str(config.plan.attention)
    echo["topk"] = int(TOPK)
    return echo


def best_effort_request_echo(argv: Sequence[str]) -> dict[str, Any]:
    echo = _empty_echo()
    try:
        args = build_parser().parse_args(list(argv))
    except (ValueError, SystemExit):
        return echo

    def maybe_int(raw: Any) -> int | None:
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None

    def maybe_float(raw: Any) -> float | None:
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return None

    echo["model"] = args.model
    echo["device"] = args.device
    echo["num_devices"] = maybe_int(args.num_devices)
    echo["input_length"] = maybe_int(args.input_length)
    echo["output_length"] = maybe_int(args.output_length)
    echo["ttft_limit"] = maybe_float(args.ttft_limit)
    echo["tpot_limit"] = maybe_int(args.tpot_limit)
    echo["serving_modes"] = maybe_int(args.serving_modes)
    echo["topk"] = int(TOPK)
    spec = DEVICE_SPECS.get(args.device)
    if spec and echo["num_devices"]:
        echo["num_nodes"] = echo["num_devices"] // spec.gpus_per_node
    plan = MODEL_PLANS.get(args.model)
    if plan:
        echo["quantize-linear-action"] = plan.linear
        echo["quantize-non-expert-linear-action"] = plan.non_expert_linear
        echo["quantize-attention-action"] = plan.attention
        if plan.mxfp4_group_size:
            echo["mxfp4-group-size"] = plan.mxfp4_group_size
    return echo


# --------------------------------------------------------------------------- 日志

def setup_logging(log_file: Path | None) -> tuple[logging.Logger, bool]:
    """返回 (logger, 文件日志是否可用)；stdout 始终可用。"""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - 防御性
            pass
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is None:
        return logger, False
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger, True
    except OSError as exc:
        logger.error("[logging] --log-file 不可写：%s", exc)
        return logger, False


# --------------------------------------------------------------------------- 执行编排

@dataclass
class RunOutcome:
    agg_candidates: list[dict[str, Any]] = field(default_factory=list)
    disagg_candidates: list[dict[str, Any]] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    run_dirs: list[Path] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    per_combo: list[str] = field(default_factory=list)
    fatal: bool = False


@dataclass(frozen=True)
class Status:
    global_code: int
    global_status: str
    global_message: str
    inner_code: int
    inner_status: str
    inner_message: str


def build_request(config: TaskConfig, serving_modes: list[str], pd_size: int) -> Any:
    """构造 RecommendRequest；每次调用都新建，避免 service 内部 mutate 互相影响。"""
    from infer_strategy_npu.schema import RecommendRequest

    extra = None
    if config.plan.mxfp4_group_size:
        extra = ["--mxfp4-group-size", str(config.plan.mxfp4_group_size)]
    return RecommendRequest(
        model=config.plan.model_id,
        device=config.spec.ms_device,
        num_devices=config.num_devices,
        input_length=config.input_length,
        output_length=config.output_length,
        tpot_limit=config.tpot_limit_ms,
        ttft_limit=config.ttft_limit_ms,
        quantize="ENABLE",
        mode="recommend",
        top_n=TOPK,
        framework="msmodeling",
        local_model_root=None,
        out_dir=None,
        quantize_linear_action=config.plan.linear,
        quantize_non_expert_linear_action=config.plan.non_expert_linear,
        quantize_attention_action=config.plan.attention,
        speculative_method=None,
        num_speculative_tokens=None,
        acceptance_length=None,
        num_draft_layers=None,
        tp_sizes=None,
        jobs=FIXED_JOBS,
        max_search_combinations=None,
        compile_model=False,
        serving_modes=list(serving_modes),
        pd_ratio_sizes=[pd_size],
        batch_range=list(FIXED_BATCH_RANGE),
        max_batched_tokens=FIXED_MAX_BATCHED_TOKENS,
        extra_msmodeling_args=extra,
        dry_run=False,
    )


def load_recommend() -> Any:
    """延迟导入重型依赖，便于离线单测替换。"""
    from infer_strategy_npu.service import recommend

    return recommend


def _is_fatal_run(data: dict[str, Any]) -> bool:
    """区分「执行失败」与「卡数不足 / SLO 无解导致的无结果」。"""
    routing = data.get("routing") or {}
    if routing and routing.get("framework") is None:
        return True
    texts = [str(routing.get("reason") or "")]
    for fw in data.get("frameworks") or []:
        if isinstance(fw, dict):
            texts.append(str(fw.get("message") or ""))
    texts.extend(str(err) for err in (data.get("errors") or []))
    joined = " | ".join(texts)
    return any(marker in joined for marker in _FATAL_MARKERS)


def run_all(config: TaskConfig, logger: logging.Logger) -> RunOutcome:
    """agg 一次 + 三种对称 PD 组合各一次；卡数不足由框架 pd_ratio_grid 自动过滤。"""
    outcome = RunOutcome()
    recommend = load_recommend()

    calls: list[tuple[str, list[str], int]] = []
    if config.serving_modes in (0, 2):
        calls.append(("agg", ["agg"], PD_RATIO_SIZES[0]))
    if config.serving_modes in (1, 2):
        for size in PD_RATIO_SIZES:
            calls.append((f"P{size}D{size}", ["disagg"], size))

    logger.info(
        "[plan] 默认 PD 搜索空间：%s；本次共 %d 次 recommend 调用",
        ", ".join(f"P{size}D{size}" for size in PD_RATIO_SIZES), len(calls),
    )
    for label, modes, size in calls:
        logger.info(
            "[running] 子任务 %s 开始：serving_modes=%s, pd_ratio_sizes=[%d]（status=2/running）",
            label, modes, size,
        )
        started = time.monotonic()
        try:
            request = build_request(config, modes, size)

            def on_progress(stage: str, done: int, total: int, _label: str = label) -> None:
                logger.info("[running] %s %s %s/%s（status=2/running）", _label, stage, done, total)

            result = recommend(request, log=logger, on_progress=on_progress)
            data = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
        except Exception as exc:  # 单次调用异常不阻断其余组合
            logger.exception("[failed] 子任务 %s 异常（%.2fs）", label, time.monotonic() - started)
            outcome.messages.append(f"{label}: {exc}")
            outcome.per_combo.append(f"{label}: 异常 {exc}")
            outcome.fatal = True
            continue

        elapsed = time.monotonic() - started
        frameworks = data.get("frameworks") or []
        fw = frameworks[0] if frameworks and isinstance(frameworks[0], dict) else {}
        agg = fw.get("agg")
        agg_count = 0
        if isinstance(agg, dict):
            agg_candidates = [c for c in (agg.get("candidates") or []) if isinstance(c, dict)]
            agg_count = len(agg_candidates)
            outcome.agg_candidates.extend(agg_candidates)
        pd_candidates = [c for c in (fw.get("pd_ratio_candidates") or []) if isinstance(c, dict)]
        outcome.disagg_candidates.extend(pd_candidates)

        commands = list(fw.get("commands") or []) + list((data.get("routing") or {}).get("commands") or [])
        for command in dict.fromkeys(str(item) for item in commands):
            outcome.commands.append(command)
            logger.info("[command] %s %s", label, command)

        grid = (fw.get("artifacts") or {}).get("pd_ratio_grid")
        if label.startswith("P") and not grid:
            logger.info("[filter] %s 被框架自动过滤（卡数不足或无可行组合）", label)
            outcome.per_combo.append(f"{label}: 已过滤")
        else:
            outcome.per_combo.append(f"{label}: agg={agg_count}, pd={len(pd_candidates)}")

        artifacts = data.get("artifacts") or {}
        run_dir = artifacts.get("run_dir")
        if run_dir:
            outcome.run_dirs.append(Path(str(run_dir)))
            logger.info("[artifacts] %s run_id=%s run_dir=%s", label, data.get("run_id"), run_dir)
        message = fw.get("message")
        if message:
            outcome.messages.append(f"{label}: {message}")
            logger.info("[framework] %s message=%s", label, message)
        for err in data.get("errors") or []:
            outcome.messages.append(f"{label}: {err}")
        if _is_fatal_run(data):
            outcome.fatal = True
        logger.info("[running] 子任务 %s 结束（%.2fs）：%s", label, elapsed, outcome.per_combo[-1])

    logger.info(
        "[summary] 汇总 agg 候选 %d 条、disagg 候选 %d 条；按 D 侧吞吐 "
        "decode_tps_per_instance 降序后取 topk=%d",
        len(outcome.agg_candidates), len(outcome.disagg_candidates), TOPK,
    )
    return outcome


# --------------------------------------------------------------------------- 结果投影

def project_parallel(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    return {key: _clean(data.get(key)) for key in PARALLEL_KEYS}


def project_agg_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    out = {key: _clean(raw.get(key)) for key in AGG_FIELDS}
    out["parallel"] = project_parallel(raw.get("parallel"))
    return out


def project_side(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    out = {key: _clean(data.get(key)) for key in SIDE_FIELDS}
    out["parallel"] = project_parallel(data.get("parallel"))
    return out


def project_disagg_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefill": project_side(raw.get("prefill")),
        "decode": project_side(raw.get("decode")),
    }


def assemble_payload(
    request_echo: dict[str, Any],
    outcome: RunOutcome,
    times: tuple[str, str, str],
    status: Status,
) -> dict[str, Any]:
    """按 4.1.3 组装契约 JSON；只输出文档字段，不夹带内部信息。"""
    modes = request_echo.get("serving_modes")
    wants_agg = modes in (0, 2)
    wants_disagg = modes in (1, 2)

    agg_best = sorted(
        outcome.agg_candidates,
        key=lambda c: _num_key(c.get("throughput_tokens_per_s")),
        reverse=True,
    )[:TOPK]
    disagg_best = sorted(
        outcome.disagg_candidates,
        key=lambda c: _num_key(c.get("decode_tps_per_instance")),
        reverse=True,
    )[:TOPK]

    created_at, started_at, finished_at = times
    return {
        "code": int(status.global_code),
        "status": status.global_status,
        "message": status.global_message,
        "common": {
            "infer_strategy_npu": {
                "code": int(status.inner_code),
                "status": status.inner_status,
                "message": status.inner_message,
                "data": {
                    "created_at": created_at,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "request": request_echo,
                    "result": {
                        "agg": {"candidates": [project_agg_candidate(c) for c in agg_best]}
                        if wants_agg else {},
                        "disagg": {"candidates": [project_disagg_candidate(c) for c in disagg_best]}
                        if wants_disagg else {},
                    },
                },
            }
        },
    }


# --------------------------------------------------------------------------- 落盘与入口

def write_result_file(path: Path, payload: dict[str, Any]) -> None:
    """终态只写一次：同目录临时文件 + os.replace 原子替换。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - 防御性
            pass


def write_internal_traces(
    outcome: RunOutcome,
    payload: dict[str, Any],
    request_echo: dict[str, Any],
    config: TaskConfig | None,
    logger: logging.Logger,
) -> None:
    """默认产物目录里补平台上下文与契约 JSON 副本，不改变 --result-file 内容。"""
    trace = {
        "platform_request": request_echo,
        "raw_args": config.raw_args if config else {},
        "num_nodes": config.num_nodes if config else None,
        "gpus_per_node": config.spec.gpus_per_node if config else None,
        "ttft_limit_ms": config.ttft_limit_ms if config else None,
        "pd_ratio_sizes": list(PD_RATIO_SIZES),
        "topk": TOPK,
    }
    for run_dir in outcome.run_dirs:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "platform_request.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("[artifacts] platform_request.json 写入失败：%s", exc)
    if outcome.run_dirs:
        try:
            outcome.run_dirs[0].mkdir(parents=True, exist_ok=True)
            (outcome.run_dirs[0] / "platform_result.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("[artifacts] platform_result.json 写入失败：%s", exc)


def _log_plan(config: TaskConfig, logger: logging.Logger) -> None:
    logger.info("[validate] 入参校验通过：%s", config.raw_args)
    logger.info(
        "[mapping] model=%s -> model_id=%s, local_dir=%s",
        config.model, config.plan.model_id, config.plan.local_dir,
    )
    logger.info(
        "[mapping] device=%s -> %s, gpus_per_node=%d, num_nodes=%d",
        config.device, config.spec.ms_device, config.spec.gpus_per_node, config.num_nodes,
    )
    logger.info(
        "[mapping] quantize=ENABLE linear=%s non_expert=%s attention=%s mxfp4-group-size=%s",
        config.plan.linear, config.plan.non_expert_linear, config.plan.attention,
        config.plan.mxfp4_group_size or "不下发/不回显",
    )
    logger.info(
        "[mapping] ttft_limit=%s s -> %d ms%s；tpot_limit=%d ms%s；serving_modes=%d；num_devices=%d",
        config.ttft_limit_s, config.ttft_limit_ms,
        "（默认值）" if config.ttft_defaulted else "（显式传入）",
        config.tpot_limit_ms,
        "（默认值）" if config.tpot_defaulted else "（显式传入）",
        config.serving_modes, config.num_devices,
    )
    logger.info(
        "[mapping] 固定下发 jobs=%d, max_batched_tokens=%d, batch_range=%s；投机参数不下发；out_dir 用框架默认值",
        FIXED_JOBS, FIXED_MAX_BATCHED_TOKENS, list(FIXED_BATCH_RANGE),
    )
    logger.info("[running] 任务状态 2/running；--result-file 仅在全局 code 确定后写入")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    created_at = _now_iso()
    started_at = created_at

    # 日志先就位，保证入参校验失败也能落盘；只写 stdout 兜底时仍会继续完成任务
    log_path = _best_effort_path(raw_argv, ("--log-file", "--log_file"))
    logger, log_ok = setup_logging(log_path)

    config: TaskConfig | None = None
    outcome = RunOutcome()
    echo: dict[str, Any] | None = None
    status: Status | None = None

    try:
        config = parse_and_validate(raw_argv)
    except ValueError as exc:
        logger.error("[validate] 入参校验失败：%s", exc)
        echo = best_effort_request_echo(raw_argv)
        message = f"入参不对：{exc}"
        status = Status(1, "failed", message, 1, "failed", message)
    else:
        echo = build_request_echo(config)
        if not log_ok:
            message = f"--log-file 不可写：{config.log_file}"
            logger.error("[logging] %s", message)
            status = Status(
                1, "failed", f"寻优任务执行失败：{message}", 1, "failed", f"任务报错失败：{message}"
            )
        else:
            _log_plan(config, logger)
            started_at = _now_iso()
            try:
                outcome = run_all(config, logger)
            except Exception as exc:
                logger.exception("[failed] 寻优执行异常")
                status = Status(
                    1, "failed", f"寻优任务执行失败：{exc}", 1, "failed", f"任务报错失败：{exc}"
                )
            else:
                if outcome.fatal:
                    detail = "；".join(outcome.messages[-3:]) or "msmodeling 执行失败"
                    status = Status(
                        1, "failed", f"寻优任务执行失败：{detail}",
                        1, "failed", f"任务报错失败：{detail}",
                    )
                elif outcome.agg_candidates or outcome.disagg_candidates:
                    status = Status(
                        0, "success", "寻优任务完成", 0, "success", "寻优策略任务完成，已找到结果"
                    )
                else:
                    status = Status(
                        0, "success", "寻优任务完成（无结果）",
                        3, "empty", "未找到满足 SLO 的部署策略",
                    )

    if status is None:  # pragma: no cover - 防御性兜底
        status = Status(1, "failed", "寻优任务执行失败", 1, "failed", "任务报错失败")

    finished_at = _now_iso()
    payload = assemble_payload(
        echo or _empty_echo(), outcome, (created_at, started_at, finished_at), status
    )

    result_path = config.result_file if config else _best_effort_path(
        raw_argv, ("--result-file", "--result_file")
    )
    if result_path is None:
        logger.error("[result] 无法确定 --result-file，结果无法落盘")
        return 1
    try:
        write_result_file(result_path, payload)
    except OSError as exc:
        logger.error("[result] 结果文件写入失败：%s", exc)
        return 1

    logger.info(
        "[done] 全局 code=%d status=%s，内层 code=%d status=%s；结果文件=%s；退出码=%d",
        status.global_code, status.global_status, status.inner_code, status.inner_status,
        result_path, status.global_code,
    )
    write_internal_traces(outcome, payload, echo or _empty_echo(), config, logger)
    return status.global_code


if __name__ == "__main__":
    sys.exit(main())

