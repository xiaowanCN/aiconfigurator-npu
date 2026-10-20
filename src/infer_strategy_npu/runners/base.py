"""框架 Runner 抽象基类与公共工具。"""

from __future__ import annotations

import abc
import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from infer_strategy_npu.models import ModelSpec
from infer_strategy_npu.output import RunArtifacts
from infer_strategy_npu.routing import RoutingDecision
from infer_strategy_npu.schema import QuantizeSpec, RecommendRequest


def build_quantize_spec(linear: str | None, attention: str | None) -> QuantizeSpec:
    return QuantizeSpec(linear=linear, attention=attention)


def meets_slo(ttft_ms: float | None, tpot_ms: float | None, request: RecommendRequest) -> bool | None:
    """按请求里的 TTFT/TPOT 上限判断是否满足 SLO；指标缺失时返回 None。"""
    if ttft_ms is None and tpot_ms is None:
        return None
    ok = True
    if ttft_ms is not None:
        ok = ok and ttft_ms <= float(request.ttft_limit)
    if tpot_ms is not None:
        ok = ok and tpot_ms <= float(request.tpot_limit)
    return ok


def sanitize_raw(mapping) -> dict:
    """把 pandas/numpy 行转成 JSON 可序列化的 dict。"""
    from infer_strategy_npu.schema import num

    try:
        items = dict(mapping)
    except Exception:  # pragma: no cover
        return {}
    return {str(k): num(v) for k, v in items.items()}


def throughput_of(tokens_per_s: float | None, per_device: float | None) -> float:
    """排序用的吞吐口径：优先集群总吞吐，缺失时退回单卡吞吐。"""
    if tokens_per_s is not None:
        return float(tokens_per_s)
    if per_device is not None:
        return float(per_device)
    return float("-inf")


ProgressFn = Callable[[str, int, int], None]
"""进度回调：``(stage, done, total)``；``stage`` 形如 ``agg`` / ``3P1D/disagg_prefill``。"""


@dataclass
class RunnerContext:
    """Runner 执行上下文。"""

    request: RecommendRequest
    model_spec: ModelSpec
    decision: RoutingDecision
    artifacts: RunArtifacts
    logger: logging.Logger
    on_progress: ProgressFn | None = None
    """可选进度回调；HTTP 服务用它把进度写盘，CLI 下为 None（不影响行为）。"""

    def report_progress(self, stage: str, done: int, total: int) -> None:
        """上报一个子任务的完成进度（total<=0 时不上报）。"""
        if self.on_progress is None or total <= 0:
            return
        try:
            self.on_progress(stage, done, total)
        except Exception:  # pragma: no cover - 进度上报失败不能影响主流程
            self.logger.debug("progress callback failed for stage=%s", stage, exc_info=True)


class FrameworkRunner(abc.ABC):
    """把统一请求下发给某一套框架，并返回归一化的 FrameworkRun。"""

    framework: str = ""

    @abc.abstractmethod
    def run(self, ctx: RunnerContext):
        raise NotImplementedError


@contextlib.contextmanager
def capture_stdout(log_path: Path):
    """把框架的 stdout 打印重定向到日志文件，保持统一 CLI 终端输出干净。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle):
            yield handle


def latest_subdir(root: Path) -> Path | None:
    """返回 root 下最近修改的子目录（aic-npu 结果目录名带随机后缀）。"""
    if not root.exists():
        return None
    subs = [p for p in root.iterdir() if p.is_dir()]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


def tail_text(path: Path, max_lines: int = 20) -> str:
    """读取日志尾部，便于把失败原因写进 run.json。"""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])
