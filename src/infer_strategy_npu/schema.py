"""统一结果数据模型（dataclass <-> dict），CLI 与未来 HTTP 共用。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from infer_strategy_npu.config import (
    DEFAULT_INPUT_LENGTH,
    DEFAULT_OUTPUT_LENGTH,
    DEFAULT_QUANTIZE,
    DEFAULT_TOP_N,
    DEFAULT_TPOT_LIMIT,
    DEFAULT_TTFT_LIMIT,
    PD_BALANCE_MAX_NODES_DEFAULT,
    SCHEMA_VERSION,
)


def num(value: Any) -> Any:
    """把 numpy / pandas 标量转为内置数值；None 与 NaN 统一成 None。"""
    if value is None:
        return None
    item = value
    if hasattr(item, "item"):
        try:
            item = item.item()
        except Exception:  # pragma: no cover - 防御性
            pass
    if isinstance(item, bool):
        return item
    if isinstance(item, (int, float)):
        return None if isinstance(item, float) and math.isnan(item) else item
    return item


def to_float(value: Any) -> float | None:
    result = num(value)
    if result is None or isinstance(result, bool):
        return None
    try:
        return float(result)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    result = num(value)
    if result is None or isinstance(result, bool):
        return None
    try:
        return int(result)
    except (TypeError, ValueError):
        return None


@dataclass
class QuantizeSpec:
    linear: str | None = None
    attention: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"linear": self.linear, "attention": self.attention}


@dataclass
class ParallelConfig:
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    """MoE 专家并行度（aic-npu: ``moe_ep``；msmodeling: ``ep_size`` / ``EP=``）。"""

    moe_dp: int | None = None
    """MoE 的数据并行度（msmodeling: ``moe_dp_size`` / ``MOE-DP=``；aic-npu 无此列）。"""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Candidate:
    """归一化后的单条部署配置候选（两套框架同构）。"""

    framework: str
    serving_mode: str
    mode: str
    throughput_tokens_per_s: float | None = None
    throughput_tokens_per_s_per_device: float | None = None
    throughput_req_per_s: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    meets_slo: bool | None = None
    concurrency: int | None = None
    batch_size: int | None = None
    num_devices: int | None = None
    parallel: ParallelConfig | None = None
    parallel_str: str | None = None
    quantize: QuantizeSpec | None = None
    backend: str | None = None
    backend_version: str | None = None
    system: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["parallel"] = self.parallel.to_dict() if self.parallel else None
        data["quantize"] = self.quantize.to_dict() if self.quantize else None
        return data


@dataclass
class ServingModeResult:
    """agg / disagg 的结果容器；disagg 额外带 prefill / decode 两个子视角。"""

    mode: str
    available: bool = False
    best: Candidate | None = None
    prefill: Candidate | None = None
    decode: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    label: str = ""
    """PD 均衡搜索产生的变体标签（如 ``1P1D``）；非 xPyD 场景为空。"""

    pd_balance: dict[str, Any] | None = None
    """该变体的 xPyD 划分明细（x / y / P卡数 / D卡数）。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "available": self.available,
            "best": self.best.to_dict() if self.best else None,
            "prefill": self.prefill.to_dict() if self.prefill else None,
            "decode": self.decode.to_dict() if self.decode else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "source_files": list(self.source_files),
            "label": self.label,
            "pd_balance": dict(self.pd_balance) if self.pd_balance else None,
        }


@dataclass
class FrameworkRun:
    framework: str
    status: str = "ok"  # ok | failed | skipped
    message: str | None = None
    commands: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    agg: ServingModeResult | None = None
    disagg: ServingModeResult | None = None
    disagg_variants: list[ServingModeResult] = field(default_factory=list)
    """PD 均衡搜索下每组 xPyD 一个变体；``disagg`` 取其中吞吐最优的一组（向后兼容）。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "status": self.status,
            "message": self.message,
            "commands": list(self.commands),
            "artifacts": self.artifacts,
            "agg": self.agg.to_dict() if self.agg else None,
            "disagg": self.disagg.to_dict() if self.disagg else None,
            "disagg_variants": [v.to_dict() for v in self.disagg_variants],
        }


@dataclass
class RecommendRequest:
    """一次推荐请求的归一化入参（CLI 与未来 HTTP 共用）。"""

    model: str
    device: str
    num_devices: int
    input_length: int = DEFAULT_INPUT_LENGTH
    output_length: int = DEFAULT_OUTPUT_LENGTH
    tpot_limit: int = DEFAULT_TPOT_LIMIT
    ttft_limit: int = DEFAULT_TTFT_LIMIT
    quantize: str = DEFAULT_QUANTIZE
    mode: str = "recommend"
    top_n: int = DEFAULT_TOP_N
    framework: str = "auto"  # auto | aic-npu | msmodeling
    local_model_root: str | None = None
    out_dir: str | None = None
    # msmodeling 透传
    quantize_linear_action: str | None = None
    """显式覆盖 msmodeling 的 --quantize-linear-action（为空则按 --quantize 映射）。"""

    quantize_attention_action: str | None = None
    """显式覆盖 msmodeling 的 --quantize-attention-action（为空则按 --quantize 映射）。"""

    speculative_method: str | None = None
    num_speculative_tokens: int | None = None
    acceptance_length: float | None = None
    num_draft_layers: int | None = None
    tp_sizes: list[int] | None = None
    """msmodeling 的 --tp-sizes 候选；为空则按模型与 --num-devices 自动推导。"""

    jobs: int | None = None
    max_search_combinations: int | None = None
    compile_model: bool = False
    """是否向 msmodeling 传 --compile；仅对分发到 msmodeling 的请求生效（aic-npu 不使用）。"""

    pd_balance_search: bool = True
    """PD 分离时是否做 xPyD 卡数均衡搜索；仅对分发到 msmodeling 的请求生效。"""

    pd_balance_max: int = PD_BALANCE_MAX_NODES_DEFAULT
    """xPyD 搜索时 P/D 节点数的上限（x、y 取值 1..pd_balance_max）。"""

    pd_balance_sizes: list[str] | None = None
    """显式指定要搜的 xPyD 组合（如 ``["1P1D", "2P1D", "3P1D"]``）。

    给定时**只搜这些组合**（忽略 ``pd_balance_max``），与 --tp-sizes 的用法一致。
    """
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedRunResult:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    generated_at: str = ""
    status: str = "ok"  # ok | partial | failed
    request: dict[str, Any] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    frameworks: list[FrameworkRun] = field(default_factory=list)
    recommendation: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "status": self.status,
            "request": self.request,
            "routing": self.routing,
            "frameworks": [f.to_dict() for f in self.frameworks],
            "recommendation": self.recommendation,
            "errors": list(self.errors),
            "artifacts": self.artifacts,
        }
