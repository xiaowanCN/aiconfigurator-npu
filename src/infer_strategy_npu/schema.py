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
    MODE_DISAGG,
    MODE_PD_RATIO,
    PD_RATIO_SIZES_DEFAULT,
    SCHEMA_VERSION,
    SERVING_MODES_DEFAULT,
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
    non_expert_linear: str | None = None
    attention: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"linear": self.linear, "non_expert_linear": self.non_expert_linear, "attention": self.attention}


@dataclass
class ParallelConfig:
    """跨框架统一的并行度。

    只保留两套框架都能给出的四个维度：``tp`` / ``pp`` / ``dp`` / ``ep``。
    其余 MoE 细分维度（aic-npu 的 ``etp``、msmodeling 的 ``MOE-TP`` / ``MOE-DP`` /
    ``DCP``）与 PP 调度细节不属于统一字段，一律落 :attr:`Candidate.extra`。
    """

    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    ep: int | None = None
    """MoE 专家并行度（aic-npu: ``ep``；msmodeling: ``ep_size`` / ``EP=``）。"""

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
class PdRatioSide:
    """PD 分离中 P 侧或 D 侧的部署视角（单实例口径 + 集群口径）。"""

    instances: int | None = None
    """该侧的实例数量。"""

    devices_per_instance: int | None = None
    """每个实例占用的卡数（即 ``--prefill/decode-devices-per-instance``）。"""

    devices_total: int | None = None
    """该侧总卡数 = ``instances * devices_per_instance``。"""

    parallel: ParallelConfig | None = None
    parallel_str: str | None = None
    batch_size: int | None = None
    concurrency: int | None = None
    qps_per_instance: float | None = None
    """单实例的 req/s（msmodeling: ``p_qps`` / ``d_qps``）。"""

    qps_allocated: float | None = None
    """该侧全部实例合计的 req/s（msmodeling: ``allocated_p_qps`` / ``allocated_d_qps``）。"""

    ttft_ms: float | None = None
    """仅 P 侧有意义。"""

    tpot_ms: float | None = None
    """仅 D 侧有意义。"""

    tokens_per_s_per_instance: float | None = None
    tokens_per_s_per_device: float | None = None
    tokens_per_s_cluster: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instances": self.instances,
            "devices_per_instance": self.devices_per_instance,
            "devices_total": self.devices_total,
            "parallel": self.parallel.to_dict() if self.parallel else None,
            "parallel_str": self.parallel_str,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "qps_per_instance": self.qps_per_instance,
            "qps_allocated": self.qps_allocated,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "tokens_per_s_per_instance": self.tokens_per_s_per_instance,
            "tokens_per_s_per_device": self.tokens_per_s_per_device,
            "tokens_per_s_cluster": self.tokens_per_s_cluster,
        }


@dataclass
class PdRatioCandidate:
    """一条 PD 分离（PD-ratio）部署候选。

    ``decode_tps_per_instance`` 是排序键，与 msmodeling ``pd_decode_tps`` 图纵轴
    同口径（``concurrency_d / tpot_d * 1000``）；集群级三个指标则是派生量，
    用于与 aic-npu 的 ``tokens/s`` / ``tokens/s/gpu`` 直接比较。
    """

    framework: str
    serving_mode: str = MODE_DISAGG
    mode: str = MODE_PD_RATIO
    label: str = ""
    """PD 组合标签，如 ``P8D8``（P 侧每实例 8 卡 / D 侧每实例 8 卡）。"""

    pd_ratio: float | None = None
    prefill: PdRatioSide | None = None
    decode: PdRatioSide | None = None

    throughput_tokens_per_s: float | None = None
    """集群端到端 token 吞吐 = ``throughput_req_per_s * output_length``。"""

    throughput_tokens_per_s_per_device: float | None = None
    tokens_per_s_per_user: float | None = None
    throughput_req_per_s: float | None = None
    request_latency_ms: float | None = None
    devices_used: int | None = None
    num_devices: int | None = None
    meets_slo: bool | None = None

    decode_tps_per_instance: float | None = None
    """排序键：单 D 实例的 decode 吞吐（token/s）。"""

    decode_tps_cluster: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None

    quantize: QuantizeSpec | None = None
    system: str | None = None
    backend: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "serving_mode": self.serving_mode,
            "mode": self.mode,
            "label": self.label,
            "pd_ratio": self.pd_ratio,
            "prefill": self.prefill.to_dict() if self.prefill else None,
            "decode": self.decode.to_dict() if self.decode else None,
            "throughput_tokens_per_s": self.throughput_tokens_per_s,
            "throughput_tokens_per_s_per_device": self.throughput_tokens_per_s_per_device,
            "tokens_per_s_per_user": self.tokens_per_s_per_user,
            "throughput_req_per_s": self.throughput_req_per_s,
            "request_latency_ms": self.request_latency_ms,
            "devices_used": self.devices_used,
            "num_devices": self.num_devices,
            "meets_slo": self.meets_slo,
            "decode_tps_per_instance": self.decode_tps_per_instance,
            "decode_tps_cluster": self.decode_tps_cluster,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "quantize": self.quantize.to_dict() if self.quantize else None,
            "system": self.system,
            "backend": self.backend,
            "extra": dict(self.extra),
            "raw": dict(self.raw),
        }


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
    """PD 分离（PD-ratio）组合标签，如 ``P8D8``；非 PD-ratio 场景为空。"""

    pd_ratio: dict[str, Any] | None = None
    """该组合的每实例卡数划分明细（prefill/decode devices_per_instance）。"""

    pd_ratio_candidates: list[dict[str, Any]] = field(default_factory=list)
    """该组合下的全部 PD 候选（供 UI 展开）。"""

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
            "pd_ratio": dict(self.pd_ratio) if self.pd_ratio else None,
            "pd_ratio_candidates": list(self.pd_ratio_candidates),
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
    """msmodeling 下每组 PD-ratio 组合一个变体；``disagg`` 取其中吞吐最优的一组。"""

    pd_ratio_candidates: list[dict[str, Any]] = field(default_factory=list)
    """跨组合合并、按单 D 实例 decode 吞吐降序的全部 PD 候选（msmodeling 专有）。"""

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
            "pd_ratio_candidates": list(self.pd_ratio_candidates),
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

    quantize_non_expert_linear_action: str | None = None
    """显式覆盖 msmodeling 的 --quantize-non-expert-linear-action（MoE 非专家线性层；为空则按 --quantize 映射）。"""

    quantize_attention_action: str | None = None
    """显式覆盖 msmodeling 的 --quantize-attention-action（KV cache 量化；为空则按 --quantize 映射）。"""

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

    serving_modes: list[str] = field(default_factory=lambda: list(SERVING_MODES_DEFAULT))
    """要搜索的部署形态：``agg``（PD 混部）/ ``disagg``（PD 分离），默认两者都跑。

    对 msmodeling 生效（真正跳过对应子任务）；aic-npu 一次调用必然同时产出两者，
    这里只过滤输出。
    """

    pd_ratio_sizes: list[int] = field(default_factory=lambda: list(PD_RATIO_SIZES_DEFAULT))
    """PD 分离搜索时每实例卡数 P/D 的枚举空间（默认 ``[8]``）。

    取 P×D 笛卡尔积，如 ``[8, 16, 32]`` 展开为 9 个组合；
    ``P + D > num_devices`` 的组合会被剔除，全部不可行时 PD 分离无结果（不报错）。
    """
    dry_run: bool = False

    def wants_serving_mode(self, mode: str) -> bool:
        """该形态是否在搜索范围内（``--serving-modes``）。"""
        return str(mode).lower() in {str(m).lower() for m in self.serving_modes}

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
