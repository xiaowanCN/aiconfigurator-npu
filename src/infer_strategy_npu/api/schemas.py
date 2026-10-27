"""API 层的 pydantic 请求模型与统一响应封装。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from infer_strategy_npu.config import (
    DEFAULT_INPUT_LENGTH,
    DEFAULT_OUTPUT_LENGTH,
    DEFAULT_QUANTIZE,
    DEFAULT_TOP_N,
    DEFAULT_TPOT_LIMIT,
    DEFAULT_TTFT_LIMIT,
    MODE_DISAGG,
    PD_RATIO_SIZES_DEFAULT,
    SERVING_MODES,
    SERVING_MODES_DEFAULT,
)
from infer_strategy_npu.devices import DEVICES
from infer_strategy_npu.models import supported_models

# 单组 PD-ratio 的实测耗时（32 卡 / 8 进程，见 PD_RATIO_REDESIGN §9.7），仅用于告警文案
PD_RATIO_SECONDS_PER_COMBO = 177
# 组合数达到该值时给出耗时告警
PD_RATIO_MANY_COMBOS = 5

# code 与 HTTP 状态码保持一致
MSG = {
    200: "查询成功",
    202: "任务已创建",
    400: "参数错误",
    404: "任务不存在",
    409: "任务状态冲突",
    429: "队列已满，请稍后重试",
    500: "服务内部错误",
    503: "服务未就绪",
}


class TaskCreateRequest(BaseModel):
    """创建任务的请求体；字段与 ``RecommendRequest`` 一一对应。"""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description=f"ModelScope ID，白名单：{', '.join(supported_models())}")
    device: str = Field(..., description="设备名，取值见 /api/v1/devices")
    num_devices: int = Field(..., gt=0)

    input_length: int = Field(DEFAULT_INPUT_LENGTH, gt=0)
    output_length: int = Field(DEFAULT_OUTPUT_LENGTH, gt=0)
    tpot_limit: int = Field(DEFAULT_TPOT_LIMIT, gt=0)
    ttft_limit: int = Field(DEFAULT_TTFT_LIMIT, gt=0)
    quantize: str = Field(DEFAULT_QUANTIZE, description="DISABLE | ENABLE")
    top_n: int = Field(DEFAULT_TOP_N, gt=0)
    framework: str = Field("auto", description="auto | aic-npu | msmodeling")
    out_dir: str | None = None
    serving_modes: list[str] = Field(
        default=list(SERVING_MODES_DEFAULT),
        description=f"要搜索的部署形态；可选 {' | '.join(SERVING_MODES)}",
    )

    @field_validator("serving_modes")
    @classmethod
    def _validate_serving_modes(cls, value: list[str]) -> list[str]:
        modes: list[str] = []
        for item in value:
            key = str(item).strip().lower()
            if key not in SERVING_MODES:
                raise ValueError(f"unsupported serving mode {item!r}; use {' | '.join(SERVING_MODES)}")
            if key not in modes:
                modes.append(key)
        if not modes:
            raise ValueError("serving_modes must not be empty")
        return modes

    # msmodeling 透传
    quantize_linear_action: str | None = None
    quantize_non_expert_linear_action: str | None = None
    quantize_attention_action: str | None = None
    speculative_method: str | None = None
    num_speculative_tokens: int | None = None
    acceptance_length: float | None = None
    num_draft_layers: int | None = None
    tp_sizes: list[int] | None = None
    jobs: int | None = Field(None, gt=0)
    max_search_combinations: int | None = None
    compile_model: bool = False

    # PD 分离搜索：每实例卡数 P/D 的枚举空间
    pd_ratio_sizes: list[int] = Field(
        default=list(PD_RATIO_SIZES_DEFAULT),
        description="每实例卡数 P/D 的枚举空间，取 P×D 笛卡尔积",
    )

    @field_validator("pd_ratio_sizes")
    @classmethod
    def _validate_pd_ratio_sizes(cls, value: list[int]) -> list[int]:
        sizes: list[int] = []
        for item in value:
            size = int(item)
            if size <= 0:
                raise ValueError(f"pd_ratio_sizes must be positive, got {item!r}")
            if size not in sizes:
                sizes.append(size)
        if not sizes:
            raise ValueError("pd_ratio_sizes must not be empty")
        return sizes

    client_token: str | None = Field(None, description="可选幂等键（当前仅记录，不去重）")

    # 由 model_validator 回填；exclude=True 保证不会传进 RecommendRequest
    pd_ratio_grid: list[dict] = Field(default_factory=list, exclude=True)
    """实际会执行的 PD-ratio 组合（已剔除 P+D > num_devices）。"""

    pd_skipped: list[dict] = Field(default_factory=list, exclude=True)
    """被剔除的 PD-ratio 组合及原因。"""

    @model_validator(mode="after")
    def _resolve_pd_ratio_grid(self) -> "TaskCreateRequest":
        from infer_strategy_npu.routing import pd_ratio_grid

        if MODE_DISAGG not in self.serving_modes:
            self.pd_ratio_grid = []
            self.pd_skipped = []
            return self

        sizes = list(self.pd_ratio_sizes or [])
        combos = pd_ratio_grid(self.num_devices, sizes)
        self.pd_ratio_grid = [
            {
                "label": c.label,
                "prefill_devices": c.prefill_devices,
                "decode_devices": c.decode_devices,
            }
            for c in combos
        ]
        # P+D > num_devices 的组合：domain 层静默剔除，这里补一条原因便于前端展示
        kept = {(c.prefill_devices, c.decode_devices) for c in combos}
        self.pd_skipped = [
            {
                "label": f"P{p}D{d}",
                "prefill_devices": p,
                "decode_devices": d,
                "reason": f"P+D={p + d} > num_devices={self.num_devices}",
            }
            for p in sizes
            for d in sizes
            if (p, d) not in kept
        ]
        return self

    def to_request_dict(self) -> dict:
        data = self.model_dump(exclude_none=True)
        data.pop("client_token", None)
        return data

    def build_warnings(self, framework: str | None = None) -> list[str]:
        """提交前的非阻断告警（不拒绝请求，仅提示）。"""
        warnings: list[str] = []
        sizes = list(self.pd_ratio_sizes or [])

        if MODE_DISAGG in self.serving_modes:
            if not self.pd_ratio_grid:
                smallest = min(sizes) if sizes else 0
                warnings.append(
                    f"PD 分离无可用组合：num_devices={self.num_devices} 装不下任何 P+D "
                    f"（最小 P+D={2 * smallest}）；可增大 num_devices 或减小 pd_ratio_sizes"
                )
            elif len(self.pd_ratio_grid) >= PD_RATIO_MANY_COMBOS:
                minutes = len(self.pd_ratio_grid) * PD_RATIO_SECONDS_PER_COMBO / 60
                warnings.append(
                    f"PD 分离将执行 {len(self.pd_ratio_grid)} 组 PD-ratio 组合，"
                    f"单组实测约 {PD_RATIO_SECONDS_PER_COMBO}s，预计总耗时约 {minutes:.0f} 分钟"
                )
            if self.pd_skipped:
                warnings.append(
                    "已剔除 "
                    + ", ".join(item["label"] for item in self.pd_skipped)
                    + "（P+D 超出 num_devices）"
                )
            if sizes and max(sizes) <= 8 and str(self.quantize).upper() == "DISABLE":
                warnings.append(
                    f"每实例最多 {max(sizes)} 卡且未开启量化，大模型通常放不下；"
                    "如无结果可改 quantize=ENABLE 或增大 pd_ratio_sizes"
                )

        # aic-npu 一次调用必然同时产出两者，serving_modes 只过滤输出
        if framework and framework != "msmodeling":
            if self.serving_modes and len(self.serving_modes) < len(SERVING_MODES):
                warnings.append(
                    "aic-npu 一次调用必然同时产出 agg 与 disagg，serving_modes 仅过滤输出，不减少计算"
                )
        return warnings


def envelope(code: int, data: Any = None, msg: str | None = None) -> dict:
    """统一响应体：``{"code": ..., "msg": ..., "data": ...}``。"""
    return {"code": code, "msg": msg or MSG.get(code, ""), "data": data}


def error_detail(field: str, reason: str) -> dict:
    return {"field": field, "reason": reason}


def supported_device_names() -> list[str]:
    return list(DEVICES)
