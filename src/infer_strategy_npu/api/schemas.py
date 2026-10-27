"""API 层的 pydantic 请求模型与统一响应封装。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from infer_strategy_npu.config import (
    DEFAULT_INPUT_LENGTH,
    DEFAULT_OUTPUT_LENGTH,
    DEFAULT_QUANTIZE,
    DEFAULT_TOP_N,
    DEFAULT_TPOT_LIMIT,
    DEFAULT_TTFT_LIMIT,
    PD_RATIO_SIZES_DEFAULT,
    SERVING_MODES,
    SERVING_MODES_DEFAULT,
)
from infer_strategy_npu.devices import DEVICES
from infer_strategy_npu.models import supported_models

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

    def to_request_dict(self) -> dict:
        data = self.model_dump(exclude_none=True)
        data.pop("client_token", None)
        return data


def envelope(code: int, data: Any = None, msg: str | None = None) -> dict:
    """统一响应体：``{"code": ..., "msg": ..., "data": ...}``。"""
    return {"code": code, "msg": msg or MSG.get(code, ""), "data": data}


def error_detail(field: str, reason: str) -> dict:
    return {"field": field, "reason": reason}


def supported_device_names() -> list[str]:
    return list(DEVICES)
