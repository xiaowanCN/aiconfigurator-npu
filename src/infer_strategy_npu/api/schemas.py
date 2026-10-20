"""API 层的 pydantic 请求模型与统一响应封装。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from infer_strategy_npu.config import (
    DEFAULT_INPUT_LENGTH,
    DEFAULT_OUTPUT_LENGTH,
    DEFAULT_QUANTIZE,
    DEFAULT_TOP_N,
    DEFAULT_TPOT_LIMIT,
    DEFAULT_TTFT_LIMIT,
    PD_BALANCE_MAX_NODES_DEFAULT,
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
    quantize: str = Field(DEFAULT_QUANTIZE, description="DISABLE | W8A8")
    top_n: int = Field(DEFAULT_TOP_N, gt=0)
    framework: str = Field("auto", description="auto | aic-npu | msmodeling")
    out_dir: str | None = None

    # msmodeling 透传
    quantize_linear_action: str | None = None
    quantize_attention_action: str | None = None
    speculative_method: str | None = None
    num_speculative_tokens: int | None = None
    acceptance_length: float | None = None
    num_draft_layers: int | None = None
    tp_sizes: list[int] | None = None
    jobs: int | None = Field(None, gt=0)
    max_search_combinations: int | None = None
    compile_model: bool = False

    # PD 均衡搜索
    pd_balance_search: bool = True
    pd_balance_max: int = Field(PD_BALANCE_MAX_NODES_DEFAULT, gt=0)
    pd_balance_sizes: list[str] | None = None

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
