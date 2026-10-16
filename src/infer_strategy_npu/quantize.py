"""--quantize 归一化：统一取值 -> 两套框架的量化动作。"""

from __future__ import annotations

# 统一 CLI 暴露的量化取值
QUANTIZE_DISABLE = "DISABLE"
QUANTIZE_W8A8 = "W8A8"

QUANTIZE_CHOICES: tuple[str, ...] = (QUANTIZE_DISABLE, QUANTIZE_W8A8)

# 输入别名（大小写 / 连字符不敏感）
_ALIASES: dict[str, str] = {
    "DISABLE": QUANTIZE_DISABLE,
    "DISABLED": QUANTIZE_DISABLE,
    "NONE": QUANTIZE_DISABLE,
    "W8A8": QUANTIZE_W8A8,
    "W8A8_DYNAMIC": QUANTIZE_W8A8,
    "W8A8-DYNAMIC": QUANTIZE_W8A8,
}

# msmodeling 侧合法取值（tensor_cast/core/quantization/datatypes.py）
# 文档亦确认：大小写与下划线/连字符写法均可，这里统一用大写下划线形式下发
MS_LINEAR_ACTIONS: tuple[str, ...] = (
    "DISABLED",
    "W8A16_STATIC",
    "W8A8_STATIC",
    "W4A8_STATIC",
    "W8A16_DYNAMIC",
    "W8A8_DYNAMIC",
    "W4A8_DYNAMIC",
    "FP8",
    "MXFP4",
)

MS_ATTENTION_ACTIONS: tuple[str, ...] = ("DISABLED", "INT8", "FP8")

# 统一取值 -> msmodeling 的 quantize action
# 注意：--quantize-linear-action 的默认值就是 W8A8_DYNAMIC，
# 因此 quantize=DISABLE 时必须显式下发 DISABLED，否则会被默认成 W8A8。
MS_ACTIONS: dict[str, tuple[str, str]] = {
    # quantize: (linear action, attention action)
    QUANTIZE_DISABLE: ("DISABLED", "DISABLED"),
    QUANTIZE_W8A8: ("W8A8_DYNAMIC", "INT8"),
}


def normalize_quantize(value: str | None) -> str:
    """归一化 --quantize 取值，非法值抛 ValueError。"""
    if value is None:
        return QUANTIZE_DISABLE
    key = str(value).strip().upper().replace("-", "_")
    if key not in _ALIASES:
        raise ValueError(f"unsupported --quantize value {value!r}; valid choices: {', '.join(QUANTIZE_CHOICES)}")
    return _ALIASES[key]


def to_ms_actions(quantize: str) -> tuple[str, str]:
    """统一 quantize -> msmodeling 的 (linear action, attention action)。"""
    return MS_ACTIONS[quantize]
