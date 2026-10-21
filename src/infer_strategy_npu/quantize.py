"""量化相关常量与归一化。

``--quantize`` 是高层开关：``DISABLE`` 表示完全不量化（三个 action 全为 ``DISABLED``）；
``ENABLE`` 表示开启量化，具体动作由 ``--quantize-linear-action`` /
``--quantize-non-expert-linear-action`` / ``--quantize-attention-action`` 三个参数控制
（缺省时取 msmodeling 原生命令的默认取值）。
"""

from __future__ import annotations

# 统一 CLI 暴露的量化开关
QUANTIZE_DISABLE = "DISABLE"
QUANTIZE_ENABLE = "ENABLE"

QUANTIZE_CHOICES: tuple[str, ...] = (QUANTIZE_DISABLE, QUANTIZE_ENABLE)

# 输入别名（大小写 / 连字符不敏感）
_ALIASES: dict[str, str] = {
    # -> DISABLE
    "DISABLE": QUANTIZE_DISABLE,
    "DISABLED": QUANTIZE_DISABLE,
    "NONE": QUANTIZE_DISABLE,
    "OFF": QUANTIZE_DISABLE,
    # -> ENABLE（含旧版量化取值，向后兼容；具体动作由三个 action 参数决定）
    "ENABLE": QUANTIZE_ENABLE,
    "ENABLED": QUANTIZE_ENABLE,
    "ON": QUANTIZE_ENABLE,
    "W8A8": QUANTIZE_ENABLE,
    "W8A8_DYNAMIC": QUANTIZE_ENABLE,
    "W8A8-DYNAMIC": QUANTIZE_ENABLE,
    "FP8": QUANTIZE_ENABLE,
    "FP8_E4M3": QUANTIZE_ENABLE,
    "FP8-E4M3": QUANTIZE_ENABLE,
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
# 非专家线性层（attention 投影 / dense MLP / shared expert 等）与专家线性层共用同一取值集合
MS_NON_EXPERT_LINEAR_ACTIONS: tuple[str, ...] = MS_LINEAR_ACTIONS

# ENABLE 时各动作的 msmodeling 默认取值（仅在对应 action 参数未显式传入时生效）
MS_LINEAR_DEFAULT = "W8A8_DYNAMIC"
MS_NON_EXPERT_LINEAR_DEFAULT = "DISABLED"
MS_ATTENTION_DEFAULT = "DISABLED"


def normalize_quantize(value: str | None) -> str:
    """把 CLI 传入的量化开关归一化为 DISABLE / ENABLE；非法值抛 ValueError。"""
    if value is None:
        return QUANTIZE_DISABLE
    key = str(value).strip().upper().replace("-", "_")
    if key not in _ALIASES:
        raise ValueError(
            f"unsupported --quantize value {value!r}; valid choices: {', '.join(QUANTIZE_CHOICES)}"
            "（旧版 W8A8 / FP8 仍可作为 ENABLE 的别名）"
        )
    return _ALIASES[key]


def resolve_ms_quantize_actions(
    quantize: str,
    linear: str | None = None,
    non_expert_linear: str | None = None,
    attention: str | None = None,
) -> tuple[str, str, str]:
    """返回下发给 msmodeling 的 (linear, non_expert_linear, attention) 三个量化动作。

    - ``--quantize DISABLE``：三项强制 ``DISABLED``（忽略显式传入的 action）。
    - ``--quantize ENABLE``：以显式传入的 action 为准；缺省时取 msmodeling 原生默认
      （linear=W8A8_DYNAMIC、non_expert_linear=DISABLED、attention=DISABLED）。
    """
    if normalize_quantize(quantize) == QUANTIZE_DISABLE:
        return ("DISABLED", "DISABLED", "DISABLED")
    return (
        linear or MS_LINEAR_DEFAULT,
        non_expert_linear or MS_NON_EXPERT_LINEAR_DEFAULT,
        attention or MS_ATTENTION_DEFAULT,
    )


def to_ms_actions(quantize: str) -> tuple[str, str]:
    """向后兼容：返回 (linear, attention) 两元组（ENABLE 时取原生默认）。"""
    linear, _, attention = resolve_ms_quantize_actions(quantize)
    return (linear, attention)
