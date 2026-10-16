"""统一设备清单。

设备名取自 msmodeling ``tensor_cast/device.py`` 中注册的 ``DeviceProfile``，
作为两套框架的设备命名唯一真源；aic-npu 只支持其中的一种（映射到 ``--system``）。
"""

from __future__ import annotations

from infer_strategy_npu.config import AIC_NPU_SYSTEM_DEFAULT

# msmodeling tensor_cast/device.py 中注册的全部 DeviceProfile（26 个）
DEVICES: tuple[str, ...] = (
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE_ROCE",
    "ATLAS_350_425T_112G",
    "ATLAS_350_425T_84G",
    "ATLAS_850_486T_112G",
    "ATLAS_850_486T_128G",
    "ATLAS_850E_486T_96G",
    "ATLAS_850E_425T_96G",
    "ATLAS_850_486T_112G_ROCE",
    "ATLAS_850_486T_128G_ROCE",
    "ATLAS_850E_486T_96G_ROCE",
    "ATLAS_850E_425T_96G_ROCE",
    "ATLAS_850_486T_112G_FM16",
    "ATLAS_850_486T_128G_FM16",
    "ATLAS_850E_486T_96G_FM16",
    "ATLAS_850E_425T_96G_FM16",
    "ATLAS_950_486T_128G",
    "ATLAS_950_486T_96G",
    "ATLAS_950_486T_144G",
)

# 统一设备名 -> aic-npu `--system`
DEVICE_TO_AIC_SYSTEM: dict[str, str] = {
    "ATLAS_800_A2_376T_64G": AIC_NPU_SYSTEM_DEFAULT,
}

AIC_NPU_SUPPORTED_DEVICES: tuple[str, ...] = tuple(sorted(DEVICE_TO_AIC_SYSTEM))


def is_valid_device(device: str) -> bool:
    return device in DEVICES


def to_aic_system(device: str) -> str | None:
    """统一设备名映射到 aic-npu 的 ``--system``；不支持时返回 None。"""
    return DEVICE_TO_AIC_SYSTEM.get(device)
