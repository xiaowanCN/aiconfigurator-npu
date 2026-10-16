"""全局默认常量、环境变量与路径解析。"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path

SCHEMA_VERSION = "1.0"

# ---- 统一 CLI 默认值 ----
DEFAULT_INPUT_LENGTH = 1024
DEFAULT_OUTPUT_LENGTH = 1024
DEFAULT_TPOT_LIMIT = 50
DEFAULT_TTFT_LIMIT = 1000
DEFAULT_QUANTIZE = "DISABLE"
DEFAULT_TOP_N = 10
DEFAULT_OUT_DIR = "/workspace/results/infer-strategy-npu"
DEFAULT_MODE = "recommend"

# ---- aic-npu 侧固定下发参数 ----
AIC_NPU_WORKSPACE = "/workspace"
AIC_NPU_MODEL_CONFIG_DIR = "model_configs"
AIC_NPU_SYSTEM_DEFAULT = "ascend_910b"
AIC_NPU_BACKEND = "vllm-ascend"
AIC_NPU_BACKEND_VERSION = "0.23.0"
AIC_NPU_DATABASE_MODE = "HYBRID"

# ---- msmodeling 侧默认下发参数 ----
MS_DEFAULT_JOBS = 8
MS_DEFAULT_MAX_SEARCH_COMBINATIONS = 100
MS_DEFAULT_SPECULATIVE_METHOD = "dflash"
MS_DEFAULT_NUM_SPECULATIVE_TOKENS = 7
MS_DEFAULT_ACCEPTANCE_LENGTH = 5
MS_DEFAULT_NUM_DRAFT_LAYERS = 6

# ---- 本地模型目录（避免 ModelScope/HF 拉取超时） ----
# msmodeling 的模型配置统一放在 model_configs/msmodeling/<模型目录>/config.json，
# 默认使用相对路径（相对 CWD；容器内 CWD=/workspace）。
ENV_LOCAL_MODEL_ROOT = "INFER_STRATEGY_LOCAL_MODEL_ROOT"
MS_MODEL_ROOT = "model_configs/msmodeling"

FRAMEWORK_AIC_NPU = "aic-npu"
FRAMEWORK_MSMODELING = "msmodeling"

MODE_AGG = "agg"
MODE_DISAGG = "disagg"
MODE_DISAGG_PREFILL = "disagg_prefill"
MODE_DISAGG_DECODE = "disagg_decode"


def get_local_model_root(explicit: str | None = None) -> str:
    """解析 msmodeling 的本地模型根目录：显式参数 > 环境变量 > 内置相对目录。

    默认返回相对路径 ``model_configs/msmodeling``（相对 CWD），
    保持命令与 run.json 在不同机器上可复现。
    """
    if explicit:
        return explicit
    env = os.environ.get(ENV_LOCAL_MODEL_ROOT)
    if env:
        return env
    return MS_MODEL_ROOT


def _is_posix(path: str) -> bool:
    return path.startswith("/") or ("/" in path and "\\" not in path)


def join_path(base: str, name: str) -> str:
    """按 base 的风格拼接路径（容器内的 POSIX 路径不会被 Windows 风格化）。"""
    if _is_posix(base):
        return posixpath.join(base, name)
    return str(Path(base) / name)


def aic_npu_model_config_dir(workspace: str = AIC_NPU_WORKSPACE) -> str:
    """容器内 aic-npu 的 model_configs 目录。"""
    return join_path(workspace, AIC_NPU_MODEL_CONFIG_DIR)


def slugify(value: str) -> str:
    """把模型名/设备名转成可用于文件名的片段。"""
    safe = "".join(ch if ch.isalnum() or ch in ".-_" else "-" for ch in str(value))
    return safe.strip("-") or "unknown"
