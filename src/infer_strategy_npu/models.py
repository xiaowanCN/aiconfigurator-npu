"""统一模型注册表：ModelScope ID -> (aic-npu model-path / msmodeling model-id / 本地目录)。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from infer_strategy_npu.config import (
    aic_npu_model_config_dir,
    get_local_model_root,
    join_path,
)


@dataclass(frozen=True)
class ModelSpec:
    """一个被统一 CLI 支持的模型。"""

    model_id: str
    """ModelScope 模型 ID（统一 CLI 的 --model 取值）。"""

    aliases: tuple[str, ...] = ()
    """额外可接受的大小写别名。"""

    aic_model_path: str | None = None
    """aic-npu 的 --model-path（容器内 model_configs 下的 config.json）。"""

    ms_model_id: str | None = None
    """msmodeling 的 model-id；为空表示不支持 msmodeling。"""

    local_dir: str | None = None
    """--local-model-root 下的本地模型目录名（含 config.json）。"""

    @property
    def supports_aic_npu(self) -> bool:
        return self.aic_model_path is not None

    @property
    def supports_msmodeling(self) -> bool:
        return self.ms_model_id is not None


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1",
        aic_model_path="deepseek-ai--DeepSeek-R1_config.json",
        ms_model_id="deepseek-ai/DeepSeek-R1",
        local_dir="DeepSeek-R1",
    ),
    ModelSpec(
        model_id="Qwen/Qwen3.8-27B",
        aliases=("Qwen/Qwen3.8-27B", "qwen3_8_27B"),
        aic_model_path="qwen3_8_27B_config.json",
        ms_model_id="Qwen/Qwen3.8-27B",
        local_dir="Qwen3.8-27B",
    ),
    ModelSpec(
        model_id="deepseek-ai/DeepSeek-V4-Pro",
        aic_model_path="deepseek-ai--DeepSeek-V4-Pro_config.json",
        ms_model_id="deepseek-ai/DeepSeek-V4-Pro",
        local_dir="DeepSeek-V4-Pro",
    ),
    ModelSpec(
        model_id="deepseek-ai/DeepSeek-V4.1-Flash",
        ms_model_id="deepseek-ai/DeepSeek-V4.1-Flash",
        local_dir="DeepSeek-V4.1-Flash",
    ),
    ModelSpec(
        model_id="ZhipuAI/GLM-5.3-Flash",
        ms_model_id="ZhipuAI/GLM-5.3-Flash",
        local_dir="GLM5.3-Flash",
    ),
    ModelSpec(
        model_id="ZhipuAI/GLM-5.3",
        ms_model_id="ZhipuAI/GLM-5.3",
        local_dir="GLM5.3",
    ),
    ModelSpec(
        model_id="ZhipuAI/GLM-5.2",
        ms_model_id="ZhipuAI/GLM-5.2",
        local_dir="GLM5.2",
    ),
    ModelSpec(
        model_id="moonshotai/Kimi-K3",
        ms_model_id="moonshotai/Kimi-K3",
        local_dir="Kimi-K3",
    ),
    ModelSpec(
        model_id="MiniMax/MiniMax-M3",
        aliases=("MiniMax/MiniMax-M3", "MiniMaxAI/MiniMax-M3"),
        ms_model_id="MiniMax/MiniMax-M3",
        local_dir="MiniMax-M3",
    ),
)

_MODEL_INDEX: dict[str, ModelSpec] = {}
for _spec in MODELS:
    _MODEL_INDEX[_spec.model_id.lower()] = _spec
    for _alias in _spec.aliases:
        _MODEL_INDEX[_alias.lower()] = _spec


def supported_models() -> tuple[str, ...]:
    return tuple(spec.model_id for spec in MODELS)


def resolve_model(name: str) -> ModelSpec:
    """按 ModelScope ID 解析模型，大小写不敏感；未注册时抛 ValueError。"""
    spec = _MODEL_INDEX.get(str(name).strip().lower())
    if spec is None:
        raise ValueError(
            f"unsupported --model {name!r}; supported models: {', '.join(supported_models())}"
        )
    return spec


def aic_npu_model_path(spec: ModelSpec, workspace: str | None = None) -> str:
    """模型在 aic-npu 下的 --model-path（容器内绝对路径）。"""
    if not spec.aic_model_path:
        raise ValueError(f"model {spec.model_id!r} is not supported by aic-npu")
    base = aic_npu_model_config_dir() if workspace is None else aic_npu_model_config_dir(workspace)
    return join_path(base, spec.aic_model_path)


def msmodeling_model_source(spec: ModelSpec, local_model_root: str | None = None) -> str:
    """msmodeling 的本地模型目录（相对路径优先）或 ModelScope ID。

    解析顺序：
    1. ``<local_model_root>/<local_dir>`` 存在 → 返回该路径（默认 ``model_configs/msmodeling/<dir>``，相对路径）
    2. ``--model`` 本身就是本地目录 → 直接返回
    3. 回退到 ModelScope ID（会联网，可能超时）
    """
    if not spec.ms_model_id:
        raise ValueError(f"model {spec.model_id!r} is not supported by msmodeling")
    root = local_model_root or get_local_model_root()
    if spec.local_dir:
        candidate = join_path(root, spec.local_dir)
        if Path(candidate).is_dir():
            return candidate
    raw = Path(spec.model_id)
    if raw.is_dir():
        return spec.model_id
    return spec.ms_model_id
