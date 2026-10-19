"""路由：根据模型 + 量化决定下发到 aic-npu 还是 msmodeling。"""

from __future__ import annotations

from dataclasses import dataclass, field

from infer_strategy_npu.config import (
    AIC_NPU_BACKEND,
    AIC_NPU_BACKEND_VERSION,
    AIC_NPU_DATABASE_MODE,
    FRAMEWORK_AIC_NPU,
    FRAMEWORK_MSMODELING,
    MS_TP_SIZES_LARGE_MODELS,
    MS_TP_SIZES_LARGE_START,
    MS_TP_SIZES_START,
)
from infer_strategy_npu.devices import AIC_NPU_SUPPORTED_DEVICES, to_aic_system
from infer_strategy_npu.models import ModelSpec, aic_npu_model_path, msmodeling_model_source
from infer_strategy_npu.quantize import QUANTIZE_DISABLE, to_ms_actions
from infer_strategy_npu.schema import RecommendRequest

# 命中这些模型且 --quantize DISABLE 时走 aic-npu
AIC_NPU_MODEL_IDS: frozenset[str] = frozenset(
    {
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen3.8-27B",
        "deepseek-ai/DeepSeek-V4-Pro",
    }
)

VALID_FRAMEWORKS = ("auto", FRAMEWORK_AIC_NPU, FRAMEWORK_MSMODELING)


class RoutingError(ValueError):
    """路由阶段无法找到可用框架。"""


@dataclass
class RoutingDecision:
    framework: str
    reason: str
    aic_system: str | None = None
    aic_model_path: str | None = None
    ms_model_source: str | None = None
    commands: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "reason": self.reason,
            "aic_system": self.aic_system,
            "aic_model_path": self.aic_model_path,
            "ms_model_source": self.ms_model_source,
            "commands": list(self.commands),
        }


def needs_quoting(value: str) -> bool:
    return (not value) or any(ch.isspace() or ch in "'\"" for ch in value)


def _q(value: object) -> str:
    text = str(value)
    return f"'{text}'" if needs_quoting(text) else text


def build_aic_npu_command(request: RecommendRequest, model_path: str, system: str, save_dir: str) -> str:
    """等价的 aic-npu 命令（用于 dry-run / run.json 复现）。"""
    parts = [
        "aic-npu",
        "default",
        "--model-path",
        model_path,
        "--total-gpus",
        request.num_devices,
        "--system",
        system,
        "--backend",
        AIC_NPU_BACKEND,
        "--backend-version",
        AIC_NPU_BACKEND_VERSION,
        "--database-mode",
        AIC_NPU_DATABASE_MODE,
        "--isl",
        request.input_length,
        "--osl",
        request.output_length,
        "--ttft",
        request.ttft_limit,
        "--tpot",
        request.tpot_limit,
        "--top-n",
        request.top_n,
        "--save-dir",
        save_dir,
    ]
    return " ".join(_q(p) for p in parts)


def resolve_ms_actions(request: RecommendRequest) -> tuple[str, str]:
    """实际下发给 msmodeling 的 (linear action, attention action)。

    优先级：显式 --quantize-linear-action / --quantize-attention-action > --quantize 映射。
    """
    default_linear, default_attention = to_ms_actions(request.quantize)
    return (
        request.quantize_linear_action or default_linear,
        request.quantize_attention_action or default_attention,
    )


def _powers_of_two(start: int, limit: int) -> list[int]:
    """从 ``start`` 起按 2 的幂递增，返回所有 <= limit 的值。"""
    value = 1
    while value < start:
        value *= 2
    sizes: list[int] = []
    while value <= limit:
        sizes.append(value)
        value *= 2
    return sizes


def resolve_tp_sizes(request: RecommendRequest) -> list[int]:
    """msmodeling 的 TP 候选（--tp-sizes）。

    显式传入时原样使用；否则按模型 + --num-devices 自动推导：
    - 大模型（MS_TP_SIZES_LARGE_MODELS）从 32 起跳；
    - 其余模型从 1 起跳；
    均为 2 的幂，上限为不超过 --num-devices 的最大 2 的幂。
    """
    if request.tp_sizes:
        return [int(v) for v in request.tp_sizes]

    start = MS_TP_SIZES_LARGE_START if is_large_tp_model(request.model) else MS_TP_SIZES_START
    sizes = _powers_of_two(start, request.num_devices)
    if not sizes:
        # num_devices 小于起始值（例如 32 起跳但只有 8 卡）时退化为从 1 推导，保证有候选
        sizes = _powers_of_two(MS_TP_SIZES_START, request.num_devices) or [1]
    return sizes


def is_large_tp_model(model_id: str | None) -> bool:
    """该模型是否属于"TP 需从 32 起跳"的大模型。"""
    return str(model_id or "").strip().lower() in {m.lower() for m in MS_TP_SIZES_LARGE_MODELS}


def tp_size_flags(request: RecommendRequest) -> list[str]:
    """msmodeling 的 --tp-sizes 参数（始终下发，便于复现）。"""
    sizes = resolve_tp_sizes(request)
    if not sizes:
        return []
    return ["--tp-sizes", *sizes]


def ms_extra_flags(request: RecommendRequest, mode: str) -> list[str]:
    """msmodeling 的量化/投机等附加参数（不含 --disagg 与 SLO）。"""
    linear, attention = resolve_ms_actions(request)
    extra = [
        "--quantize-linear-action",
        linear,
        "--quantize-attention-action",
        attention,
    ]
    method = (request.speculative_method or "").lower()
    if mode == "disagg_decode" and method and method != "none":
        extra += ["--speculative-method", method]
        if request.num_speculative_tokens is not None:
            extra += ["--num-speculative-tokens", request.num_speculative_tokens]
        if request.acceptance_length is not None:
            extra += ["--acceptance-length", request.acceptance_length]
        # 文档约束：--num-draft-layers 仅适用于 dflash / dspark，不可与 mtp 同用
        if method in ("dflash", "dspark") and request.num_draft_layers is not None:
            extra += ["--num-draft-layers", request.num_draft_layers]
    if request.jobs is not None:
        extra += ["--jobs", request.jobs]
    if request.max_search_combinations is not None:
        extra += ["--max-search-combinations", request.max_search_combinations]
    extra += tp_size_flags(request)
    return extra


def slo_flags(request: RecommendRequest, mode: str) -> list[str]:
    """SLO 相关参数。

    - PD 混部(agg)：文档列出的关键参数为 ``--tpot-limit`` 与 ``--ttft-limit``，两个都下发，
      与 aic-npu（同时传 --ttft / --tpot）保持一致的语义。
    - PD 分离-Prefill：``--disagg --ttft-limit``
    - PD 分离-Decode：``--disagg --tpot-limit``
    """
    if mode == "disagg_prefill":
        return ["--disagg", "--ttft-limit", request.ttft_limit]
    if mode == "disagg_decode":
        return ["--disagg", "--tpot-limit", request.tpot_limit]
    return ["--tpot-limit", request.tpot_limit, "--ttft-limit", request.ttft_limit]


def build_msmodeling_command(request: RecommendRequest, model_source: str, *, mode: str) -> str:
    """等价的 msmodeling 命令（用于 dry-run / run.json 复现）。"""
    parts = [
        "python",
        "-m",
        "cli.inference.throughput_optimizer",
        model_source,
        "--device",
        request.device,
        "--num-devices",
        request.num_devices,
        "--input-length",
        request.input_length,
        "--output-length",
        request.output_length,
    ]
    if request.compile_model:
        parts.append("--compile")
    parts += ms_extra_flags(request, mode)
    parts += slo_flags(request, mode)
    return " ".join(_q(p) for p in parts)


def route(request: RecommendRequest, spec: ModelSpec, *, aic_save_dir: str) -> RoutingDecision:
    """决定下发框架；无法路由时抛 RoutingError。"""
    forced = str(request.framework or "auto").lower()
    if forced not in VALID_FRAMEWORKS:
        raise RoutingError(f"unsupported --framework {request.framework!r}; use auto|aic-npu|msmodeling")

    if forced == FRAMEWORK_AIC_NPU:
        return _route_to_aic_npu(request, spec, aic_save_dir, reason="forced by --framework aic-npu")

    if forced == FRAMEWORK_MSMODELING:
        return _route_to_msmodeling(request, spec, reason="forced by --framework msmodeling")

    if spec.model_id in AIC_NPU_MODEL_IDS and request.quantize == QUANTIZE_DISABLE:
        return _route_to_aic_npu(
            request,
            spec,
            aic_save_dir,
            reason=f"model {spec.model_id} is in the aic-npu allowlist and --quantize=DISABLE",
        )

    return _route_to_msmodeling(
        request,
        spec,
        reason=(
            f"model {spec.model_id} with quantize={request.quantize} is served by msmodeling "
            "(aic-npu only covers DeepSeek-R1 / Qwen3.8-27B / DeepSeek-V4-Pro with quantize=DISABLE)"
        ),
    )


def _route_to_aic_npu(request: RecommendRequest, spec: ModelSpec, save_dir: str, *, reason: str) -> RoutingDecision:
    if not spec.supports_aic_npu:
        raise RoutingError(
            f"model {spec.model_id!r} has no aic-npu model config; rerun with --framework msmodeling"
        )
    system = to_aic_system(request.device)
    if system is None:
        raise RoutingError(
            f"device {request.device!r} is not supported by aic-npu "
            f"(only {', '.join(AIC_NPU_SUPPORTED_DEVICES)} maps to a --system); "
            "use --framework msmodeling for other devices"
        )
    model_path = aic_npu_model_path(spec)
    return RoutingDecision(
        framework=FRAMEWORK_AIC_NPU,
        reason=reason,
        aic_system=system,
        aic_model_path=model_path,
        commands=[build_aic_npu_command(request, model_path, system, save_dir)],
    )


def _route_to_msmodeling(request: RecommendRequest, spec: ModelSpec, *, reason: str) -> RoutingDecision:
    if not spec.supports_msmodeling:
        raise RoutingError(f"model {spec.model_id!r} has no msmodeling model id")
    source = msmodeling_model_source(spec, request.local_model_root)
    return RoutingDecision(
        framework=FRAMEWORK_MSMODELING,
        reason=reason,
        ms_model_source=source,
        commands=[build_msmodeling_command(request, source, mode=m) for m in ("agg", "disagg_prefill", "disagg_decode")],
    )
