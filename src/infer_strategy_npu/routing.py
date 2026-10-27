"""路由：根据模型 + 量化决定下发到 aic-npu 还是 msmodeling。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from infer_strategy_npu.config import (
    AIC_NPU_BACKEND,
    AIC_NPU_BACKEND_VERSION,
    AIC_NPU_DATABASE_MODE,
    FRAMEWORK_AIC_NPU,
    FRAMEWORK_MSMODELING,
    MODE_AGG,
    MODE_DISAGG,
    MODE_DISAGG_DECODE,
    MODE_DISAGG_PREFILL,
    MODE_PD_RATIO,
)
from infer_strategy_npu.devices import AIC_NPU_SUPPORTED_DEVICES, to_aic_system
from infer_strategy_npu.models import ModelSpec, aic_npu_model_path, msmodeling_model_source
from infer_strategy_npu.quantize import QUANTIZE_DISABLE, resolve_ms_quantize_actions
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


def resolve_ms_actions(request: RecommendRequest) -> tuple[str, str, str]:
    """实际下发给 msmodeling 的 (linear, non_expert_linear, attention) 动作。

    优先级：显式 --quantize-linear-action / --quantize-non-expert-linear-action /
    --quantize-attention-action > --quantize 映射（DISABLE 强制全 DISABLED）。
    """
    return resolve_ms_quantize_actions(
        request.quantize,
        request.quantize_linear_action,
        request.quantize_non_expert_linear_action,
        request.quantize_attention_action,
    )


@dataclass(frozen=True)
class PdRatioCombo:
    """一组 PD-ratio 组合：P 侧每实例卡数 + D 侧每实例卡数。

    与旧的 ``PdBalance`` 不同，这里的两个数字是**每实例**卡数，集群总卡数由
    ``--num-devices`` 单独给出，实例数配比由 msmodeling 内部反推。
    """

    prefill_devices: int
    decode_devices: int

    @property
    def label(self) -> str:
        """目录名/标签，如 ``P8D8``、``P16D8``。"""
        return f"P{self.prefill_devices}D{self.decode_devices}"

    def to_dict(self) -> dict[str, int]:
        return {"prefill_devices": self.prefill_devices, "decode_devices": self.decode_devices}


def pd_ratio_grid(total_devices: int, sizes: Sequence[int] | None = None) -> list[PdRatioCombo]:
    """生成 PD-ratio 组合列表：``sizes`` 与自身的笛卡尔积。

    默认 ``sizes=(8,)`` → 只有 ``P8D8`` 一组；``sizes=(8, 16, 32)`` 则展开成
    9 组 ``(8,8)(8,16)(8,32)(16,8)...(32,32)``。

    剔除规则（不报错，无可行组合时返回空列表）：
    - ``p + d > total_devices``：装不下 1 个 P 实例 + 1 个 D 实例
    - ``p > total`` 或 ``d > total``
    """
    values: list[int] = []
    for raw in sizes or ():
        value = int(raw)
        if value > 0 and value not in values:
            values.append(value)
    total = int(total_devices)
    grid: list[PdRatioCombo] = []
    for prefill in values:
        if prefill > total:
            continue
        for decode in values:
            if decode > total or prefill + decode > total:
                continue
            grid.append(PdRatioCombo(prefill, decode))
    return grid


def tp_size_flags(request: RecommendRequest, device_count: int | None = None) -> list[str]:
    """msmodeling 的 --tp-sizes 参数。

    **缺省（None / 空列表）时不下发**，由 msmodeling 自行推导（所有能整除该侧
    卡数的 2 的幂）；显式传入时才原样下发。

    保留 ``device_count`` 形参仅为兼容旧调用点，实际不再使用。
    """
    sizes = [int(v) for v in (request.tp_sizes or ())]
    if not sizes:
        return []
    return ["--tp-sizes", *sizes]


def ms_extra_flags(request: RecommendRequest, mode: str, device_count: int | None = None) -> list[str]:
    """msmodeling 的量化/投机等附加参数（不含 PD-ratio 专属参数与 SLO）。

    ``device_count`` 形参保留以兼容旧调用点，实际不再参与 --tp-sizes 推导。
    """
    linear, non_expert_linear, attention = resolve_ms_actions(request)
    extra = [
        "--quantize-linear-action",
        linear,
        "--quantize-non-expert-linear-action",
        non_expert_linear,
        "--quantize-attention-action",
        attention,
    ]
    method = (request.speculative_method or "").lower()
    # PD-ratio 模式下 msmodeling 内部只对 Decode 阶段生效投机，但参数需在顶层下发
    if mode in (MODE_DISAGG_DECODE, MODE_PD_RATIO) and method and method != "none":
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
    extra += tp_size_flags(request, device_count)
    return extra


def slo_flags(request: RecommendRequest, mode: str) -> list[str]:
    """SLO 相关参数。

    - PD 混部(agg)：``--tpot-limit`` 与 ``--ttft-limit`` 都下发，与 aic-npu
      （同时传 --ttft / --tpot）保持一致的语义。
    - PD 分离(pd_ratio)：同样两个都下发，但**不能**带 ``--disagg``
      （msmodeling 的 ``pd_ratio_mutex_disagg`` 校验器会直接报错）；
      msmodeling 内部按阶段各取其一（P 用 ttft、D 用 tpot）。
    - PD 分离-Prefill / Decode（aic-npu 侧复现用，msmodeling 旧路径）：带 ``--disagg``。
    """
    if mode == MODE_DISAGG_PREFILL:
        return ["--disagg", "--ttft-limit", request.ttft_limit]
    if mode == MODE_DISAGG_DECODE:
        return ["--disagg", "--tpot-limit", request.tpot_limit]
    return ["--tpot-limit", request.tpot_limit, "--ttft-limit", request.ttft_limit]


def build_msmodeling_command(
    request: RecommendRequest,
    model_source: str,
    *,
    mode: str,
    combo: PdRatioCombo | None = None,
) -> str:
    """等价的 msmodeling 命令（用于 dry-run / run.json 复现）。

    PD 混部(agg) 与 PD 分离(pd_ratio) 都用**集群总卡数**下发 ``--num-devices``；
    PD 分离额外给出每侧的每实例卡数，一条命令即完成 P/D 两阶段与实例配比。
    """
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
    # --compile 是 msmodeling 专属开关（aic-npu 侧无对应参数，故只在这里下发）
    if request.compile_model:
        parts.append("--compile")
    if mode == MODE_PD_RATIO and combo is not None:
        parts += [
            "--enable-optimize-prefill-decode-ratio",
            "--prefill-devices-per-instance",
            combo.prefill_devices,
            "--decode-devices-per-instance",
            combo.decode_devices,
        ]
    parts += ms_extra_flags(request, mode)
    parts += slo_flags(request, mode)
    return " ".join(_q(p) for p in parts)


def build_msmodeling_commands(request: RecommendRequest, model_source: str) -> list[str]:
    """一次运行要下发的全部 msmodeling 命令，用于 dry-run 展示。

    - PD 混部(agg)：``--serving-modes`` 含 ``agg`` 时 1 条，用集群总卡数。
    - PD 分离(pd_ratio)：``--serving-modes`` 含 ``disagg`` 时，对 PD-ratio 网格里
      每组各 1 条（**不再**是旧的 P/D 两条）。
    """
    commands: list[str] = []
    if request.wants_serving_mode(MODE_AGG):
        commands.append(build_msmodeling_command(request, model_source, mode=MODE_AGG))
    if request.wants_serving_mode(MODE_DISAGG):
        for combo in pd_ratio_grid(request.num_devices, request.pd_ratio_sizes):
            commands.append(build_msmodeling_command(request, model_source, mode=MODE_PD_RATIO, combo=combo))
    return commands


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
        commands=build_msmodeling_commands(request, source),
    )
