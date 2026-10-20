"""路由：根据模型 + 量化决定下发到 aic-npu 还是 msmodeling。"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from infer_strategy_npu.config import (
    AIC_NPU_BACKEND,
    AIC_NPU_BACKEND_VERSION,
    AIC_NPU_DATABASE_MODE,
    FRAMEWORK_AIC_NPU,
    FRAMEWORK_MSMODELING,
    MODE_AGG,
    MODE_DISAGG_DECODE,
    MODE_DISAGG_PREFILL,
    MS_TP_SIZES_LARGE_MODELS,
    MS_TP_SIZES_LARGE_START,
    MS_TP_SIZES_START,
    PD_BALANCE_MIN_NODES,
    PD_BALANCE_TABLE_MAX,
    get_pd_balance_granularity,
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


_ALLOWED_DEVICE_COUNTS: dict[int, tuple[int, ...]] = {}


def allowed_device_counts(granularity: int) -> tuple[int, ...]:
    """预生成"合理卡数"线性表：``granularity`` 的所有倍数（升序，上限 PD_BALANCE_TABLE_MAX）。

    只按 ``granularity`` 缓存一次；因为「常见 TP（8/16/32/64/128）的并集」恰好等于
    「其中最小值的倍数」，所以这张表就是那次线性筛的结果（8 → 8,16,24,32,...）。
    超过表上限的取值走 :func:`floor_to_allowed` 里的算术兜底。
    """
    cached = _ALLOWED_DEVICE_COUNTS.get(granularity)
    if cached is None:
        cached = tuple(range(granularity, PD_BALANCE_TABLE_MAX + 1, granularity))
        _ALLOWED_DEVICE_COUNTS[granularity] = cached
    return cached


def floor_to_allowed(value: int, granularity: int) -> int:
    """向下取到最近的"合理卡数"（表中 <= value 的最大值）；无可用值返回 0。

    例：granularity=16 时 113 -> 112、56 -> 48、25 -> 16。
    """
    if value <= 0:
        return 0
    table = allowed_device_counts(granularity)
    if value <= table[-1]:
        idx = bisect.bisect_right(table, value) - 1
        return table[idx] if idx >= 0 else 0
    # 超出表上限：直接按算术向下取整，结果与表语义一致
    return value - (value % granularity)


@dataclass(frozen=True)
class PdBalance:
    """一组 xPyD 的 P/D 卡数划分。

    每节点卡数 ``unit`` 先按 :func:`floor_to_allowed` 落到"合理卡数"上，
    则 ``prefill_devices = unit * x``、``decode_devices = unit * y``，
    天然满足 ``P + D <= total``，且两侧都是 ``granularity`` 的倍数、比例严格为 ``x:y``。
    """

    prefill_nodes: int
    decode_nodes: int
    prefill_devices: int
    decode_devices: int

    @property
    def label(self) -> str:
        """目录名/标签，如 ``1P1D``、``3P1D``。"""
        return f"{self.prefill_nodes}P{self.decode_nodes}D"


_PD_LABEL_RE = re.compile(r"^(\d+)\s*[Pp]\s*(\d+)\s*[Dd]$")


def parse_pd_balance_label(value: str) -> tuple[int, int]:
    """解析 ``1P1D`` / ``3P1D`` 这样的 xPyD 写法，返回 ``(x, y)``。

    大小写与空格不敏感（``1p1d``、``1 P 1 D`` 均可）；x、y 必须 >= 1。
    """
    match = _PD_LABEL_RE.match(str(value).strip())
    if not match:
        raise ValueError(f"invalid xPyD spec {value!r}; expected like '1P1D' or '3P1D'")
    x, y = int(match.group(1)), int(match.group(2))
    if x < PD_BALANCE_MIN_NODES or y < PD_BALANCE_MIN_NODES:
        raise ValueError(f"invalid xPyD spec {value!r}: P/D 节点数必须 >= {PD_BALANCE_MIN_NODES}")
    return x, y


def pd_balance_grid(
    total_devices: int,
    max_nodes: int,
    labels: Sequence[str] | None = None,
    granularity: int | None = None,
) -> list[PdBalance]:
    """生成 xPyD 组合列表。

    每节点卡数 ``unit = floor_to_allowed(total // (x + y))``，即先按节点数均分，
    再向下落到"合理卡数"（``granularity`` 的倍数）上。这样即使 ``--num-devices``
    是 1013 这类基数，也能得到能被常见 TP 整除的 P/D 卡数，只是总用量略小于给定值。

    - 未给 ``labels``：枚举 ``1 <= x, y <= max_nodes``，并按 **P:D 的最简比例**去重
      （``1P1D`` / ``2P2D`` / ``3P3D`` 都是 1:1，只保留最先出现的那组），
      卡数取整后撞车的组合也会去掉。
    - 给了 ``labels``（对应 ``--pd-balance-sizes 1P1D 2P1D 3P1D``）：**只搜指定的组合**，
      按用户给定顺序；此时不再按比例去重（不同比例的卡数划分确实不同），但仍会去掉
      卡数完全相同的重复项，并跳过卡数不够切分的组合。
    """
    if labels:
        pairs = [parse_pd_balance_label(v) for v in labels]
        dedupe_ratio = False
    else:
        pairs = [
            (x, y)
            for x in range(PD_BALANCE_MIN_NODES, max_nodes + 1)
            for y in range(PD_BALANCE_MIN_NODES, max_nodes + 1)
        ]
        dedupe_ratio = True

    gran = get_pd_balance_granularity(granularity)
    grid: list[PdBalance] = []
    seen_ratios: set[tuple[int, int]] = set()
    seen_devices: set[tuple[int, int]] = set()
    for x, y in pairs:
        unit = floor_to_allowed(int(total_devices) // (x + y), gran)
        if unit <= 0:
            continue
        combo = PdBalance(x, y, unit * x, unit * y)
        devices = (combo.prefill_devices, combo.decode_devices)
        if devices in seen_devices:
            continue
        if dedupe_ratio:
            gcd = math.gcd(x, y)
            ratio = (x // gcd, y // gcd)
            if ratio in seen_ratios:
                continue
            seen_ratios.add(ratio)
        seen_devices.add(devices)
        grid.append(combo)
    return grid


def resolve_tp_sizes(request: RecommendRequest, device_count: int | None = None) -> list[int]:
    """msmodeling 的 TP 候选（--tp-sizes）。

    显式传入时原样使用；否则按模型 + 可用卡数自动推导：
    - 大模型（MS_TP_SIZES_LARGE_MODELS）从 32 起跳；
    - 其余模型从 1 起跳；
    均为 2 的幂，上限为不超过**当前侧可用卡数**的最大 2 的幂。

    候选还必须是**能整除该侧卡数**的 2 的幂（否则 msmodeling 会因 tp 不整除
    ``--num-devices`` 报错，如 42 卡配 tp=32）。若按起始值筛完为空，则回退到
    "能整除该侧卡数的最大 2 的幂"，保证至少有一个候选（例：48 卡 -> tp=16）。

    小模型起始值仍为 1（`[1, 2, 4, ...]`），大模型从 32 起跳，两者都受整除约束。

    ``device_count`` 为该侧的卡数：PD 混部用总卡数，PD 分离时 P/D 各自用 xPyD
    拆分后分到的卡数，因此 TP 上限会随划分动态变化。
    """
    if request.tp_sizes:
        return [int(v) for v in request.tp_sizes]

    limit = int(device_count if device_count is not None else request.num_devices)
    if limit <= 0:
        return [1]
    start = MS_TP_SIZES_LARGE_START if is_large_tp_model(request.model) else MS_TP_SIZES_START
    # max_divisor = 能整除 limit 的最大 2 的幂；所有更小的 2 的幂自然也能整除
    max_divisor = limit & -limit
    sizes = []
    value = 1
    while value <= max_divisor:
        if value >= start:
            sizes.append(value)
        value *= 2
    if not sizes:
        # 起始值超过可整除上限（如大模型 32 起跳但只有 48 卡）-> 回退到最大可整除值
        sizes = [max_divisor] if max_divisor > 0 else [1]
    return sizes


def is_large_tp_model(model_id: str | None) -> bool:
    """该模型是否属于"TP 需从 32 起跳"的大模型。"""
    return str(model_id or "").strip().lower() in {m.lower() for m in MS_TP_SIZES_LARGE_MODELS}


def tp_size_flags(request: RecommendRequest, device_count: int | None = None) -> list[str]:
    """msmodeling 的 --tp-sizes 参数（始终下发，便于复现）。"""
    sizes = resolve_tp_sizes(request, device_count)
    if not sizes:
        return []
    return ["--tp-sizes", *sizes]


def ms_extra_flags(request: RecommendRequest, mode: str, device_count: int | None = None) -> list[str]:
    """msmodeling 的量化/投机等附加参数（不含 --disagg 与 SLO）。

    ``device_count`` 为该子任务实际可用的卡数（PD 分离时是 xPyD 拆分后某一侧的卡数），
    决定 --tp-sizes 的上限。
    """
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
    extra += tp_size_flags(request, device_count)
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


def build_msmodeling_command(
    request: RecommendRequest,
    model_source: str,
    *,
    mode: str,
    device_count: int | None = None,
) -> str:
    """等价的 msmodeling 命令（用于 dry-run / run.json 复现）。

    ``device_count`` 为该子任务实际下发的卡数：PD 混部是总卡数；PD 分离开启
    --pd-balance-search 后，P/D 各自用 xPyD 拆分到的卡数。
    """
    devices = int(device_count if device_count is not None else request.num_devices)
    parts = [
        "python",
        "-m",
        "cli.inference.throughput_optimizer",
        model_source,
        "--device",
        request.device,
        "--num-devices",
        devices,
        "--input-length",
        request.input_length,
        "--output-length",
        request.output_length,
    ]
    # --compile 是 msmodeling 专属开关（aic-npu 侧无对应参数，故只在这里下发）
    if request.compile_model:
        parts.append("--compile")
    parts += ms_extra_flags(request, mode, devices)
    parts += slo_flags(request, mode)
    return " ".join(_q(p) for p in parts)


def build_msmodeling_commands(request: RecommendRequest, model_source: str) -> list[str]:
    """一次运行要下发的全部 msmodeling 命令（含 xPyD 各组），用于 dry-run 展示。

    - PD 混部(agg)：始终 1 条，用总卡数。
    - PD 分离：开启 --pd-balance-search 时，对 xPyD 网格里每组各下发
      disagg_prefill(P 卡数) + disagg_decode(D 卡数) 两条；关闭时按旧行为
      两侧都用总卡数各 1 条。
    """
    commands = [build_msmodeling_command(request, model_source, mode=MODE_AGG)]
    if request.pd_balance_search:
        for combo in pd_balance_grid(request.num_devices, request.pd_balance_max, request.pd_balance_sizes):
            commands.append(
                build_msmodeling_command(
                    request, model_source, mode=MODE_DISAGG_PREFILL, device_count=combo.prefill_devices
                )
            )
            commands.append(
                build_msmodeling_command(
                    request, model_source, mode=MODE_DISAGG_DECODE, device_count=combo.decode_devices
                )
            )
    else:
        commands.append(build_msmodeling_command(request, model_source, mode=MODE_DISAGG_PREFILL))
        commands.append(build_msmodeling_command(request, model_source, mode=MODE_DISAGG_DECODE))
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
