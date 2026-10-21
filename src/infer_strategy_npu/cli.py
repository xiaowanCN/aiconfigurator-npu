"""infer-strategy-npu 命令行入口。

    infer-strategy-npu -recommend --model <ModelScope ID> --device <设备名> --num-devices <N>
        [--input-length 1024] [--output-length 1024]
        [--tpot-limit 50] [--ttft-limit 1000] [--quantize DISABLE|ENABLE]
"""

from __future__ import annotations

import argparse
import logging
import sys

from infer_strategy_npu import __version__
from infer_strategy_npu.config import (
    DEFAULT_INPUT_LENGTH,
    DEFAULT_OUTPUT_LENGTH,
    DEFAULT_OUT_DIR,
    DEFAULT_QUANTIZE,
    DEFAULT_TOP_N,
    DEFAULT_TPOT_LIMIT,
    DEFAULT_TTFT_LIMIT,
    MS_DEFAULT_ACCEPTANCE_LENGTH,
    MS_DEFAULT_NUM_DRAFT_LAYERS,
    MS_DEFAULT_NUM_SPECULATIVE_TOKENS,
    MS_DEFAULT_SPECULATIVE_METHOD,
    PD_BALANCE_MAX_NODES_DEFAULT,
)
from infer_strategy_npu.devices import DEVICES
from infer_strategy_npu.models import resolve_model, supported_models
from infer_strategy_npu.output import render_summary
from infer_strategy_npu.quantize import (
    MS_ATTENTION_ACTIONS,
    MS_LINEAR_ACTIONS,
    MS_NON_EXPERT_LINEAR_ACTIONS,
    QUANTIZE_CHOICES,
    normalize_quantize,
)
from infer_strategy_npu.routing import VALID_FRAMEWORKS, parse_pd_balance_label
from infer_strategy_npu.schema import RecommendRequest
from infer_strategy_npu.service import recommend

SPECULATIVE_CHOICES = ("none", "mtp", "dflash", "dspark")


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be a positive integer (> 0)")
    return parsed


def model_type(value: str) -> str:
    """校验 --model 并返回注册的规范 ID。"""
    try:
        return resolve_model(value).model_id
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def quantize_type(value: str) -> str:
    try:
        return normalize_quantize(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def pd_balance_label(value: str) -> str:
    """校验 xPyD 写法（如 ``1P1D`` / ``3P1D``），返回规范化后的大写形式。"""
    try:
        x, y = parse_pd_balance_label(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return f"{x}P{y}D"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer-strategy-npu",
        description="统一入口：整合 aic-npu 与 msmodeling，给出 NPU 推理部署策略推荐。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  infer-strategy-npu -recommend --model Qwen/Qwen3.8-27B \\\n"
            "      --device ATLAS_800_A2_376T_64G --num-devices 128 \\\n"
            "      --input-length 1024 --output-length 1024 --tpot-limit 50 --ttft-limit 1000 --quantize DISABLE\n"
            "\n"
            "  infer-strategy-npu -recommend --model ZhipuAI/GLM-5.3-Flash \\\n"
            "      --device ATLAS_800_A2_376T_64G --num-devices 128 --quantize ENABLE\n"
            "\n"
            "  infer-strategy-npu -recommend --model ZhipuAI/GLM-5.3 \\\n"
            "      --device ATLAS_800_A2_376T_64G --num-devices 128 --quantize ENABLE \\\n"
            "      --quantize-linear-action FP8 --quantize-attention-action FP8\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"infer-strategy-npu {__version__}")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-recommend",
        "--recommend",
        dest="mode",
        action="store_const",
        const="recommend",
        help="推荐模式：搜索并输出 PD 混部(agg) / PD 分离(disagg) 的最优部署配置",
    )

    parser.add_argument("--model", required=True, type=model_type, help="ModelScope 模型 ID（见 --list-models）")
    parser.add_argument("--device", required=True, choices=list(DEVICES), help="统一设备名（见 --list-devices）")
    parser.add_argument("--num-devices", required=True, type=positive_int, help="NPU 数量（正整数）")
    parser.add_argument("--input-length", type=positive_int, default=DEFAULT_INPUT_LENGTH, help="输入长度（默认 1024）")
    parser.add_argument("--output-length", type=positive_int, default=DEFAULT_OUTPUT_LENGTH, help="输出长度（默认 1024）")
    parser.add_argument("--tpot-limit", type=positive_int, default=DEFAULT_TPOT_LIMIT, help="TPOT 上限(ms)（默认 50）")
    parser.add_argument("--ttft-limit", type=positive_int, default=DEFAULT_TTFT_LIMIT, help="TTFT 上限(ms)（默认 1000）")
    parser.add_argument(
        "--quantize",
        type=quantize_type,
        default=DEFAULT_QUANTIZE,
        help=(
            f"量化总开关（默认 {DEFAULT_QUANTIZE}；可选 {', '.join(QUANTIZE_CHOICES)}）。"
            "DISABLE=完全不量化；ENABLE=开启量化，由三个 --quantize-*-action 决定具体动作"
            "（旧版 W8A8/FP8 仍可作为 ENABLE 的别名）"
        ),
    )

    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"结果输出目录（默认 {DEFAULT_OUT_DIR}）")
    parser.add_argument("--top-n", type=positive_int, default=DEFAULT_TOP_N, help="每类返回的候选数量（默认 10）")
    parser.add_argument(
        "--framework",
        choices=list(VALID_FRAMEWORKS),
        default="auto",
        help="强制指定下发框架（默认 auto，按模型+量化自动路由）",
    )
    parser.add_argument(
        "--local-model-root",
        default=None,
        help="msmodeling 本地模型根目录（含各模型 config.json），默认取环境变量 INFER_STRATEGY_LOCAL_MODEL_ROOT",
    )

    ms = parser.add_argument_group("msmodeling passthrough")
    ms.add_argument(
        "--quantize-linear-action",
        choices=list(MS_LINEAR_ACTIONS),
        default=None,
        help="msmodeling 专家线性层量化动作（默认 W8A8_DYNAMIC；仅 --quantize ENABLE 生效）",
    )
    ms.add_argument(
        "--quantize-non-expert-linear-action",
        choices=list(MS_NON_EXPERT_LINEAR_ACTIONS),
        default=None,
        help=(
            "msmodeling 非专家线性层量化动作（attention 投影 / dense MLP / shared expert 等，"
            "默认 DISABLED；仅 --quantize ENABLE 生效，主要用于 DeepSeek V4 类 MoE）"
        ),
    )
    ms.add_argument(
        "--quantize-attention-action",
        choices=list(MS_ATTENTION_ACTIONS),
        default=None,
        help="msmodeling KV cache 量化动作（默认 DISABLED；仅 --quantize ENABLE 生效）",
    )
    ms.add_argument(
        "--speculative-method",
        choices=list(SPECULATIVE_CHOICES),
        default=MS_DEFAULT_SPECULATIVE_METHOD,
        help=f"PD分离-Decode 的投机解码方法（默认 {MS_DEFAULT_SPECULATIVE_METHOD}）",
    )
    ms.add_argument("--num-speculative-tokens", type=int, default=MS_DEFAULT_NUM_SPECULATIVE_TOKENS)
    ms.add_argument("--acceptance-length", type=float, default=MS_DEFAULT_ACCEPTANCE_LENGTH)
    ms.add_argument("--num-draft-layers", type=int, default=MS_DEFAULT_NUM_DRAFT_LAYERS)
    ms.add_argument(
        "--tp-sizes",
        type=positive_int,
        nargs="+",
        metavar="N",
        default=None,
        help=(
            "msmodeling 搜索的 TP 候选（空格分隔，如 --tp-sizes 32 64）；"
            "默认按模型与 --num-devices 推导为 2 的幂，大模型从 32 起跳，其余从 1 起跳"
        ),
    )
    ms.add_argument("--jobs", type=positive_int, default=None, help="msmodeling 搜索并发度")
    ms.add_argument("--max-search-combinations", type=int, default=None)
    ms.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help=(
            "向 msmodeling 传 --compile（开启 FX 图改写/融合，结果更准但更慢）；"
            "不设置则为 False。仅对分发到 msmodeling 的请求生效，aic-npu 分支忽略该参数"
        ),
    )
    ms.add_argument(
        "--pd-balance-search",
        dest="pd_balance_search",
        action="store_true",
        default=True,
        help=(
            "PD 分离时做 xPyD 卡数均衡搜索（默认开启）：把 --num-devices 按 x:y 切给 P/D 两侧，"
            "而非两侧各用满总卡数；仅对分发到 msmodeling 的请求生效，aic-npu 分支忽略"
        ),
    )
    ms.add_argument(
        "--no-pd-balance-search",
        dest="pd_balance_search",
        action="store_false",
        help="关闭 xPyD 卡数均衡搜索，退回 PD 两侧各用满 --num-devices 的旧行为",
    )
    ms.add_argument(
        "--pd-balance-max",
        type=positive_int,
        default=PD_BALANCE_MAX_NODES_DEFAULT,
        metavar="N",
        help=f"xPyD 搜索时 P/D 节点数上限，x、y 取 1..N（默认 {PD_BALANCE_MAX_NODES_DEFAULT}）",
    )
    ms.add_argument(
        "--pd-balance-sizes",
        type=pd_balance_label,
        nargs="+",
        metavar="xPyD",
        default=None,
        help=(
            "显式指定要搜的 xPyD 组合（空格分隔，如 --pd-balance-sizes 1P1D 2P1D 3P1D）；"
            "给定时只搜这些组合并忽略 --pd-balance-max。卡数按各自 x:y 比例切分，"
            "不够切分的组合会被跳过"
        ),
    )

    out = parser.add_argument_group("output")
    out.add_argument("--format", choices=("table", "json"), default="table", help="终端输出格式（默认 table）")
    out.add_argument("--dry-run", action="store_true", help="只打印将要下发的框架命令，不实际执行")
    out.add_argument("--quiet", action="store_true", help="不在终端输出摘要")
    out.add_argument("--verbose", action="store_true", help="打开 DEBUG 日志")
    return parser


def _handle_list_flags(argv: list[str]) -> bool:
    if "--list-models" in argv:
        print("支持的模型（ModelScope ID）:")
        for name in supported_models():
            print(f"  - {name}")
        return True
    if "--list-devices" in argv:
        print("支持的设备（取自 msmodeling tensor_cast/device.py）:")
        for name in DEVICES:
            print(f"  - {name}")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _handle_list_flags(argv):
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    # --pd-balance-sizes 的可行性：x+y 超过卡数时 unit=floor(total/(x+y))=0，无法切分
    if args.pd_balance_sizes:
        for label in args.pd_balance_sizes:
            x, y = parse_pd_balance_label(label)
            if x + y > args.num_devices:
                parser.error(
                    f"--pd-balance-sizes {label} needs at least {x + y} devices "
                    f"(x+y), but --num-devices={args.num_devices}"
                )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] [%(name)s] %(message)s",
    )

    request = RecommendRequest(
        model=args.model,
        device=args.device,
        num_devices=args.num_devices,
        input_length=args.input_length,
        output_length=args.output_length,
        tpot_limit=args.tpot_limit,
        ttft_limit=args.ttft_limit,
        quantize=args.quantize,
        mode=args.mode,
        top_n=args.top_n,
        framework=args.framework,
        local_model_root=args.local_model_root,
        out_dir=args.out_dir,
        quantize_linear_action=args.quantize_linear_action,
        quantize_non_expert_linear_action=args.quantize_non_expert_linear_action,
        quantize_attention_action=args.quantize_attention_action,
        speculative_method=(None if args.speculative_method == "none" else args.speculative_method),
        num_speculative_tokens=args.num_speculative_tokens,
        acceptance_length=args.acceptance_length,
        num_draft_layers=args.num_draft_layers,
        tp_sizes=args.tp_sizes,
        jobs=args.jobs,
        max_search_combinations=args.max_search_combinations,
        compile_model=args.compile,
        pd_balance_search=args.pd_balance_search,
        pd_balance_max=args.pd_balance_max,
        pd_balance_sizes=args.pd_balance_sizes,
        dry_run=args.dry_run,
    )

    result = recommend(request)

    if not args.quiet:
        if args.format == "json":
            import json

            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))
        else:
            print(render_summary(result))

    return 0 if result.status in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
