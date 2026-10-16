"""infer-strategy-npu 命令行入口。

    infer-strategy-npu -recommend --model <ModelScope ID> --device <设备名> --num-devices <N>
        [--input-length 1024] [--output-length 1024]
        [--tpot-limit 50] [--ttft-limit 1000] [--quantize DISABLE|W8A8]
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
)
from infer_strategy_npu.devices import DEVICES
from infer_strategy_npu.models import resolve_model, supported_models
from infer_strategy_npu.output import render_summary
from infer_strategy_npu.quantize import (
    MS_ATTENTION_ACTIONS,
    MS_LINEAR_ACTIONS,
    QUANTIZE_CHOICES,
    normalize_quantize,
)
from infer_strategy_npu.routing import VALID_FRAMEWORKS
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
            "      --device ATLAS_800_A2_376T_64G --num-devices 128 --quantize W8A8\n"
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
        help=f"量化方式（默认 {DEFAULT_QUANTIZE}；可选 {', '.join(QUANTIZE_CHOICES)}）",
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
        help="覆盖 msmodeling --quantize-linear-action（默认由 --quantize 推导）",
    )
    ms.add_argument(
        "--quantize-attention-action",
        choices=list(MS_ATTENTION_ACTIONS),
        default=None,
        help="覆盖 msmodeling --quantize-attention-action（KV cache 量化，默认由 --quantize 推导）",
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
    ms.add_argument("--jobs", type=positive_int, default=None, help="msmodeling 搜索并发度")
    ms.add_argument("--max-search-combinations", type=int, default=None)
    ms.add_argument("--no-compile", action="store_true", help="不向 msmodeling 传 --compile")

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
        quantize_attention_action=args.quantize_attention_action,
        speculative_method=(None if args.speculative_method == "none" else args.speculative_method),
        num_speculative_tokens=args.num_speculative_tokens,
        acceptance_length=args.acceptance_length,
        num_draft_layers=args.num_draft_layers,
        jobs=args.jobs,
        max_search_combinations=args.max_search_combinations,
        compile_model=not args.no_compile,
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
