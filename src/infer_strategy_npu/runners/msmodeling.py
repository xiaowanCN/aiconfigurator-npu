"""msmodeling Runner：复用 throughput_optimizer 的声明式 parser，直接调 ParallelRunner 取结构化结果。

msmodeling 自身没有结果落盘能力，这里负责把 ``OptimizerSummary`` 序列化到
``<run>/msmodeling/<mode>/{summary.csv,candidates.json}``。
"""

from __future__ import annotations

import re
import sys

from infer_strategy_npu.config import (
    FRAMEWORK_MSMODELING,
    MODE_AGG,
    MODE_DISAGG,
    MODE_DISAGG_DECODE,
    MODE_DISAGG_PREFILL,
)
from infer_strategy_npu.quantize import resolve_ms_quantize_actions
from infer_strategy_npu.routing import PdBalance, pd_balance_grid
from infer_strategy_npu.runners.base import (
    FrameworkRunner,
    RunnerContext,
    capture_stdout,
    meets_slo,
    sanitize_raw,
    throughput_of,
)
from infer_strategy_npu.schema import (
    Candidate,
    FrameworkRun,
    ParallelConfig,
    QuantizeSpec,
    ServingModeResult,
    to_float,
    to_int,
)

MS_SUBTASKS = (
    ("agg", MODE_AGG),
    ("disagg_prefill", MODE_DISAGG_PREFILL),
    ("disagg_decode", MODE_DISAGG_DECODE),
)


def _best_variant(variants: list[ServingModeResult]) -> ServingModeResult | None:
    """在多组 xPyD 变体里挑吞吐最优的一组（无可用变体时返回 None）。"""
    usable = [v for v in variants if v.available and v.best is not None]
    if not usable:
        return None
    return max(usable, key=lambda v: throughput_of(v.best.throughput_tokens_per_s, v.best.throughput_tokens_per_s_per_device))

# 负向断言 (?<![A-Za-z-]) 用于避免 "MOE-TP=1" / "MOE-DP=1" 被 TP / DP 误匹配
_PARALLEL_RE = {
    "tp": re.compile(r"(?<![A-Za-z\-])TP\s*=\s*(\d+)", re.IGNORECASE),
    "pp": re.compile(r"(?<![A-Za-z\-])PP\s*=\s*(\d+)", re.IGNORECASE),
    "dp": re.compile(r"(?<![A-Za-z\-])DP\s*=\s*(\d+)", re.IGNORECASE),
    "ep": re.compile(r"(?<![A-Za-z\-])EP\s*=\s*(\d+)", re.IGNORECASE),
    "moe_tp": re.compile(r"MOE[-_ ]?TP\s*=\s*(\d+)", re.IGNORECASE),
    "moe_dp": re.compile(r"MOE[-_ ]?DP\s*=\s*(\d+)", re.IGNORECASE),
}


def _first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def parse_parallel(value: str | None) -> ParallelConfig:
    """从 msmodeling 的 ``TP=32 | PP=1 | DP=4 | EP=128 | MOE-TP=1 | MOE-DP=1`` 串解析并行度。

    ``EP`` 就是 MoE 专家并行，映射到统一字段 ``moe_ep``；``MOE-TP`` / ``MOE-DP``
    分别映射到 ``moe_tp`` / ``moe_dp``。
    """
    if not value:
        return ParallelConfig()
    text = str(value)
    return ParallelConfig(
        tp=_first_int(_PARALLEL_RE["tp"], text),
        pp=_first_int(_PARALLEL_RE["pp"], text),
        dp=_first_int(_PARALLEL_RE["dp"], text),
        moe_tp=_first_int(_PARALLEL_RE["moe_tp"], text),
        moe_ep=_first_int(_PARALLEL_RE["ep"], text),
        moe_dp=_first_int(_PARALLEL_RE["moe_dp"], text),
    )


def _int_or(value: object, fallback: int | None) -> int | None:
    parsed = to_int(value)
    return parsed if parsed is not None else fallback


def parallel_from_ms_row(row) -> ParallelConfig:
    """msmodeling 单行的并行度：优先取数值列（更可靠），缺失时回退解析 ``parallel`` 串。

    summary.csv 同时提供了 ``tp_size/pp_size/dp_size/ep_size/moe_dp_size`` 数值列与
    ``parallel`` 字符串，数值列优先；``moe_tp`` 无独立数值列，只能从串里取。
    """
    parsed = parse_parallel(_row_get(row, "parallel"))
    return ParallelConfig(
        tp=_int_or(_row_get(row, "tp_size"), parsed.tp),
        pp=_int_or(_row_get(row, "pp_size"), parsed.pp),
        dp=_int_or(_row_get(row, "dp_size"), parsed.dp),
        moe_tp=parsed.moe_tp,
        moe_ep=_int_or(_row_get(row, "ep_size"), parsed.moe_ep),
        moe_dp=_int_or(_row_get(row, "moe_dp_size"), parsed.moe_dp),
    )


def _row_get(row, key: str):
    try:
        return row.get(key)
    except (AttributeError, KeyError):
        return None


class MsmodelingRunner(FrameworkRunner):
    framework = FRAMEWORK_MSMODELING

    def run(self, ctx: RunnerContext) -> FrameworkRun:
        run = FrameworkRun(
            framework=self.framework,
            commands=list(ctx.decision.commands),
            artifacts={"root": str(ctx.artifacts.msmodeling_dir)},
        )

        try:
            self._import_msmodeling()
        except Exception as exc:  # pragma: no cover - 依赖缺失
            run.status = "failed"
            run.message = f"failed to import msmodeling (is the msmodeling whl installed?): {exc}"
            return run

        request = ctx.request
        failures: list[str] = []

        # 进度总量：agg 1 个 + 每个 xPyD 组 2 个（prefill/decode）
        total_steps = 1 + 2 * len(self._pd_grid(ctx))
        done_steps = 0

        # ---- PD 混部(agg)：与 xPyD 无关，始终用总卡数 ----
        agg = ServingModeResult(mode=MODE_AGG)
        agg_candidates, agg_best = self._safe_subtask(
            ctx,
            failures,
            scope="agg",
            task_name="agg",
            mode=MODE_AGG,
            device_count=request.num_devices,
            rel_dir="agg",
            log_name="msmodeling_agg",
        )
        if agg_best is not None:
            agg.available = True
            agg.candidates = agg_candidates
            agg.best = agg_best
        run.agg = agg
        done_steps += 1
        ctx.report_progress("agg", done_steps, total_steps)

        # ---- PD 分离(disagg)：按 xPyD 分组，每组单独跑 P/D 两侧 ----
        variants: list[ServingModeResult] = []
        run.artifacts["pd_balance_search"] = bool(request.pd_balance_search)
        if request.pd_balance_search:
            grid = self._pd_grid(ctx)
            run.artifacts["pd_balance_grid"] = [
                {
                    "label": combo.label,
                    "prefill_nodes": combo.prefill_nodes,
                    "decode_nodes": combo.decode_nodes,
                    "prefill_devices": combo.prefill_devices,
                    "decode_devices": combo.decode_devices,
                }
                for combo in grid
            ]
            if not grid:
                failures.append(
                    f"pd-balance-search: num_devices={request.num_devices} 过小, 切分不出任何 xPyD 组合"
                )
            for combo in grid:
                variant = self._run_disagg_variant(ctx, combo, failures)
                done_steps += 2
                ctx.report_progress(f"{combo.label}/disagg", done_steps, total_steps)
                if variant is not None:
                    variants.append(variant)
        else:
            variant = self._run_disagg_variant(ctx, None, failures)
            done_steps += 2
            ctx.report_progress("disagg", done_steps, total_steps)
            if variant is not None:
                variants.append(variant)

        run.disagg_variants = variants
        # disagg 保留"最优变体"，使既有渲染/推荐逻辑无需感知多组 xPyD
        run.disagg = _best_variant(variants)

        if failures:
            run.message = " | ".join(failures)
            if not (agg.available or run.disagg is not None):
                run.status = "failed"
        return run

    # ------------------------------------------------------------------ xPyD

    def _pd_grid(self, ctx: RunnerContext) -> list[PdBalance]:
        """当前请求的 xPyD 组合列表；关闭 --pd-balance-search 时为空（走单一 disagg）。"""
        request = ctx.request
        if not request.pd_balance_search:
            return []
        return pd_balance_grid(request.num_devices, request.pd_balance_max, request.pd_balance_sizes)

    def _run_disagg_variant(
        self,
        ctx: RunnerContext,
        combo: PdBalance | None,
        failures: list[str],
    ) -> ServingModeResult | None:
        """跑一组 xPyD 的 PD 分离（P 侧 / D 侧各一次），返回该变体；两侧都失败则返回 None。"""
        request = ctx.request
        if combo is None:
            # 关闭 --pd-balance-search 时沿用旧行为：P/D 两侧都用总卡数
            label = ""
            prefill_devices = request.num_devices
            decode_devices = request.num_devices
        else:
            label = combo.label
            prefill_devices = combo.prefill_devices
            decode_devices = combo.decode_devices

        prefix = f"{label}/" if label else ""
        scope = f"{label}:" if label else ""

        variant = ServingModeResult(mode=MODE_DISAGG, label=label)
        if combo is not None:
            variant.pd_balance = {
                "prefill_nodes": combo.prefill_nodes,
                "decode_nodes": combo.decode_nodes,
                "prefill_devices": prefill_devices,
                "decode_devices": decode_devices,
            }

        _, prefill_best = self._safe_subtask(
            ctx,
            failures,
            scope=f"{scope}disagg_prefill",
            task_name="disagg_prefill",
            mode=MODE_DISAGG_PREFILL,
            device_count=prefill_devices,
            rel_dir=f"{prefix}disagg_prefill",
            log_name=f"msmodeling_{label}_disagg_prefill" if label else "msmodeling_disagg_prefill",
        )
        _, decode_best = self._safe_subtask(
            ctx,
            failures,
            scope=f"{scope}disagg_decode",
            task_name="disagg_decode",
            mode=MODE_DISAGG_DECODE,
            device_count=decode_devices,
            rel_dir=f"{prefix}disagg_decode",
            log_name=f"msmodeling_{label}_disagg_decode" if label else "msmodeling_disagg_decode",
        )

        if prefill_best is not None:
            variant.prefill = prefill_best
            variant.available = True
        if decode_best is not None:
            variant.decode = decode_best
            variant.available = True

        # disagg 的整体推荐口径取 decode（端到端吞吐）
        if variant.available and variant.best is None:
            variant.best = variant.decode or variant.prefill

        return variant if variant.available else None

    def _safe_subtask(
        self,
        ctx: RunnerContext,
        failures: list[str],
        *,
        scope: str,
        task_name: str,
        mode: str,
        device_count: int,
        rel_dir: str,
        log_name: str,
    ) -> tuple[list[Candidate], Candidate | None]:
        """包一层异常处理，把失败记入 failures 而不是中断整轮搜索。"""
        try:
            return self._run_subtask(ctx, task_name, mode, device_count, rel_dir, log_name)
        except SystemExit as exc:
            failures.append(f"{scope} msmodeling argument parsing exited (code={exc.code})")
        except Exception as exc:
            failures.append(f"{scope} {exc}")
        return [], None

    # ------------------------------------------------------------------ 子任务

    def _run_subtask(
        self,
        ctx: RunnerContext,
        task_name: str,
        mode: str,
        device_count: int,
        rel_dir: str,
        log_name: str,
    ):
        """跑一个 msmodeling 子任务，返回 (candidates, best)。

        ``device_count`` 是本子任务实际可用的卡数（PD 分离时为 xPyD 拆分后某一侧的卡数），
        ``rel_dir`` 是产物在 ``msmodeling/`` 下的相对目录（xPyD 场景带 ``<xPyD>/`` 前缀）。
        """
        from infer_strategy_npu.routing import ms_extra_flags, slo_flags

        request = ctx.request
        argv = [
            ctx.decision.ms_model_source,
            "--device",
            request.device,
            "--num-devices",
            str(device_count),
            "--input-length",
            str(request.input_length),
            "--output-length",
            str(request.output_length),
        ]
        if request.compile_model:
            argv.append("--compile")
        argv += [str(x) for x in ms_extra_flags(request, mode, device_count)]
        argv += [str(x) for x in slo_flags(request, mode)]

        sub_dir = ctx.artifacts.msmodeling_dir / rel_dir
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")

        args = self._parse_args(argv)

        from tensor_cast.core.compilation_config import apply_compilation_config

        apply_compilation_config(args.compilation_config)

        from cli.utils import check_device_targets

        targets = check_device_targets(args, ctx.logger)
        if not targets:
            raise RuntimeError(f"device {request.device!r} rejected by msmodeling device check")
        args.device = request.device  # ParallelRunner 需要单个 profile 名（字符串）

        from serving_cast.parallel_runner import ParallelRunner

        with capture_stdout(ctx.artifacts.log_path(log_name)):
            runner = ParallelRunner(args)
            summaries = runner.run_agg() if mode == MODE_AGG else runner.run_disagg()

        if not summaries:
            return [], None

        summary = summaries[0]
        df = summary.get_summary_df()
        ctx.artifacts.write_dataframe(f"msmodeling/{rel_dir}/summary.csv", df)

        best_row = summary.get_best_result_row()
        if best_row is None:
            if df is None or getattr(df, "empty", True):
                return [], None
            best_row = df.iloc[0]

        candidates = self._candidates_from_df(ctx, df, mode)
        best = self._candidate_from_row(ctx, best_row, mode)
        ctx.artifacts.write_json(
            f"msmodeling/{rel_dir}/candidates.json",
            {
                "best": best.to_dict() if best else None,
                "candidates": [c.to_dict() for c in candidates],
            },
        )
        return candidates, best

    def _parse_args(self, argv: list[str]):
        """用 throughput_optimizer 的声明式 parser 解析参数（含全部归一化与校验）。"""
        from cli.inference.throughput_optimizer import arg_parse

        old_argv = sys.argv
        sys.argv = ["infer-strategy-npu", *argv]
        try:
            return arg_parse()
        finally:
            sys.argv = old_argv

    def _import_msmodeling(self):
        # 注册全部 DeviceProfile（--device 的 choices 与 comm grid 校验依赖它）
        from tensor_cast import device_profiles  # noqa: F401

        return True

    # ------------------------------------------------------------------ 归一化

    def _candidates_from_df(self, ctx: RunnerContext, df, mode: str) -> list[Candidate]:
        if df is None or getattr(df, "empty", True):
            return []
        return [self._candidate_from_row(ctx, df.iloc[i], mode) for i in range(len(df))]

    def _candidate_from_row(self, ctx: RunnerContext, row, mode: str) -> Candidate:
        request = ctx.request
        default_linear, default_non_expert_linear, default_attention = resolve_ms_quantize_actions(
            request.quantize,
            request.quantize_linear_action,
            request.quantize_non_expert_linear_action,
            request.quantize_attention_action,
        )
        serving_mode = MODE_AGG if mode == MODE_AGG else MODE_DISAGG

        ttft = to_float(_row_get(row, "ttft"))
        tpot = to_float(_row_get(row, "tpot"))
        tokens_per_s = to_float(_row_get(row, "token/s"))
        per_device = to_float(_row_get(row, "token/s/device"))
        req_per_s = to_float(_row_get(row, "qps_req_s"))
        if req_per_s is None and tokens_per_s is not None and request.output_length:
            req_per_s = tokens_per_s / float(request.output_length)

        parallel_str = _row_get(row, "parallel")
        return Candidate(
            framework=self.framework,
            serving_mode=serving_mode,
            mode=mode,
            throughput_tokens_per_s=tokens_per_s,
            throughput_tokens_per_s_per_device=per_device,
            throughput_req_per_s=req_per_s,
            ttft_ms=ttft,
            tpot_ms=tpot,
            meets_slo=meets_slo(ttft, tpot, request),
            concurrency=to_int(_row_get(row, "concurrency")),
            batch_size=to_int(_row_get(row, "batch_size")),
            num_devices=to_int(_row_get(row, "num_devices")),
            parallel=parallel_from_ms_row(row),
            parallel_str=parallel_str,
            quantize=QuantizeSpec(
                linear=_row_get(row, "quantize_linear_action") or default_linear,
                non_expert_linear=default_non_expert_linear,
                attention=_row_get(row, "quantize_attention_action") or default_attention,
            ),
            backend=self.framework,
            backend_version=None,
            system=_row_get(row, "device_name") or request.device,
            extra={"model_id": _row_get(row, "model_id")},
            raw=sanitize_raw(row),
        )
