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
from infer_strategy_npu.quantize import to_ms_actions
from infer_strategy_npu.runners.base import (
    FrameworkRunner,
    RunnerContext,
    capture_stdout,
    meets_slo,
    sanitize_raw,
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

_PARALLEL_RE = {
    "tp": re.compile(r"TP\s*=\s*(\d+)", re.IGNORECASE),
    "pp": re.compile(r"PP\s*=\s*(\d+)", re.IGNORECASE),
    "dp": re.compile(r"DP\s*=\s*(\d+)", re.IGNORECASE),
}


def _first_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def parse_parallel(value: str | None) -> ParallelConfig:
    """从 msmodeling 的 ``TP=8 | PP=1 | DP=1`` 串解析并行度。"""
    if not value:
        return ParallelConfig()
    text = str(value)
    return ParallelConfig(
        tp=_first_int(_PARALLEL_RE["tp"], text),
        pp=_first_int(_PARALLEL_RE["pp"], text),
        dp=_first_int(_PARALLEL_RE["dp"], text),
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

        agg = ServingModeResult(mode=MODE_AGG)
        disagg = ServingModeResult(mode=MODE_DISAGG)
        failures: list[str] = []

        for task_name, mode in MS_SUBTASKS:
            log_path = ctx.artifacts.log_path(f"msmodeling_{task_name}")
            try:
                candidates, best = self._run_subtask(ctx, task_name, mode, log_path)
            except SystemExit as exc:
                failures.append(f"{task_name}: msmodeling argument parsing exited (code={exc.code})")
                continue
            except Exception as exc:
                failures.append(f"{task_name}: {exc}")
                continue

            if best is None:
                failures.append(f"{task_name}: no configuration satisfied the SLO filters")
                continue

            if mode == MODE_AGG:
                agg.available = True
                agg.candidates = candidates
                agg.best = best
            else:
                disagg.available = True
                if mode == MODE_DISAGG_PREFILL:
                    disagg.prefill = best
                else:
                    disagg.decode = best

        # disagg 的整体推荐口径取 decode（端到端吞吐）
        if disagg.available and disagg.best is None:
            disagg.best = disagg.decode

        run.agg = agg
        run.disagg = disagg

        if failures:
            run.message = " | ".join(failures)
            if not (agg.available or disagg.available):
                run.status = "failed"
        return run

    # ------------------------------------------------------------------ 子任务

    def _run_subtask(self, ctx: RunnerContext, task_name: str, mode: str, log_path):
        """跑一个 msmodeling 子任务，返回 (candidates, best)。"""
        from infer_strategy_npu.routing import ms_extra_flags, slo_flags

        request = ctx.request
        argv = [
            ctx.decision.ms_model_source,
            "--device",
            request.device,
            "--num-devices",
            str(request.num_devices),
            "--input-length",
            str(request.input_length),
            "--output-length",
            str(request.output_length),
        ]
        if request.compile_model:
            argv.append("--compile")
        argv += [str(x) for x in ms_extra_flags(request, mode)]
        argv += [str(x) for x in slo_flags(request, mode)]

        sub_dir = ctx.artifacts.msmodeling_dir / task_name
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

        with capture_stdout(log_path):
            runner = ParallelRunner(args)
            summaries = runner.run_agg() if mode == MODE_AGG else runner.run_disagg()

        if not summaries:
            return [], None

        summary = summaries[0]
        df = summary.get_summary_df()
        ctx.artifacts.write_dataframe(f"msmodeling/{task_name}/summary.csv", df)

        best_row = summary.get_best_result_row()
        if best_row is None:
            if df is None or getattr(df, "empty", True):
                return [], None
            best_row = df.iloc[0]

        candidates = self._candidates_from_df(ctx, df, mode)
        best = self._candidate_from_row(ctx, best_row, mode)
        ctx.artifacts.write_json(
            f"msmodeling/{task_name}/candidates.json",
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
        default_linear, default_attention = to_ms_actions(request.quantize)
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
            parallel=parse_parallel(parallel_str),
            parallel_str=parallel_str,
            quantize=QuantizeSpec(
                linear=_row_get(row, "quantize_linear_action") or default_linear,
                attention=_row_get(row, "quantize_attention_action") or default_attention,
            ),
            backend=self.framework,
            backend_version=None,
            system=_row_get(row, "device_name") or request.device,
            extra={"model_id": _row_get(row, "model_id")},
            raw=sanitize_raw(row),
        )
