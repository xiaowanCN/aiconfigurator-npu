"""msmodeling Runner：复用 throughput_optimizer 的声明式 parser，直接调 ParallelRunner 取结构化结果。

msmodeling 自身没有结果落盘能力，这里负责把 ``OptimizerSummary`` 序列化到
``<run>/msmodeling/agg/`` 与 ``<run>/msmodeling/pd_ratio/<PxDy>/``。

PD 分离走 msmodeling 原生 PD-ratio 模式（``--enable-optimize-prefill-decode-ratio``），
每个 ``(每实例 P 卡数, 每实例 D 卡数)`` 组合一次调用，内部并行完成 P/D 两阶段与
实例配比；旧的"自研 xPyD 切分 + P/D 两次 --disagg"逻辑已整体移除。
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
    MODE_PD_RATIO,
)
from infer_strategy_npu.pd_ratio_metrics import (
    cluster_tokens_per_s,
    decode_tps_per_instance,
    prefill_tps_per_instance,
    request_latency_ms,
    tokens_per_s_per_device,
    tokens_per_s_per_user,
)
from infer_strategy_npu.quantize import resolve_ms_quantize_actions
from infer_strategy_npu.routing import PdRatioCombo, pd_ratio_grid
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
    PdRatioCandidate,
    PdRatioSide,
    QuantizeSpec,
    ServingModeResult,
    to_float,
    to_int,
)

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

    只取统一字段 ``tp`` / ``pp`` / ``dp`` / ``ep``；``MOE-TP`` / ``MOE-DP`` 属于
    非统一字段，另行落到 :func:`parse_moe_extra`。
    """
    if not value:
        return ParallelConfig()
    text = str(value)
    return ParallelConfig(
        tp=_first_int(_PARALLEL_RE["tp"], text),
        pp=_first_int(_PARALLEL_RE["pp"], text),
        dp=_first_int(_PARALLEL_RE["dp"], text),
        ep=_first_int(_PARALLEL_RE["ep"], text),
    )


def parse_moe_extra(value: str | None) -> dict[str, int | None]:
    """并行串里的非统一 MoE 维度（``MOE-TP`` / ``MOE-DP``）。"""
    if not value:
        return {}
    text = str(value)
    return {
        "moe_tp": _first_int(_PARALLEL_RE["moe_tp"], text),
        "moe_dp": _first_int(_PARALLEL_RE["moe_dp"], text),
    }


def _int_or(value: object, fallback: int | None) -> int | None:
    parsed = to_int(value)
    return parsed if parsed is not None else fallback


def _row_get(row, key: str):
    try:
        return row.get(key)
    except (AttributeError, KeyError):
        return None


def parallel_from_ms_row(row) -> ParallelConfig:
    """msmodeling 单行的并行度：优先取数值列（更可靠），缺失时回退解析 ``parallel`` 串。"""
    return _sided_parallel(row, "")


def _sided_parallel(row, suffix: str) -> ParallelConfig:
    """PD-ratio 行里某一侧的并行度（``suffix`` 为 ``_p`` / ``_d``）。

    PD 结果表把并行拆成了 ``tp_size_p`` / ``tp_size_d`` 等数值列，
    ``parallel_p`` / ``parallel_d`` 字符串列仅作兜底。
    """
    parsed = parse_parallel(_row_get(row, f"parallel{suffix}"))
    return ParallelConfig(
        tp=_int_or(_row_get(row, f"tp_size{suffix}"), parsed.tp),
        pp=_int_or(_row_get(row, f"pp_size{suffix}"), parsed.pp),
        dp=_int_or(_row_get(row, f"dp_size{suffix}"), parsed.dp),
        ep=_int_or(_row_get(row, f"ep_size{suffix}"), parsed.ep),
    )


def _sided_moe_extra(row, suffix: str) -> dict[str, int | None]:
    parsed = parse_moe_extra(_row_get(row, f"parallel{suffix}"))
    if suffix:
        return parsed
    moe_dp = _int_or(_row_get(row, "moe_dp_size"), parsed.get("moe_dp"))
    return {"moe_tp": parsed.get("moe_tp"), "moe_dp": moe_dp}


def _prepared_pd_ratio_df(summary):
    """PD-ratio 的"全部候选行"。

    优先复用 msmodeling 内部流水线 ``_prepare_pd_ratio_results()``（它产出的行就是
    终端「Top N PD Ratio Configurations」表的每一行，是最全的候选集）；取不到时
    回退到原始 PD 交叉表（缺失实例数相关列，对应字段留空）。
    """
    prepare = getattr(summary, "_prepare_pd_ratio_results", None)
    if callable(prepare):
        try:
            df = prepare()
            if df is not None and not getattr(df, "empty", True):
                return df
        except Exception:  # pragma: no cover - 上游内部结构变化时的兜底
            pass
    return summary.get_summary_df()


def _best_variant(variants: list[ServingModeResult]) -> ServingModeResult | None:
    """在多组 PD-ratio 变体里挑集群吞吐最优的一组（无可用变体时返回 None）。"""
    usable = [v for v in variants if v.available and v.best is not None]
    if not usable:
        return None
    return max(usable, key=lambda v: throughput_of(v.best.throughput_tokens_per_s, v.best.throughput_tokens_per_s_per_device))


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

        want_agg = request.wants_serving_mode(MODE_AGG)
        want_disagg = request.wants_serving_mode(MODE_DISAGG)
        sizes = list(request.pd_ratio_sizes or ())
        grid = pd_ratio_grid(request.num_devices, sizes) if want_disagg else []

        total_steps = (1 if want_agg else 0) + len(grid)
        done_steps = 0

        run.artifacts["serving_modes"] = list(request.serving_modes)
        run.artifacts["pd_ratio_sizes"] = sizes
        run.artifacts["pd_ratio_grid"] = [{**c.to_dict(), "label": c.label} for c in grid]

        # ---- PD 混部(agg)：与 PD-ratio 无关，始终用集群总卡数 ----
        if want_agg:
            agg = ServingModeResult(mode=MODE_AGG)
            agg_candidates, agg_best = self._safe_subtask(
                ctx,
                failures,
                scope="agg",
                mode=MODE_AGG,
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
        else:
            run.agg = None

        # ---- PD 分离(disagg)：每组 PD-ratio 组合一次调用 ----
        if want_disagg and not grid:
            smallest = min(sizes) if sizes else 0
            failures.append(
                f"pd-ratio: --num-devices={request.num_devices} 装不下任何 PD 组合"
                f"（最小 P+D={2 * smallest}）；可增大 --num-devices 或减小 --pd-ratio-sizes"
            )

        variants: list[ServingModeResult] = []
        all_candidates: list[PdRatioCandidate] = []
        for combo in grid:
            candidates, best = self._run_pd_ratio_combo(ctx, combo, failures)
            done_steps += 1
            ctx.report_progress(f"{combo.label}/pd_ratio", done_steps, total_steps)
            if not candidates:
                continue
            variant = ServingModeResult(mode=MODE_DISAGG, label=combo.label)
            variant.pd_ratio = combo.to_dict()
            variant.pd_ratio_candidates = [c.to_dict() for c in candidates]
            variant.available = True
            variant.prefill = self._side_candidate(ctx, best, "prefill")
            variant.decode = self._side_candidate(ctx, best, "decode")
            variant.best = self._overall_candidate(ctx, best)
            variants.append(variant)
            all_candidates.extend(candidates)

        # 全部候选按"单 D 实例 decode 吞吐"降序（与 msmodeling pd_decode_tps 图纵轴同口径）
        all_candidates.sort(key=lambda c: c.decode_tps_per_instance if c.decode_tps_per_instance is not None else float("-inf"), reverse=True)

        run.disagg_variants = variants
        run.disagg = _best_variant(variants)
        run.pd_ratio_candidates = [c.to_dict() for c in all_candidates]
        self._write_all_candidates(ctx, all_candidates)

        if failures:
            run.message = " | ".join(failures)
            if not ((run.agg and run.agg.available) or run.disagg is not None):
                run.status = "failed"
        return run

    # ------------------------------------------------------------------ PD-ratio

    def _run_pd_ratio_combo(
        self,
        ctx: RunnerContext,
        combo: PdRatioCombo,
        failures: list[str],
    ) -> tuple[list[PdRatioCandidate], PdRatioCandidate | None]:
        """跑一组 PD-ratio 组合（一次调用完成 P/D 两阶段与实例配比）。"""
        request = ctx.request
        rel_dir = f"pd_ratio/{combo.label}"
        log_name = f"msmodeling_{combo.label}_pd_ratio"
        try:
            summary = self._run_subtask(
                ctx,
                mode=MODE_PD_RATIO,
                rel_dir=rel_dir,
                log_name=log_name,
                combo=combo,
            )
        except SystemExit as exc:
            failures.append(f"{combo.label} msmodeling argument parsing exited (code={exc.code})")
            return [], None
        except Exception as exc:
            failures.append(f"{combo.label} {exc}")
            return [], None

        if summary is None:
            failures.append(
                f"{combo.label} msmodeling 未产出任何 PD 配置"
                f"（P/D 各 {combo.prefill_devices}/{combo.decode_devices} 卡/实例下无可行并行组合，通常是显存放不下）；"
                f"可增大 --pd-ratio-sizes（如 8 16）或开启量化 --quantize ENABLE 降低显存需求"
            )
            return [], None

        raw_df = summary.get_summary_df()
        ctx.artifacts.write_dataframe(f"msmodeling/{rel_dir}/pd_ratio_raw.csv", raw_df)
        candidates_df = _prepared_pd_ratio_df(summary)
        ctx.artifacts.write_dataframe(f"msmodeling/{rel_dir}/pd_ratio_candidates.csv", candidates_df)
        if candidates_df is None or getattr(candidates_df, "empty", True):
            failures.append(f"{combo.label} no PD configuration satisfied TTFT={request.ttft_limit}/TPOT={request.tpot_limit}")
            return [], None

        candidates = [self._pd_candidate_from_row(ctx, row, combo) for _, row in candidates_df.iterrows()]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return [], None
        best = candidates[0]
        ctx.artifacts.write_json(f"msmodeling/{rel_dir}/pd_ratio_best.json", best.to_dict())
        return candidates, best

    def _pd_candidate_from_row(self, ctx: RunnerContext, row, combo: PdRatioCombo) -> PdRatioCandidate | None:
        """PD 结果表里的一行 -> 统一 ``PdRatioCandidate``（含全部派生吞吐）。"""
        request = ctx.request

        p_instances = to_int(_row_get(row, "p_instances"))
        d_instances = to_int(_row_get(row, "d_instances"))
        p_devices = to_int(_row_get(row, "num_devices_p")) or combo.prefill_devices
        d_devices = to_int(_row_get(row, "num_devices_d")) or combo.decode_devices

        p_qps = to_float(_row_get(row, "p_qps"))
        d_qps = to_float(_row_get(row, "d_qps"))
        p_qps_alloc = to_float(_row_get(row, "allocated_p_qps"))
        d_qps_alloc = to_float(_row_get(row, "allocated_d_qps"))
        balanced_qps = to_float(_row_get(row, "balanced_qps"))
        ttft = to_float(_row_get(row, "ttft_p"))
        tpot = to_float(_row_get(row, "tpot_d"))
        p_batch = to_int(_row_get(row, "batch_size_p"))
        d_batch = to_int(_row_get(row, "batch_size_d"))
        p_concurrency = to_int(_row_get(row, "concurrency_p"))
        d_concurrency = to_int(_row_get(row, "concurrency_d"))
        pd_ratio = to_float(_row_get(row, "pd_ratio"))

        devices_used = to_int(_row_get(row, "allocated_devices"))
        if devices_used is None and p_instances and d_instances:
            devices_used = p_instances * p_devices + d_instances * d_devices

        # ---- 派生吞吐：与 msmodeling pd_decode_tps 图纵轴同口径 ----
        decode_tps_inst = decode_tps_per_instance(d_concurrency, tpot)
        decode_tps_dev = tokens_per_s_per_device(decode_tps_inst, d_devices)
        decode_tps_cluster = (
            decode_tps_inst * d_instances if decode_tps_inst is not None and d_instances else None
        )
        prefill_tps_inst = prefill_tps_per_instance(p_qps, request.input_length)
        prefill_tps_dev = tokens_per_s_per_device(prefill_tps_inst, p_devices)
        prefill_tps_cluster = (
            prefill_tps_inst * p_instances if prefill_tps_inst is not None and p_instances else None
        )
        cluster_tps = cluster_tokens_per_s(balanced_qps, request.output_length)
        latency = request_latency_ms(ttft, tpot, request.output_length)

        p_parallel = _sided_parallel(row, "_p")
        d_parallel = _sided_parallel(row, "_d")
        linear, non_expert_linear, attention = resolve_ms_quantize_actions(
            request.quantize,
            request.quantize_linear_action,
            request.quantize_non_expert_linear_action,
            request.quantize_attention_action,
        )

        return PdRatioCandidate(
            framework=self.framework,
            label=combo.label,
            pd_ratio=pd_ratio,
            prefill=PdRatioSide(
                instances=p_instances,
                devices_per_instance=p_devices,
                devices_total=p_instances * p_devices if p_instances else None,
                parallel=p_parallel,
                parallel_str=_row_get(row, "parallel_p"),
                batch_size=p_batch,
                concurrency=p_concurrency,
                qps_per_instance=p_qps,
                qps_allocated=p_qps_alloc,
                ttft_ms=ttft,
                tokens_per_s_per_instance=prefill_tps_inst,
                tokens_per_s_per_device=prefill_tps_dev,
                tokens_per_s_cluster=prefill_tps_cluster,
            ),
            decode=PdRatioSide(
                instances=d_instances,
                devices_per_instance=d_devices,
                devices_total=d_instances * d_devices if d_instances else None,
                parallel=d_parallel,
                parallel_str=_row_get(row, "parallel_d"),
                batch_size=d_batch,
                concurrency=d_concurrency,
                qps_per_instance=d_qps,
                qps_allocated=d_qps_alloc,
                tpot_ms=tpot,
                tokens_per_s_per_instance=decode_tps_inst,
                tokens_per_s_per_device=decode_tps_dev,
                tokens_per_s_cluster=decode_tps_cluster,
            ),
            throughput_tokens_per_s=cluster_tps,
            throughput_tokens_per_s_per_device=tokens_per_s_per_device(cluster_tps, devices_used),
            tokens_per_s_per_user=tokens_per_s_per_user(request.output_length, latency),
            throughput_req_per_s=balanced_qps,
            request_latency_ms=latency,
            devices_used=devices_used,
            num_devices=request.num_devices,
            meets_slo=meets_slo(ttft, tpot, request),
            decode_tps_per_instance=decode_tps_inst,
            decode_tps_cluster=decode_tps_cluster,
            ttft_ms=ttft,
            tpot_ms=tpot,
            quantize=QuantizeSpec(linear=linear, non_expert_linear=non_expert_linear, attention=attention),
            system=request.device,
            backend=self.framework,
            extra={
                "prefill_devices_per_instance": combo.prefill_devices,
                "decode_devices_per_instance": combo.decode_devices,
                "prefill_moe": _sided_moe_extra(row, "_p"),
                "decode_moe": _sided_moe_extra(row, "_d"),
            },
            raw=sanitize_raw(row),
        )

    def _overall_candidate(self, ctx: RunnerContext, pdc: PdRatioCandidate) -> Candidate:
        """PD-ratio 变体的"整体"视角（集群口径），供推荐与渲染比较。"""
        request = ctx.request
        linear, non_expert_linear, attention = resolve_ms_quantize_actions(
            request.quantize,
            request.quantize_linear_action,
            request.quantize_non_expert_linear_action,
            request.quantize_attention_action,
        )
        p_str = pdc.prefill.parallel_str if pdc.prefill else None
        d_str = pdc.decode.parallel_str if pdc.decode else None
        return Candidate(
            framework=self.framework,
            serving_mode=MODE_DISAGG,
            mode=MODE_DISAGG,
            throughput_tokens_per_s=pdc.throughput_tokens_per_s,
            throughput_tokens_per_s_per_device=pdc.throughput_tokens_per_s_per_device,
            throughput_req_per_s=pdc.throughput_req_per_s,
            ttft_ms=pdc.ttft_ms,
            tpot_ms=pdc.tpot_ms,
            meets_slo=pdc.meets_slo,
            concurrency=pdc.decode.concurrency if pdc.decode else None,
            batch_size=pdc.decode.batch_size if pdc.decode else None,
            num_devices=pdc.devices_used,
            parallel=None,
            parallel_str=f"prefill[{p_str}]/decode[{d_str}]" if (p_str or d_str) else None,
            quantize=QuantizeSpec(linear=linear, non_expert_linear=non_expert_linear, attention=attention),
            backend=self.framework,
            system=pdc.system,
            extra={
                "pd_ratio": pdc.pd_ratio,
                "label": pdc.label,
                "tokens_per_s_per_user": pdc.tokens_per_s_per_user,
                "request_latency_ms": pdc.request_latency_ms,
                "decode_tps_per_instance": pdc.decode_tps_per_instance,
                "prefill": {
                    "instances": pdc.prefill.instances if pdc.prefill else None,
                    "devices_per_instance": pdc.prefill.devices_per_instance if pdc.prefill else None,
                    "parallel": p_str,
                    "batch_size": pdc.prefill.batch_size if pdc.prefill else None,
                    "concurrency": pdc.prefill.concurrency if pdc.prefill else None,
                    "qps_per_instance": pdc.prefill.qps_per_instance if pdc.prefill else None,
                },
                "decode": {
                    "instances": pdc.decode.instances if pdc.decode else None,
                    "devices_per_instance": pdc.decode.devices_per_instance if pdc.decode else None,
                    "parallel": d_str,
                    "batch_size": pdc.decode.batch_size if pdc.decode else None,
                    "concurrency": pdc.decode.concurrency if pdc.decode else None,
                    "qps_per_instance": pdc.decode.qps_per_instance if pdc.decode else None,
                },
            },
        )

    def _side_candidate(self, ctx: RunnerContext, pdc: PdRatioCandidate, side: str) -> Candidate | None:
        """PD-ratio 变体的 P 侧 / D 侧视角（单实例口径）。"""
        src = pdc.prefill if side == "prefill" else pdc.decode
        if src is None:
            return None
        request = ctx.request
        linear, non_expert_linear, attention = resolve_ms_quantize_actions(
            request.quantize,
            request.quantize_linear_action,
            request.quantize_non_expert_linear_action,
            request.quantize_attention_action,
        )
        mode = MODE_DISAGG_PREFILL if side == "prefill" else MODE_DISAGG_DECODE
        return Candidate(
            framework=self.framework,
            serving_mode=MODE_DISAGG,
            mode=mode,
            throughput_tokens_per_s=src.tokens_per_s_per_instance,
            throughput_tokens_per_s_per_device=src.tokens_per_s_per_device,
            throughput_req_per_s=src.qps_per_instance,
            ttft_ms=src.ttft_ms,
            tpot_ms=src.tpot_ms,
            meets_slo=meets_slo(src.ttft_ms, src.tpot_ms, request),
            concurrency=src.concurrency,
            batch_size=src.batch_size,
            num_devices=src.devices_per_instance,
            parallel=src.parallel,
            parallel_str=src.parallel_str,
            quantize=QuantizeSpec(linear=linear, non_expert_linear=non_expert_linear, attention=attention),
            backend=self.framework,
            system=pdc.system,
            extra={"instances": src.instances, "qps_allocated": src.qps_allocated},
        )

    # ------------------------------------------------------------------ 子任务

    def _run_subtask(self, ctx: RunnerContext, mode: str, rel_dir: str, log_name: str, combo: PdRatioCombo | None = None):
        """跑一个 msmodeling 子任务，返回 ``OptimizerSummary``（PD-ratio 只用第 0 个）。

        PD-ratio 与 agg 都用**集群总卡数**下发 ``--num-devices``；PD-ratio 额外给出
        每侧的每实例卡数，由 msmodeling 内部完成两阶段求解与实例配比。
        """
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
        if combo is not None:
            argv += [
                "--enable-optimize-prefill-decode-ratio",
                "--prefill-devices-per-instance",
                str(combo.prefill_devices),
                "--decode-devices-per-instance",
                str(combo.decode_devices),
            ]
        argv += [str(x) for x in ms_extra_flags(request, mode)]
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
            return None

        summary = summaries[0]
        if mode != MODE_AGG:
            return summary

        # ---- agg：保留既有的 summary.csv / candidates.json 落盘 ----
        df = summary.get_summary_df()
        ctx.artifacts.write_dataframe(f"msmodeling/{rel_dir}/summary.csv", df)
        best_row = summary.get_best_result_row()
        if best_row is None:
            if df is None or getattr(df, "empty", True):
                return None
            best_row = df.iloc[0]
        candidates = self._candidates_from_df(ctx, df)
        best = self._candidate_from_row(ctx, best_row)
        ctx.artifacts.write_json(
            f"msmodeling/{rel_dir}/candidates.json",
            {
                "best": best.to_dict() if best else None,
                "candidates": [c.to_dict() for c in candidates],
            },
        )
        return summary

    def _safe_subtask(
        self,
        ctx: RunnerContext,
        failures: list[str],
        *,
        scope: str,
        mode: str,
        rel_dir: str,
        log_name: str,
        combo: PdRatioCombo | None = None,
    ) -> tuple[list[Candidate], Candidate | None]:
        """包一层异常处理，把失败记入 failures 而不是中断整轮搜索。"""
        request = ctx.request
        try:
            summary = self._run_subtask(ctx, mode, rel_dir, log_name, combo)
            if summary is None:
                failures.append(f"{scope} msmodeling produced no result")
                return [], None
            df = summary.get_summary_df()
            candidates = self._candidates_from_df(ctx, df)
            best_row = summary.get_best_result_row()
            if best_row is None:
                if df is None or getattr(df, "empty", True):
                    failures.append(
                        f"{scope} no configuration satisfied "
                        f"TTFT={request.ttft_limit}/TPOT={request.tpot_limit}"
                    )
                    return [], None
                best_row = df.iloc[0]
            return candidates, self._candidate_from_row(ctx, best_row)
        except SystemExit as exc:
            failures.append(f"{scope} msmodeling argument parsing exited (code={exc.code})")
        except Exception as exc:
            failures.append(f"{scope} {exc}")
        return [], None

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

    # ------------------------------------------------------------------ 落盘

    @staticmethod
    def _write_all_candidates(ctx: RunnerContext, candidates: list[PdRatioCandidate]) -> None:
        """跨组合合并的候选：先写 JSON（无依赖），再尽力写 CSV。"""
        if not candidates:
            return
        rows = [_flatten_pd_candidate(index, c) for index, c in enumerate(candidates, start=1)]
        ctx.artifacts.write_json("msmodeling/pd_ratio/all_candidates.json", rows)
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return
        ctx.artifacts.write_dataframe("msmodeling/pd_ratio/all_candidates.csv", pd.DataFrame(rows))

    # ------------------------------------------------------------------ 归一化

    def _candidates_from_df(self, ctx: RunnerContext, df, mode: str = MODE_AGG) -> list[Candidate]:
        if df is None or getattr(df, "empty", True):
            return []
        return [self._candidate_from_row(ctx, df.iloc[i], mode) for i in range(len(df))]

    def _candidate_from_row(self, ctx: RunnerContext, row, mode: str = MODE_AGG) -> Candidate:
        request = ctx.request
        default_linear, default_non_expert_linear, default_attention = resolve_ms_quantize_actions(
            request.quantize,
            request.quantize_linear_action,
            request.quantize_non_expert_linear_action,
            request.quantize_attention_action,
        )

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
            serving_mode=MODE_AGG if mode == MODE_AGG else MODE_DISAGG,
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
            extra={"model_id": _row_get(row, "model_id"), **_sided_moe_extra(row, "")},
            raw=sanitize_raw(row),
        )


def _flatten_pd_candidate(rank: int, c: PdRatioCandidate) -> dict:
    """把 PD 候选压平成一行，便于 CSV / 前端表格直接消费。"""
    prefill = c.prefill or PdRatioSide()
    decode = c.decode or PdRatioSide()
    row: dict = {
        "rank": rank,
        "label": c.label,
        "pd_ratio": c.pd_ratio,
        "p_instances": prefill.instances,
        "d_instances": decode.instances,
        "p_devices_per_instance": prefill.devices_per_instance,
        "d_devices_per_instance": decode.devices_per_instance,
        "p_devices_total": prefill.devices_total,
        "d_devices_total": decode.devices_total,
        "devices_used": c.devices_used,
        "num_devices": c.num_devices,
        "p_parallel": prefill.parallel_str,
        "d_parallel": decode.parallel_str,
        "p_batch_size": prefill.batch_size,
        "d_batch_size": decode.batch_size,
        "p_concurrency": prefill.concurrency,
        "d_concurrency": decode.concurrency,
        "p_qps_per_instance": prefill.qps_per_instance,
        "d_qps_per_instance": decode.qps_per_instance,
        "p_qps_allocated": prefill.qps_allocated,
        "d_qps_allocated": decode.qps_allocated,
        "ttft_ms": c.ttft_ms,
        "tpot_ms": c.tpot_ms,
        "request_latency_ms": c.request_latency_ms,
        "decode_tps_per_instance": c.decode_tps_per_instance,
        "decode_tps_per_device": decode.tokens_per_s_per_device,
        "decode_tps_cluster": c.decode_tps_cluster,
        "prefill_tps_per_instance": prefill.tokens_per_s_per_instance,
        "prefill_tps_cluster": prefill.tokens_per_s_cluster,
        "throughput_tokens_per_s": c.throughput_tokens_per_s,
        "throughput_tokens_per_s_per_device": c.throughput_tokens_per_s_per_device,
        "tokens_per_s_per_user": c.tokens_per_s_per_user,
        "throughput_req_per_s": c.throughput_req_per_s,
        "meets_slo": c.meets_slo,
    }
    for side, src in (("p", prefill), ("d", decode)):
        parallel = src.parallel
        row[f"{side}_tp"] = parallel.tp if parallel else None
        row[f"{side}_pp"] = parallel.pp if parallel else None
        row[f"{side}_dp"] = parallel.dp if parallel else None
        row[f"{side}_ep"] = parallel.ep if parallel else None
    return row
