"""aic-npu Runner：调用 ``aiconfigurator_npu.cli.api.cli_default`` 并解析 agg/disagg 结果。"""

from __future__ import annotations

from infer_strategy_npu.config import (
    AIC_NPU_BACKEND,
    AIC_NPU_BACKEND_VERSION,
    AIC_NPU_DATABASE_MODE,
    FRAMEWORK_AIC_NPU,
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
    tail_text,
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


def _row_get(row, key: str):
    try:
        return row.get(key)
    except (AttributeError, KeyError):
        return None


def _parallel(row, prefix: str = "") -> ParallelConfig:
    """aic-npu 的并行列 -> 统一并行字段。

    aic-npu 的 ``moe_ep`` 就是 MoE 专家并行，映射到统一字段 ``ep``；
    ``moe_tp``（aic-npu 记作 ``etp``）属于非统一字段，落到 ``extra``。
    """
    return ParallelConfig(
        tp=to_int(_row_get(row, f"{prefix}tp")),
        pp=to_int(_row_get(row, f"{prefix}pp")),
        dp=to_int(_row_get(row, f"{prefix}dp")),
        ep=to_int(_row_get(row, f"{prefix}moe_ep")),
    )


def _moe_extra(row, prefix: str = "") -> dict[str, int | None]:
    return {"moe_tp": to_int(_row_get(row, f"{prefix}moe_tp"))}


def _side_devices(row, prefix: str) -> int | None:
    """aic-npu 的 ``gpus/worker = tp * pp * dp``，对应 PD-ratio 的每实例卡数。"""
    values = (to_int(_row_get(row, f"{prefix}{key}")) for key in ("tp", "pp", "dp"))
    devices = 1
    for value in values:
        if value is None:
            return None
        devices *= value
    return devices


class AicNpuRunner(FrameworkRunner):
    framework = FRAMEWORK_AIC_NPU

    def run(self, ctx: RunnerContext) -> FrameworkRun:
        request = ctx.request
        decision = ctx.decision
        log_path = ctx.artifacts.log_path("aic_npu")

        run = FrameworkRun(
            framework=self.framework,
            commands=list(decision.commands),
            artifacts={"log": str(log_path)},
        )

        try:
            from aiconfigurator_npu.cli.api import cli_default
        except Exception as exc:  # pragma: no cover - 依赖缺失
            run.status = "failed"
            run.message = f"failed to import aiconfigurator_npu: {exc}"
            return run

        try:
            with capture_stdout(log_path):
                result = cli_default(
                    model_path=decision.aic_model_path,
                    total_gpus=request.num_devices,
                    system=decision.aic_system,
                    backend=AIC_NPU_BACKEND,
                    backend_version=AIC_NPU_BACKEND_VERSION,
                    database_mode=AIC_NPU_DATABASE_MODE,
                    isl=request.input_length,
                    osl=request.output_length,
                    ttft=float(request.ttft_limit),
                    tpot=float(request.tpot_limit),
                    top_n=request.top_n,
                    save_dir=None,
                )
        except SystemExit as exc:  # aic-npu 内部 parser.error / 早退
            run.status = "failed"
            run.message = f"aic-npu exited with code {exc.code}: {tail_text(log_path)}"
            return run
        except Exception as exc:
            run.status = "failed"
            run.message = f"aic-npu failed: {exc}"
            return run

        # aiconfigurator_npu 的 save_results 在 model_path 为本地文件时, 会把结果目录
        # 拼成绝对路径(写到 model_path 同级目录)而非 save_dir; 这里不依赖它的落盘布局,
        # 直接消费 cli_default 返回的 in-memory best_configs(与落盘 csv 内容一致).
        best_configs = result.best_configs or {}

        run.artifacts["chosen_exp"] = getattr(result, "chosen_exp", None)
        if getattr(result, "best_throughputs", None):
            run.artifacts["best_throughputs"] = {k: to_float(v) for k, v in result.best_throughputs.items()}

        # 把 best_configs 写成 csv, 落到 out_dir/aic_npu/<mode>/, 便于 HTTP/解析直接消费
        for mode in (MODE_AGG, MODE_DISAGG):
            df = best_configs.get(mode)
            if df is not None:
                ctx.artifacts.write_dataframe(f"aic_npu/{mode}/best_config_topn.csv", df)

        # aic-npu 一次调用必然同时产出 agg + disagg，--serving-modes 只过滤输出，
        # 省不掉求解（要真正跳过需要动 aiconfigurator_npu.cli.api）。
        notes: list[str] = []
        if not request.wants_serving_mode(MODE_AGG) or not request.wants_serving_mode(MODE_DISAGG):
            notes.append(
                f"aic-npu 一次调用同时产出 agg/disagg，--serving-modes={request.serving_modes} 仅过滤输出，不减少计算"
            )
        run.agg = (
            self._parse_mode(ctx, best_configs.get(MODE_AGG), MODE_AGG)
            if request.wants_serving_mode(MODE_AGG)
            else None
        )
        run.disagg = (
            self._parse_mode(ctx, best_configs.get(MODE_DISAGG), MODE_DISAGG)
            if request.wants_serving_mode(MODE_DISAGG)
            else None
        )

        if not self._any_available(run):
            run.status = "failed"
            notes.append(f"aic-npu produced no usable agg/disagg results: {tail_text(log_path)}")
        if notes:
            run.message = " | ".join(notes)
        # aic-npu 一次调用即产出 agg+disagg，进度按单个步骤上报
        ctx.report_progress("aic_npu", 1, 1)
        return run

    # ------------------------------------------------------------------ 解析

    @staticmethod
    def _any_available(run: FrameworkRun) -> bool:
        return any(
            mode_result is not None and mode_result.available for mode_result in (run.agg, run.disagg)
        )

    def _parse_mode(self, ctx: RunnerContext, df, mode: str) -> ServingModeResult:
        output = ServingModeResult(mode=mode)
        if df is None or df.empty:
            return output

        output.available = True
        output.source_files.append("<in-memory best_configs from cli_default>")
        rows = [df.iloc[i] for i in range(len(df))]

        if mode == MODE_AGG:
            output.candidates = [self._agg_candidate(ctx, row) for row in rows]
            output.best = output.candidates[0]
        else:
            best_row = rows[0]
            output.candidates = [self._disagg_overall(ctx, best_row)]
            output.best = output.candidates[0]
            output.prefill = self._disagg_side(ctx, best_row, "(p)", MODE_DISAGG_PREFILL)
            output.decode = self._disagg_side(ctx, best_row, "(d)", MODE_DISAGG_DECODE)
        return output

    def _agg_candidate(self, ctx: RunnerContext, row) -> Candidate:
        request = ctx.request
        linear, attention = to_ms_actions(request.quantize)
        ttft = to_float(_row_get(row, "ttft"))
        tpot = to_float(_row_get(row, "tpot"))
        return Candidate(
            framework=self.framework,
            serving_mode=MODE_AGG,
            mode=MODE_AGG,
            throughput_tokens_per_s=to_float(_row_get(row, "tokens/s")),
            throughput_tokens_per_s_per_device=to_float(_row_get(row, "tokens/s/gpu")),
            throughput_req_per_s=to_float(_row_get(row, "seq/s")),
            ttft_ms=ttft,
            tpot_ms=tpot,
            meets_slo=meets_slo(ttft, tpot, request),
            concurrency=to_int(_row_get(row, "concurrency")),
            batch_size=to_int(_row_get(row, "bs")),
            num_devices=to_int(_row_get(row, "num_total_gpus")),
            parallel=_parallel(row),
            parallel_str=_row_get(row, "parallel"),
            quantize=QuantizeSpec(linear=linear, attention=attention),
            backend=_row_get(row, "backend"),
            backend_version=_row_get(row, "version"),
            system=_row_get(row, "system"),
            extra={
                "tokens_per_s_gpu_cluster": to_float(_row_get(row, "tokens/s/gpu_cluster")),
                "power_w": to_float(_row_get(row, "power_w")),
                "gemm_quant": _row_get(row, "gemm"),
                "kvcache_quant": _row_get(row, "kvcache"),
                **_moe_extra(row),
            },
            raw=sanitize_raw(row),
        )

    def _disagg_overall(self, ctx: RunnerContext, row) -> Candidate:
        request = ctx.request
        linear, attention = to_ms_actions(request.quantize)
        ttft = to_float(_row_get(row, "ttft"))
        tpot = to_float(_row_get(row, "tpot"))
        p_str = _row_get(row, "(p)parallel")
        d_str = _row_get(row, "(d)parallel")
        return Candidate(
            framework=self.framework,
            serving_mode=MODE_DISAGG,
            mode=MODE_DISAGG,
            throughput_tokens_per_s=to_float(_row_get(row, "tokens/s")),
            throughput_tokens_per_s_per_device=to_float(_row_get(row, "tokens/s/gpu")),
            throughput_req_per_s=to_float(_row_get(row, "seq/s")),
            ttft_ms=ttft,
            tpot_ms=tpot,
            meets_slo=meets_slo(ttft, tpot, request),
            concurrency=to_int(_row_get(row, "concurrency")),
            batch_size=to_int(_row_get(row, "(p)bs")),
            num_devices=to_int(_row_get(row, "num_total_gpus")),
            parallel=None,
            parallel_str=f"prefill[{p_str}]/decode[{d_str}]" if (p_str or d_str) else None,
            quantize=QuantizeSpec(linear=linear, attention=attention),
            backend=_row_get(row, "(p)backend") or _row_get(row, "(d)backend"),
            backend_version=_row_get(row, "(p)version") or _row_get(row, "(d)version"),
            system=_row_get(row, "(p)system") or _row_get(row, "(d)system"),
            extra={
                "tokens_per_s_gpu_cluster": to_float(_row_get(row, "tokens/s/gpu_cluster")),
                "power_w": to_float(_row_get(row, "power_w")),
                "prefill": {
                    "parallel": p_str,
                    "workers": to_int(_row_get(row, "(p)workers")),
                    "devices_per_instance": _side_devices(row, "(p)"),
                    "bs": to_int(_row_get(row, "(p)bs")),
                    "seq_per_s_per_worker": to_float(_row_get(row, "(p)seq/s/worker")),
                    **_moe_extra(row, "(p)"),
                },
                "decode": {
                    "parallel": d_str,
                    "workers": to_int(_row_get(row, "(d)workers")),
                    "devices_per_instance": _side_devices(row, "(d)"),
                    "bs": to_int(_row_get(row, "(d)bs")),
                    "seq_per_s_per_worker": to_float(_row_get(row, "(d)seq/s/worker")),
                    **_moe_extra(row, "(d)"),
                },
            },
            raw=sanitize_raw(row),
        )

    def _disagg_side(self, ctx: RunnerContext, row, prefix: str, mode: str) -> Candidate:
        """用 disagg 行的 (p)* / (d)* 列构造 PD分离-Prefill / PD分离-Decode 视角。"""
        request = ctx.request
        linear, attention = to_ms_actions(request.quantize)
        workers = to_int(_row_get(row, f"{prefix}workers"))
        seq_per_worker = to_float(_row_get(row, f"{prefix}seq/s/worker"))
        req_per_s = None
        if workers is not None and seq_per_worker is not None:
            req_per_s = workers * seq_per_worker
        length = request.input_length if mode == MODE_DISAGG_PREFILL else request.output_length
        tokens_per_s = None if req_per_s is None else req_per_s * length

        tp = to_int(_row_get(row, f"{prefix}tp"))
        pp = to_int(_row_get(row, f"{prefix}pp"))
        dp = to_int(_row_get(row, f"{prefix}dp"))
        devices = None
        if workers is not None and None not in (tp, pp, dp):
            devices = workers * tp * pp * dp

        ttft = to_float(_row_get(row, "ttft")) if mode == MODE_DISAGG_PREFILL else None
        tpot = to_float(_row_get(row, "tpot")) if mode == MODE_DISAGG_DECODE else None

        return Candidate(
            framework=self.framework,
            serving_mode=MODE_DISAGG,
            mode=mode,
            throughput_tokens_per_s=tokens_per_s,
            throughput_tokens_per_s_per_device=(
                None if (tokens_per_s is None or not devices) else tokens_per_s / devices
            ),
            throughput_req_per_s=req_per_s,
            ttft_ms=ttft,
            tpot_ms=tpot,
            meets_slo=meets_slo(ttft, tpot, request),
            concurrency=to_int(_row_get(row, "concurrency")),
            batch_size=to_int(_row_get(row, f"{prefix}bs")),
            num_devices=devices,
            parallel=_parallel(row, prefix),
            parallel_str=_row_get(row, f"{prefix}parallel"),
            quantize=QuantizeSpec(linear=linear, attention=attention),
            backend=_row_get(row, f"{prefix}backend"),
            backend_version=_row_get(row, f"{prefix}version"),
            system=_row_get(row, f"{prefix}system"),
            extra={"workers": workers, "seq_per_s_per_worker": seq_per_worker, **_moe_extra(row, prefix)},
            raw=sanitize_raw(row),
        )
