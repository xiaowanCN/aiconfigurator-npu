"""统一推荐服务：CLI 与未来 HTTP 共用的编排层。"""

from __future__ import annotations

import logging
from datetime import datetime

from infer_strategy_npu.config import FRAMEWORK_AIC_NPU, get_local_model_root
from infer_strategy_npu.models import resolve_model
from infer_strategy_npu.output import RunArtifacts, render_summary
from infer_strategy_npu.quantize import normalize_quantize
from infer_strategy_npu.routing import RoutingError, route
from infer_strategy_npu.runners import AicNpuRunner, MsmodelingRunner
from infer_strategy_npu.runners.base import ProgressFn, RunnerContext, throughput_of
from infer_strategy_npu.schema import Candidate, RecommendRequest, UnifiedRunResult

logger = logging.getLogger("infer_strategy_npu")


def _collect_best_candidates(frameworks) -> list[Candidate]:
    candidates: list[Candidate] = []
    for fw in frameworks:
        for mode_result in (fw.agg, fw.disagg):
            if mode_result is None or not mode_result.available:
                continue
            if mode_result.best is not None:
                candidates.append(mode_result.best)
    return candidates


def _build_recommendation(frameworks) -> dict | None:
    candidates = _collect_best_candidates(frameworks)
    if not candidates:
        return None
    best = max(candidates, key=lambda c: throughput_of(c.throughput_tokens_per_s, c.throughput_tokens_per_s_per_device))
    return {
        "framework": best.framework,
        "serving_mode": best.serving_mode,
        "mode": best.mode,
        "best": best.to_dict(),
        "reason": "highest throughput (tokens/s) among the agg / disagg best configurations",
    }


def recommend(
    request: RecommendRequest,
    *,
    log: logging.Logger | None = None,
    run_id: str | None = None,
    on_progress: ProgressFn | None = None,
) -> UnifiedRunResult:
    """执行一次推荐：路由 -> 下发框架 -> 归一化结果 -> 落盘。

    ``run_id`` 由 HTTP 服务传入（任务 id == 结果目录名）；CLI 不传则自动生成。
    ``on_progress`` 为可选的进度回调（见 :data:`~.runners.base.ProgressFn`），
    CLI 下为 None，不影响既有行为。
    """
    active_logger = log or logger
    result = UnifiedRunResult(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        request=request.to_dict(),
    )

    try:
        request.quantize = normalize_quantize(request.quantize)
        result.request = request.to_dict()
    except ValueError as exc:
        result.status = "failed"
        result.errors.append(str(exc))
        return result

    # msmodeling 的本地模型根目录：默认用相对路径 model_configs/msmodeling
    if not request.local_model_root:
        request.local_model_root = get_local_model_root()

    try:
        model_spec = resolve_model(request.model)
    except ValueError as exc:
        result.status = "failed"
        result.errors.append(str(exc))
        return result
    result.request = request.to_dict()

    artifacts = RunArtifacts.create(
        request.out_dir,
        model=model_spec.model_id,
        device=request.device,
        num_devices=request.num_devices,
        run_id=run_id,
    )
    result.run_id = artifacts.run_id
    result.artifacts = {
        "run_dir": str(artifacts.run_dir),
        "logs_dir": str(artifacts.logs_dir),
    }

    try:
        decision = route(request, model_spec, aic_save_dir=str(artifacts.aic_npu_dir))
    except RoutingError as exc:
        result.status = "failed"
        result.errors.append(str(exc))
        result.routing = {"framework": None, "reason": str(exc), "commands": []}
        run_json = artifacts.write_json("run.json", result.to_dict())
        result.artifacts["run_json"] = str(run_json)
        return result

    result.routing = decision.to_dict()

    if request.dry_run:
        artifacts.write_json("request.json", request.to_dict())
        run_json = artifacts.write_json("run.json", result.to_dict())
        result.artifacts["run_json"] = str(run_json)
        return result

    runner = AicNpuRunner() if decision.framework == FRAMEWORK_AIC_NPU else MsmodelingRunner()
    ctx = RunnerContext(
        request=request,
        model_spec=model_spec,
        decision=decision,
        artifacts=artifacts,
        logger=active_logger,
        on_progress=on_progress,
    )

    framework_run = runner.run(ctx)
    result.frameworks = [framework_run]
    if framework_run.status != "ok":
        result.status = "failed" if framework_run.status == "failed" else "partial"
        if framework_run.message:
            result.errors.append(framework_run.message)

    result.recommendation = _build_recommendation(result.frameworks)
    if result.recommendation is None and result.status == "ok":
        result.status = "partial"
        result.errors.append("no usable agg/disagg result was produced")

    artifacts.write_json("request.json", request.to_dict())
    run_json = artifacts.write_json("run.json", result.to_dict())
    summary_path = artifacts.write_text("summary.txt", render_summary(result))
    result.artifacts["run_json"] = str(run_json)
    result.artifacts["summary"] = str(summary_path)

    # 单文件静态报告：CLI 与 HTTP 共用，失败不影响主流程
    if result.status in ("ok", "partial"):
        try:
            from infer_strategy_npu.reporting import write_report

            report = write_report(result.to_dict(), artifacts.run_dir)
            result.artifacts["report"] = str(report)
        except Exception:  # noqa: BLE001
            pass

    result.artifacts["files"] = list(artifacts.files)
    return result
