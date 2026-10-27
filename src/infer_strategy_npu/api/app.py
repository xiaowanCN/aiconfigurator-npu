"""HTTP 服务：把 infer-strategy-npu 封装成 RESTful 接口。

启动：
    pip install -e ".[api]"
    infer-strategy-npu-server --host 0.0.0.0 --port 8000
    # 或 uvicorn infer_strategy_npu.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from infer_strategy_npu import __version__
from infer_strategy_npu.api.schemas import (
    TaskCreateRequest,
    envelope,
    error_detail,
    supported_device_names,
)
from infer_strategy_npu.api.tasks import InvalidState, QueueFull, TaskManager
from infer_strategy_npu.models import resolve_model, supported_models

API = "/api/v1"

app = FastAPI(
    title="infer-strategy-npu",
    version=__version__,
    description="NPU 推理部署策略推荐（aic-npu / msmodeling）的异步任务服务",
)

# 前端 demo 通常跑在独立端口（如 vite 的 5173），必须放开跨域，否则浏览器会直接拦掉请求。
# 可用 INFER_STRATEGY_API_CORS_ORIGINS 指定白名单（逗号分隔），默认 "*"。
_cors_origins = os.environ.get("INFER_STRATEGY_API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


manager = TaskManager(
    os.environ.get("INFER_STRATEGY_API_OUT_DIR"),
    max_concurrent=_env_int("INFER_STRATEGY_API_MAX_CONCURRENT", 1),
    max_queued=_env_int("INFER_STRATEGY_API_MAX_QUEUED", 32),
    task_timeout=_env_int("INFER_STRATEGY_API_TASK_TIMEOUT_SEC", 7200),
)


def _task_view(rec, *, include_raw: bool = True, include_variant_candidates: bool = True) -> dict:
    """任务详情：状态 + 进度 + 结果（结果内嵌）。"""
    progress = manager.progress(rec.task_id)
    return {
        **rec.to_dict(),
        "progress": progress,
        "request": rec.request,
        "result": _trim_result(manager.result(rec.task_id), include_raw, include_variant_candidates),
        "artifacts": {
            "run_dir": str(manager.out_dir / rec.task_id),
            "run_json": str(manager.out_dir / rec.task_id / "run.json"),
            "summary": str(manager.out_dir / rec.task_id / "summary.txt"),
        },
    }


def _trim_result(result: dict | None, include_raw: bool, include_variant_candidates: bool) -> dict | None:
    """裁剪 run.json 里体积最大的两块（PD-ratio 下 397KB -> 约 60KB）。

    - ``raw``：每条候选都挂着一个 48 字段的原始 PD 行；
    - ``disagg_variants[].pd_ratio_candidates``：与 framework 级的
      ``pd_ratio_candidates`` 完全重复。
    """
    if result is None or (include_raw and include_variant_candidates):
        return result

    frameworks = result.get("frameworks")
    if not frameworks:
        return result

    # 1) 先摘掉与 framework 级 pd_ratio_candidates 重复的变体级副本
    if not include_variant_candidates:
        for fw in frameworks:
            for variant in fw.get("disagg_variants") or []:
                variant.pop("pd_ratio_candidates", None)

    # 2) 再递归摘掉所有 raw（每条 PD 候选挂着一个 48 字段的原始行）
    if not include_raw:
        result["frameworks"] = _drop_raw(frameworks)
        if result.get("recommendation"):
            result["recommendation"] = _drop_raw(result["recommendation"])
    return result


def _drop_raw(obj: Any) -> Any:
    """递归删除所有名为 ``raw`` 的键。"""
    if isinstance(obj, dict):
        return {k: _drop_raw(v) for k, v in obj.items() if k != "raw"}
    if isinstance(obj, list):
        return [_drop_raw(v) for v in obj]
    return obj


def _pd_view(result: dict | None, *, limit: int | None, include_raw: bool) -> dict | None:
    """PD 分离专用视图：跨组合的全局排序候选 + 每组摘要。"""
    if result is None:
        return None
    req = result.get("request") or {}
    frameworks = result.get("frameworks") or []
    fw = next((f for f in frameworks if f.get("pd_ratio_candidates")), frameworks[0] if frameworks else None)
    if fw is None:
        return None

    candidates = list(fw.get("pd_ratio_candidates") or [])
    if not include_raw:
        candidates = [{k: v for k, v in c.items() if k != "raw"} for c in candidates]
    if limit and limit > 0:
        candidates = candidates[:limit]

    variants = [
        {
            "label": v.get("label"),
            "pd_ratio": v.get("pd_ratio"),
            "best": v.get("best"),
            "prefill": v.get("prefill"),
            "decode": v.get("decode"),
            "candidate_count": len(v.get("pd_ratio_candidates") or []),
        }
        for v in (fw.get("disagg_variants") or [])
    ]

    return {
        "available": bool(candidates),
        "framework": fw.get("framework"),
        "serving_modes": req.get("serving_modes"),
        "pd_ratio_sizes": req.get("pd_ratio_sizes"),
        "num_devices": req.get("num_devices"),
        "variants": variants,
        "candidates": candidates,
        "best": (fw.get("disagg") or {}).get("best"),
        "message": fw.get("message"),
        "errors": result.get("errors"),
    }


def _validate_payload(payload) -> tuple[dict, str]:
    """复用领域层白名单校验 model / device，返回 (request_dict, framework_hint)。"""
    request = payload.to_request_dict()
    try:
        resolve_model(request["model"])
    except ValueError as exc:
        raise HTTPException(400, detail=envelope(400, {"errors": [error_detail("model", str(exc))]})) from exc
    if request["device"] not in supported_device_names():
        raise HTTPException(
            400,
            detail=envelope(400, {"errors": [error_detail("device", f"unsupported device {request['device']!r}")]}),
        )
    return request, _resolve_framework(request)


def _resolve_framework(request: dict) -> str | None:
    """只做路由判定（不执行、不落盘），用于生成告警与 preview。"""
    from infer_strategy_npu.routing import route
    from infer_strategy_npu.schema import RecommendRequest

    forced = str(request.get("framework") or "auto").lower()
    if forced == "aic-npu":
        return "aic-npu"
    if forced == "msmodeling":
        return "msmodeling"
    try:
        spec = resolve_model(request["model"])
        decision = route(RecommendRequest(**request), spec, aic_save_dir="<preview>")
    except Exception:
        return None
    return decision.framework


def _pd_meta(payload) -> dict:
    """PD-ratio 网格 + 非阻断告警。"""
    return {
        "pd_ratio_grid": list(payload.pd_ratio_grid),
        "pd_skipped": list(payload.pd_skipped),
        "warnings": payload.build_warnings(_resolve_framework(payload.to_request_dict())),
    }


# ------------------------------------------------------------------ 元数据


@app.get(f"{API}/health", summary="健康检查")
def health() -> dict:
    running = sum(1 for r in manager._tasks.values() if r.status == "RUNNING")
    queued = sum(1 for r in manager._tasks.values() if r.status == "PENDING")
    return envelope(
        200,
        {
            "status": "ok",
            "version": __version__,
            "running": running,
            "queued": queued,
            "max_concurrent": manager.max_concurrent,
            "out_dir": str(manager.out_dir),
            "ui": UI_DIR.is_dir(),  # 是否由本服务托管静态界面
        },
        "服务正常",
    )


@app.get(f"{API}/models", summary="支持的模型")
def models() -> dict:
    return envelope(200, list(supported_models()))


@app.get(f"{API}/devices", summary="支持的设备")
def devices() -> dict:
    return envelope(200, supported_device_names())


# ------------------------------------------------------------------ 任务


@app.post(f"{API}/tasks", summary="创建任务（异步）", status_code=202)
def create_task(payload: TaskCreateRequest, response: Response) -> dict:
    # 模型/设备的取值校验复用领域层的白名单
    request, _ = _validate_payload(payload)

    try:
        rec = manager.create(request)
    except QueueFull as exc:
        raise HTTPException(429, detail=envelope(429, {"reason": str(exc)})) from exc

    response.headers["Location"] = f"{API}/tasks/{rec.task_id}"
    view = _task_view(rec)
    view.update(_pd_meta(payload))  # PD-ratio 网格与告警（不阻断）
    return envelope(202, view)


@app.post(f"{API}/tasks/preview", summary="dry-run：返回将要下发的命令，不创建任务")
def preview_task(payload: TaskCreateRequest) -> dict:
    """提交前先看会跑几组、命令长什么样（不 import msmodeling，秒级返回）。"""
    request, framework = _validate_payload(payload)
    from infer_strategy_npu.routing import route
    from infer_strategy_npu.schema import RecommendRequest

    try:
        spec = resolve_model(request["model"])
        decision = route(RecommendRequest(**request), spec, aic_save_dir="<preview>")
    except Exception as exc:
        raise HTTPException(400, detail=envelope(400, {"reason": f"{type(exc).__name__}: {exc}"})) from exc

    grid = list(payload.pd_ratio_grid)
    return envelope(
        200,
        {
            "framework": framework,
            "reason": decision.reason,
            **_pd_meta(payload),
            "commands": list(decision.commands),
            "total_steps": len(decision.commands),
            "pd_combo_count": len(grid),
        },
    )


@app.get(f"{API}/tasks", summary="任务列表")
def list_tasks(
    status: str | None = Query(None),
    model: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> dict:
    items = manager.list(status=status, model=model)
    total = len(items)
    start = (page - 1) * page_size
    chunk = items[start : start + page_size]
    return envelope(
        200,
        {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    **r.to_dict(),
                    "progress": manager.progress(r.task_id),
                }
                for r in chunk
            ],
        },
    )


@app.get(f"{API}/tasks/{{task_id}}", summary="任务详情（状态 + 进度 + 结果）")
def get_task(
    task_id: str,
    include_raw: bool = Query(True, description="是否保留各候选的 raw 明细（PD-ratio 下体积最大）"),
    include_variant_candidates: bool = Query(True, description="是否保留 disagg_variants[].pd_ratio_candidates（与 result 顶层重复）"),
) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    return envelope(200, _task_view(rec, include_raw=include_raw, include_variant_candidates=include_variant_candidates))


@app.get(f"{API}/tasks/{{task_id}}/result", summary="完整结果（run.json）")
def get_result(
    task_id: str,
    include_raw: bool = Query(True, description="是否保留各候选的 raw 明细（PD-ratio 下体积最大）"),
    include_variant_candidates: bool = Query(True, description="是否保留 disagg_variants[].pd_ratio_candidates（与 result 顶层重复）"),
) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    result = _trim_result(manager.result(task_id), include_raw, include_variant_candidates)
    return envelope(200, result)


@app.get(f"{API}/tasks/{{task_id}}/pd", summary="PD 分离专用视图（跨组合全局排序候选）")
def get_pd(
    task_id: str,
    limit: int | None = Query(None, ge=1, description="只返回 Top-N 候选"),
    include_raw: bool = Query(True, description="是否保留 raw 明细"),
) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    return envelope(200, _pd_view(manager.result(task_id), limit=limit, include_raw=include_raw))


@app.get(f"{API}/tasks/{{task_id}}/summary", summary="人类可读摘要")
def get_summary(task_id: str) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    return envelope(200, {"text": manager.summary_text(task_id)})


@app.get(f"{API}/tasks/{{task_id}}/logs", summary="日志文件列表")
def get_logs(task_id: str) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    return envelope(200, manager.logs(task_id))


@app.get(f"{API}/tasks/{{task_id}}/logs/{{name}}", summary="日志内容")
def get_log(task_id: str, name: str, tail: int | None = Query(None, ge=1)) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, detail=envelope(400, {"errors": [error_detail("name", "invalid log name")]}))
    text = manager.log_text(task_id, name, tail)
    if text is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id, "name": name}))
    return envelope(200, {"name": name, "text": text})


@app.post(f"{API}/tasks/{{task_id}}/cancel", summary="取消任务", status_code=202)
def cancel_task(task_id: str) -> dict:
    try:
        rec = manager.cancel(task_id)
    except KeyError as exc:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id})) from exc
    except InvalidState as exc:
        raise HTTPException(409, detail=envelope(409, {"task_id": task_id, "reason": str(exc)})) from exc
    return envelope(202, _task_view(rec), "取消请求已接受")


@app.delete(f"{API}/tasks/{{task_id}}", summary="删除任务与产物")
def delete_task(task_id: str) -> dict:
    if not manager.delete(task_id):
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    return envelope(200, {"task_id": task_id}, "删除成功")


@app.exception_handler(HTTPException)
async def _http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


# ------------------------------------------------------------------ 静态 UI
# 由 FastAPI 自身托管 ui/：同源访问，既绕开 file:// 的 CORS 限制，
# 也让表单可以直接 POST /api/v1/tasks。
# 注意：必须挂在所有 API 路由**之后**，否则会吞掉 /api/v1/*。
UI_DIR = Path(__file__).resolve().parents[3] / "ui"

if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
else:  # pragma: no cover - 以 wheel 安装、无 ui/ 目录时跳过
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infer-strategy-npu-server", description="启动 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--max-queued", type=int, default=None)
    parser.add_argument("--task-timeout", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    if args.out_dir:
        manager.out_dir = Path(args.out_dir).expanduser()
        manager.out_dir.mkdir(parents=True, exist_ok=True)
    if args.max_concurrent is not None:
        manager.max_concurrent = max(1, args.max_concurrent)
    if args.max_queued is not None:
        manager.max_queued = max(0, args.max_queued)
    if args.task_timeout is not None:
        manager.task_timeout = args.task_timeout
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
