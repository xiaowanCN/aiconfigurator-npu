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

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infer_strategy_npu import __version__
from infer_strategy_npu.api.schemas import (
    TaskCreateRequest,
    envelope,
    error_detail,
    supported_device_names,
)
from infer_strategy_npu.api.tasks import InvalidState, QueueFull, TaskManager
from infer_strategy_npu.models import supported_models

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


def _task_view(rec) -> dict:
    """任务详情：状态 + 进度 + 结果（结果内嵌）。"""
    progress = manager.progress(rec.task_id)
    return {
        **rec.to_dict(),
        "progress": progress,
        "request": rec.request,
        "result": manager.result(rec.task_id),
        "artifacts": {
            "run_dir": str(manager.out_dir / rec.task_id),
            "run_json": str(manager.out_dir / rec.task_id / "run.json"),
            "summary": str(manager.out_dir / rec.task_id / "summary.txt"),
        },
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
    request = payload.to_request_dict()
    # 模型/设备的取值校验复用领域层的白名单
    try:
        from infer_strategy_npu.models import resolve_model

        resolve_model(request["model"])
    except ValueError as exc:
        raise HTTPException(400, detail=envelope(400, {"errors": [error_detail("model", str(exc))]})) from exc
    if request["device"] not in supported_device_names():
        raise HTTPException(
            400,
            detail=envelope(400, {"errors": [error_detail("device", f"unsupported device {request['device']!r}")]}),
        )

    try:
        rec = manager.create(request)
    except QueueFull as exc:
        raise HTTPException(429, detail=envelope(429, {"reason": str(exc)})) from exc

    response.headers["Location"] = f"{API}/tasks/{rec.task_id}"
    return envelope(202, _task_view(rec))


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
def get_task(task_id: str) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    return envelope(200, _task_view(rec))


@app.get(f"{API}/tasks/{{task_id}}/result", summary="完整结果（run.json）")
def get_result(task_id: str) -> dict:
    rec = manager.get(task_id)
    if rec is None:
        raise HTTPException(404, detail=envelope(404, {"task_id": task_id}))
    result = manager.result(task_id)
    return envelope(200, result)


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
