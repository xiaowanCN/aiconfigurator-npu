"""任务层：任务注册、子进程执行、状态与进度落盘。

设计要点：
- 任务在**独立子进程**（spawn）里跑，崩溃隔离、可取消、不受 ``sys.argv`` 改写影响
- 状态/进度全部落盘（``task.json`` / ``progress.json`` / ``done.json``），
  因此不需要 IPC，且服务重启后可重建
- 任务 id == 结果目录名（``<out_dir>/<task_id>/``）
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from infer_strategy_npu.config import DEFAULT_OUT_DIR
from infer_strategy_npu.output import make_run_id

TERMINAL_STATUSES = ("SUCCESS", "FAILED", "CANCELLED", "TIMEOUT")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ 子进程入口


def _run_task(out_dir: str, task_id: str, request_dict: dict) -> None:
    """子进程入口：调 recommend()，把进度与终态写盘。"""
    from infer_strategy_npu.schema import RecommendRequest
    from infer_strategy_npu.service import recommend

    run_dir = Path(out_dir) / task_id

    def on_progress(stage: str, done: int, total: int) -> None:
        # stage 形如 "agg" 或 "P8D8/pd_ratio"；拆出 kind/label 便于调用方分类
        head, sep, tail = stage.partition("/")
        if sep:
            label, kind = head, tail  # "P8D8" / "pd_ratio"
        else:
            label, kind = None, stage  # None / "agg"
        _atomic_write_json(
            run_dir / "progress.json",
            {
                "stage": stage,
                "kind": kind,          # "agg" | "pd_ratio"
                "label": label or None,  # PD-ratio 组合标签，如 "P8D8"
                "done": done,
                "total": total,
                "percent": round(100.0 * done / total, 2) if total else 0.0,
                "updated_at": _now(),
            },
        )

    status = "SUCCESS"
    error: str | None = None
    try:
        result = recommend(RecommendRequest(**request_dict), run_id=task_id, on_progress=on_progress)
        if result.status == "failed":
            status = "FAILED"
            error = " | ".join(result.errors) or "run failed"
    except Exception as exc:  # 子进程里的任何异常都要落到 done.json，避免"静默失败"
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"

    _atomic_write_json(
        run_dir / "done.json",
        {"status": status, "error": error, "finished_at": _now()},
    )

    # 静态报告：任务成功即产出 <run_dir>/report.html（单文件，可直接分发给三方平台）
    if status == "SUCCESS":
        try:
            from infer_strategy_npu.reporting import write_report_from_run_json

            write_report_from_run_json(run_dir / "run.json")
        except Exception:  # 报告失败不影响主流程
            pass


# ------------------------------------------------------------------ 任务记录


@dataclass
class TaskRecord:
    task_id: str
    request: dict
    status: str = "PENDING"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    error: str | None = None
    process: Any | None = field(default=None, repr=False)

    @property
    def run_dir(self) -> Path:
        return Path(TaskRecord.out_dir) / self.task_id

    out_dir: str = DEFAULT_OUT_DIR  # 类属性默认值，由 TaskManager 覆盖

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
            "error": self.error,
        }


# ------------------------------------------------------------------ 管理器


class TaskManager:
    """内存注册表 + 有限并发 + 落盘状态。"""

    def __init__(
        self,
        out_dir: str | None = None,
        *,
        max_concurrent: int = 1,
        max_queued: int = 32,
        task_timeout: int = 7200,
    ) -> None:
        self.out_dir = Path(out_dir or DEFAULT_OUT_DIR).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max(1, int(max_concurrent))
        self.max_queued = max(0, int(max_queued))
        self.task_timeout = int(task_timeout)
        self._tasks: dict[str, TaskRecord] = {}
        self._order: list[str] = []
        TaskRecord.out_dir = str(self.out_dir)
        self._recover()

    # -------------------------------------------------------------- 内部

    def _recover(self) -> None:
        """重启后重建注册表：把遗留的 RUNNING/PENDING 标记为 FAILED。"""
        for entry in sorted(self.out_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta = _read_json(entry / "task.json")
            if not meta:
                continue
            rec = TaskRecord(
                task_id=meta.get("task_id", entry.name),
                request=meta.get("request", {}),
                status=meta.get("status", "PENDING"),
                created_at=meta.get("created_at", _now()),
            )
            if rec.status not in TERMINAL_STATUSES:
                rec.status = "FAILED"
                rec.error = "service restarted before the task finished"
            self._tasks[rec.task_id] = rec
            self._order.append(rec.task_id)

    def _new_task_id(self, request: dict) -> str:
        # 复用领域的 make_run_id，保证 CLI 与 HTTP 的目录命名规则完全一致
        return make_run_id(
            str(request.get("model") or "unknown"),
            str(request.get("device") or "unknown"),
            request.get("num_devices", 0),
        )

    def _running(self) -> list[TaskRecord]:
        return [r for r in self._tasks.values() if r.status == "RUNNING"]

    def _pending(self) -> list[TaskRecord]:
        return [self._tasks[t] for t in self._order if self._tasks[t].status == "PENDING"]

    def _pump(self) -> None:
        for rec in self._pending():
            if len(self._running()) >= self.max_concurrent:
                break
            self._start(rec)

    def _start(self, rec: TaskRecord) -> None:
        proc = mp.get_context("spawn").Process(
            target=_run_task,
            args=(str(self.out_dir), rec.task_id, rec.request),
            daemon=False,
        )
        proc.start()
        rec.process = proc
        rec.pid = proc.pid
        rec.status = "RUNNING"
        rec.started_at = _now()
        self._persist(rec)

    def _persist(self, rec: TaskRecord) -> None:
        _atomic_write_json(
            self.out_dir / rec.task_id / "task.json",
            {**rec.to_dict(), "request": rec.request},
        )

    def _finalize(self, rec: TaskRecord) -> None:
        proc = rec.process
        if proc is not None:
            proc.join(timeout=5)
            rec.process = None
        done = _read_json(self.out_dir / rec.task_id / "done.json") or {}
        rec.status = done.get("status") or "FAILED"
        rec.error = done.get("error") or (None if rec.status == "SUCCESS" else "task exited abnormally")
        rec.finished_at = done.get("finished_at") or _now()
        rec.pid = None
        self._persist(rec)

    def _poll(self) -> None:
        for rec in list(self._tasks.values()):
            if rec.status != "RUNNING":
                continue
            proc = rec.process
            # 超时
            if self.task_timeout > 0 and rec.started_at:
                try:
                    started = datetime.fromisoformat(rec.started_at)
                    elapsed = (datetime.now().astimezone() - started).total_seconds()
                except ValueError:
                    elapsed = 0.0
                if elapsed > self.task_timeout:
                    if proc is not None:
                        proc.terminate()
                        proc.join(timeout=10)
                        if proc.is_alive():
                            proc.kill()
                    rec.status = "TIMEOUT"
                    rec.error = f"task exceeded timeout ({self.task_timeout}s)"
                    rec.finished_at = _now()
                    rec.process = None
                    self._persist(rec)
                    continue
            if proc is None or not proc.is_alive():
                if rec.status not in TERMINAL_STATUSES:
                    self._finalize(rec)
        self._pump()

    # -------------------------------------------------------------- 对外

    def create(self, request: dict) -> TaskRecord:
        self._poll()
        queued = len(self._pending())
        if queued >= self.max_queued:
            raise QueueFull(f"queue is full (max_queued={self.max_queued})")
        task_id = self._new_task_id(request)
        rec = TaskRecord(task_id=task_id, request=request)
        self._tasks[task_id] = rec
        self._order.append(task_id)
        self._persist(rec)
        self._pump()
        return rec

    def get(self, task_id: str) -> TaskRecord | None:
        self._poll()
        return self._tasks.get(task_id)

    def list(self, *, status: str | None = None, model: str | None = None) -> list[TaskRecord]:
        self._poll()
        items = [self._tasks[t] for t in self._order if t in self._tasks]
        if status:
            items = [r for r in items if r.status == status]
        if model:
            items = [r for r in items if str(r.request.get("model")) == model]
        return items

    def cancel(self, task_id: str) -> TaskRecord:
        rec = self._tasks.get(task_id)
        if rec is None:
            raise KeyError(task_id)
        if rec.status in TERMINAL_STATUSES:
            raise InvalidState(f"task {task_id} already finished (status={rec.status})")
        if rec.status == "RUNNING" and rec.process is not None:
            rec.process.terminate()
            rec.process.join(timeout=10)
            if rec.process.is_alive():
                rec.process.kill()
            rec.process = None
        rec.status = "CANCELLED"
        rec.finished_at = _now()
        rec.pid = None
        self._persist(rec)
        self._pump()
        return rec

    def delete(self, task_id: str) -> bool:
        rec = self._tasks.pop(task_id, None)
        if rec is None:
            return False
        self._order = [t for t in self._order if t != task_id]
        import shutil

        shutil.rmtree(self.out_dir / task_id, ignore_errors=True)
        return True

    # -------------------------------------------------------------- 读取产物

    def progress(self, task_id: str) -> dict:
        data = _read_json(self.out_dir / task_id / "progress.json")
        if not data:
            return {
                "done": 0,
                "total": 0,
                "percent": 0.0,
                "stage": None,
                "kind": None,
                "label": None,
                "updated_at": None,
            }
        data.setdefault("kind", None)
        data.setdefault("label", None)
        return data

    def result(self, task_id: str) -> dict | None:
        return _read_json(self.out_dir / task_id / "run.json")

    def summary_text(self, task_id: str) -> str | None:
        path = self.out_dir / task_id / "summary.txt"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def logs(self, task_id: str) -> list[str]:
        log_dir = self.out_dir / task_id / "logs"
        if not log_dir.is_dir():
            return []
        return sorted(p.name for p in log_dir.glob("*.log"))

    def log_text(self, task_id: str, name: str, tail: int | None = None) -> str | None:
        path = self.out_dir / task_id / "logs" / name
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        if tail and tail > 0:
            text = "\n".join(text.splitlines()[-tail:])
        return text


class QueueFull(RuntimeError):
    """队列已满。"""


class InvalidState(RuntimeError):
    """任务状态不允许该操作。"""
