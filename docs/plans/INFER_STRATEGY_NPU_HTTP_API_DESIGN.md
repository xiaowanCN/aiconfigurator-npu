# infer-strategy-npu HTTP 服务设计（草案）

> 目标：把现有的 `infer-strategy-npu -recommend` 封装成 RESTful HTTP 服务，
> 支持**创建任务 / 查询任务（状态 + 进度 + 结果）**，以适配"单次运行耗时很久"的特点。
> 本文为设计草案，待评审补充后再实施。
> 分支：`feature/infer-strategy-npu-unified-cli`

---

## 1. 现状与约束

| 项 | 现状 | 对 HTTP 封装的影响 |
|---|---|---|
| 核心入口 | `infer_strategy_npu.service.recommend(request, log=None) -> UnifiedRunResult` | 已是纯函数，可直接被服务层调用 |
| 执行耗时 | 单次数分钟 ~ 数十分钟（128 卡 + xPyD 时 15 次 msmodeling 搜索） | **必须异步**：创建任务立即返回，客户端轮询 |
| 产物 | `<out_dir>/<run_id>/{run.json, request.json, summary.txt, aic_npu/, msmodeling/, logs/}` | 结果天然落盘，服务可直接读盘返回 |
| 任务标识 | `run_id = {yyyyMMdd-HHmmss}-{model}-{device}-{num_devices}`（见 `output.RunArtifacts.create`） | 时间戳前缀便于排序，但**同一秒同参数会重名**，需补唯一后缀 |
| 进度 | **目前无任何进度上报** | 需要在 runner 里加钩子（见 §5） |
| 依赖 | 仅 numpy/pandas/scipy/pyyaml/munch/matplotlib/plotext/prettytable，**无 Web 框架** | 需新增可选依赖 `fastapi + uvicorn + pydantic` |
| msmodeling | 内部用 `ProcessPoolExecutor`（spawn），且需真实 NPU | 服务应部署在 NPU 主机上；子进程隔离更安全 |

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│  客户端                                                        │
└───────────────┬──────────────────────────────────────────────┘
                │  HTTP (REST / JSON)
┌───────────────▼──────────────────────────────────────────────┐
│  API 层  src/infer_strategy_npu/api/                          │
│   ├─ app.py        FastAPI 应用、路由、统一响应封装             │
│   ├─ schemas.py    pydantic 请求/响应模型                      │
│   └─ deps.py       鉴权、配置注入                              │
├──────────────────────────────────────────────────────────────┤
│  任务层  src/infer_strategy_npu/api/tasks.py                   │
│   ├─ TaskManager   注册表、队列、并发控制、超时、取消            │
│   ├─ TaskRecord    状态机 + 进度 + 结果引用                     │
│   └─ worker.py     子进程入口（spawn），调 recommend()，写进度   │
├──────────────────────────────────────────────────────────────┤
│  领域层（已有，不改语义）                                       │
│   service.recommend() → routing → runners(aic-npu/msmodeling)  │
├──────────────────────────────────────────────────────────────┤
│  存储层  <out_dir>/<task_id>/                                  │
│   run.json（完整结果） request.json task.json progress.json     │
│   summary.txt  logs/*.log  aic_npu/ msmodeling/                │
└──────────────────────────────────────────────────────────────┘
```

**关键设计：任务在独立子进程中执行**

- 崩溃隔离：msmodeling/torch 的 OOM、abort 不会拖垮 API 进程
- 可取消：直接对子进程 PID 发信号
- 避免 GIL / 全局状态污染（`sys.argv` 被 runner 改写过，见 `_parse_args`）
- 启动方式用 **spawn**（与 msmodeling 内部一致，避免 fork 带来的 CUDA/NPU 上下文问题）

---

## 3. 任务标识与产物目录

保留"以开始时间戳识别"的习惯，同时保证唯一性：

```
task_id = {yyyyMMdd-HHmmss}-{slug(model)}-{slug(device)}-{num_devices}-{short8}
                                                                    └ uuid4().hex[:8]
```

- 前缀仍是时间戳 → 目录按创建时间自然排序，肉眼可定位
- `short8` 消除同秒重名
- **`task_id == run_id == 产物目录名`**，三者一致，不需要额外映射表

需要的改动（很小）：

- `RunArtifacts.create(..., run_id: str | None = None)` 支持外部指定
- `service.recommend()` 透传 `run_id`；或由 API 层先建目录再调用

---

## 4. 任务状态机

```
                 ┌──────────► CANCELLED
                 │  (取消请求)
   PENDING ──► RUNNING ──► SUCCESS
                 │
                 ├────────► FAILED      (异常/超时/非零退出)
                 └────────► TIMEOUT     (可选，也可归入 FAILED)
```

| 状态 | 含义 | 终态 |
|---|---|---|
| `PENDING` | 已创建，排队中（并发已满） | 否 |
| `RUNNING` | 子进程执行中 | 否 |
| `SUCCESS` | 正常结束（`UnifiedRunResult.status` 为 `ok`/`partial` 且退出码 0） | 是 |
| `FAILED` | 异常、非零退出、或结果 `status=failed` | 是 |
| `CANCELLED` | 被用户取消 | 是 |
| `TIMEOUT` | 超过 `max_runtime_sec`（可选用 `FAILED` 表达） | 是 |

任务记录字段（`task.json`）：

```jsonc
{
  "task_id": "20261020-143012-ZhipuAI-GLM-5.3-Flash-ATLAS_800_A2_376T_64G-128-a1b2c3d4",
  "status": "RUNNING",
  "progress": { "done": 5, "total": 15, "percent": 33.3,
                "stage": "msmodeling/3P1D/disagg_prefill", "updated_at": "..." },
  "created_at": "...", "started_at": "...", "finished_at": null,
  "elapsed_sec": 123.4,
  "pid": 12345,
  "error": null,
  "artifacts": { "run_dir": ".../20261020-143012-...-a1b2c3d4",
                 "run_json": ".../run.json", "summary": ".../summary.txt" }
}
```

---

## 5. 进度上报机制（需新增钩子）

现状：runner 内部完成一个子任务就继续下一个，外部完全看不到进度。

**方案：在 `recommend()` 加一个可选的进度回调，由子进程写 `progress.json`**

```python
def recommend(request, *, log=None, on_progress=None) -> UnifiedRunResult: ...
```

- `RunnerContext` 增加 `report_progress(stage: str, done: int, total: int)` 回调
- 各 runner 在每个子任务开始/结束时调用：
  - aic-npu：总数 1（一次 `cli_default`）
  - msmodeling：`total = 1(agg) + 2 * len(xPyD 网格)`，
    stage 依次为 `agg`、`{xPyD}/disagg_prefill`、`{xPyD}/disagg_decode`
- 子进程把进度 **原子写** 到 `<run_dir>/progress.json`（先写 tmp 再 rename），
  API 层按需读盘即可，**不需要队列/IPC**，且天然支持重启恢复

进度粒度是"子任务级"（粗粒度），对分钟级任务足够；更细（并行度候选级）需要改 msmodeling 内部，暂不做。

---

## 6. RESTful 接口设计

Base path：`/api/v1`，全部 `application/json`。

| Method | Path | 说明 | 成功码 |
|---|---|---|---|
| `POST` | `/api/v1/tasks` | 创建任务（异步，立即返回） | `202` |
| `GET` | `/api/v1/tasks` | 任务列表（分页 / 按 status、model 过滤） | `200` |
| `GET` | `/api/v1/tasks/{task_id}` | 任务详情：状态 + 进度 + 结果摘要 | `200` |
| `GET` | `/api/v1/tasks/{task_id}/result` | 完整结果（`run.json`，即 `UnifiedRunResult`） | `200` |
| `GET` | `/api/v1/tasks/{task_id}/summary` | 人类可读摘要（`summary.txt`） | `200` |
| `GET` | `/api/v1/tasks/{task_id}/logs` | 日志文件列表 | `200` |
| `GET` | `/api/v1/tasks/{task_id}/logs/{name}` | 日志内容（支持 `?tail=200`） | `200` |
| `POST` | `/api/v1/tasks/{task_id}/cancel` | 取消任务 | `202` |
| `DELETE` | `/api/v1/tasks/{task_id}` | 删除任务记录与产物 | `200` |
| `POST` | `/api/v1/tasks/preview` | **dry-run**：同步返回将要下发的命令 | `200` |
| `GET` | `/api/v1/models` | 支持的模型白名单 | `200` |
| `GET` | `/api/v1/devices` | 支持的设备清单 | `200` |
| `GET` | `/api/v1/health` | 健康检查（含当前并发数/队列长度） | `200` |

### 6.1 创建任务

`POST /api/v1/tasks`

请求体（字段与 `RecommendRequest` 一一对应，均为可选，缺省沿用 CLI 默认）：

```json
{
  "model": "ZhipuAI/GLM-5.3-Flash",
  "device": "ATLAS_800_A2_376T_64G",
  "num_devices": 128,
  "input_length": 8192,
  "output_length": 1024,
  "tpot_limit": 200,
  "ttft_limit": 30000,
  "quantize": "W8A8",
  "framework": "auto",
  "tp_sizes": null,
  "pd_balance_search": true,
  "pd_balance_max": 3,
  "pd_balance_sizes": null,
  "compile_model": false,
  "jobs": 32,
  "dry_run": false,
  "client_token": "optional-idempotency-key"
}
```

响应 `202 Accepted`（响应头带 `Location: /api/v1/tasks/{task_id}`）：

```json
{
  "code": 202,
  "msg": "任务已创建",
  "data": {
    "task_id": "20261020-143012-ZhipuAI-GLM-5.3-Flash-ATLAS_800_A2_376T_64G-128-a1b2c3d4",
    "status": "PENDING",
    "created_at": "2026-10-20T14:30:12+08:00",
    "progress": { "done": 0, "total": 15, "percent": 0.0, "stage": null },
    "location": "/api/v1/tasks/20261020-143012-ZhipuAI-GLM-5.3-Flash-ATLAS_800_A2_376T_64G-128-a1b2c3d4"
  }
}
```

### 6.2 查询任务（核心）

`GET /api/v1/tasks/{task_id}`

```json
{
  "code": 200,
  "msg": "查询成功",
  "data": {
    "task_id": "20261020-143012-...-a1b2c3d4",
    "status": "RUNNING",
    "progress": {
      "done": 5, "total": 15, "percent": 33.3,
      "stage": "msmodeling/3P1D/disagg_prefill",
      "updated_at": "2026-10-20T14:33:01+08:00"
    },
    "created_at": "...", "started_at": "...", "finished_at": null,
    "elapsed_sec": 169.3,
    "request": { "model": "ZhipuAI/GLM-5.3-Flash", "num_devices": 128, "...": "..." },
    "routing": { "framework": "msmodeling", "reason": "..." },
    "result": null,
    "error": null,
    "artifacts": { "run_dir": "...", "run_json": "...", "summary": "..." }
  }
}
```

`SUCCESS` 时 `data.result` 内嵌完整 `UnifiedRunResult`（或置为 `null` 并引导去
`/result` 取全量，避免详情接口过大 —— 见 §12 待确认）。

`FAILED` 时：

```json
{
  "code": 200,
  "msg": "查询成功",
  "data": { "status": "FAILED", "error": "msmodeling 1P1D:disagg_decode device rejected ...",
            "result": null, "progress": { "...": "..." } }
}
```

> 注意：**查询到一个"执行失败"的任务，HTTP 层仍是 200**（查询这个动作成功了），
> 失败体现在 `data.status` / `data.error`。这与"code 表示接口调用结果"的口径一致。

### 6.3 取完整结果

`GET /api/v1/tasks/{task_id}/result` → `data` 直接是 `run.json` 的内容（`UnifiedRunResult`）。

### 6.4 取消

`POST /api/v1/tasks/{task_id}/cancel` → `202 {"code":202,"msg":"取消请求已接受"}`
终态任务再取消 → `409`。

### 6.5 列表

`GET /api/v1/tasks?status=RUNNING&model=Qwen/Qwen3.8-27B&page=1&page_size=20`

```json
{
  "code": 200, "msg": "查询成功",
  "data": { "total": 37, "page": 1, "page_size": 20,
            "items": [ { "task_id": "...", "status": "SUCCESS", "percent": 100.0, "created_at": "..." } ] }
}
```

---

## 7. 统一响应封装

所有接口统一：

```json
{ "code": 200, "msg": "查询成功", "data": { } }
```

| 字段 | 说明 |
|---|---|
| `code` | 业务码，**与 HTTP 状态码保持一致**（RESTful 语义完整） |
| `msg` | 中文提示，供前端直接展示 |
| `data` | 业务数据；失败时为 `null`（或保留部分上下文） |
| `request_id` | （可选）链路追踪 id，写入响应头 `X-Request-Id` |

业务码表：

| code | msg 示例 | 场景 |
|---|---|---|
| 200 | 查询成功 | 查询/列表/删除成功 |
| 202 | 任务已创建 / 取消请求已接受 | 异步受理 |
| 400 | 参数错误：`--model` 不在白名单 | 校验失败（含字段级 detail） |
| 401 | 未授权 | 鉴权开启但 token 缺失/错误 |
| 404 | 任务不存在 | task_id 未找到 |
| 409 | 任务已结束，无法取消 | 状态冲突 |
| 422 | 参数校验不通过 | pydantic 校验失败（也可并入 400） |
| 429 | 队列已满，请稍后重试 | 超过 `max_queued_tasks` |
| 500 | 服务内部错误 | 未捕获异常 |
| 503 | 服务未就绪 / 依赖不可用 | msmodeling 未安装等 |

错误响应示例（`data` 带字段级明细）：

```json
{
  "code": 400,
  "msg": "参数错误",
  "data": { "errors": [ { "field": "model", "reason": "unsupported --model 'Foo/Bar'" } ] }
}
```

> **待确认**：是否需要"HTTP 恒为 200、仅用 body.code 表达错误"的口径（部分前端框架更习惯）。
> 本文默认采用"HTTP 与 code 一致"，因为更符合 RESTful。

---

## 8. 并发、队列、超时、取消

| 项 | 设计 | 配置 |
|---|---|---|
| 并发上限 | 串行或极小并发（搜索本身吃满 CPU/NPU） | `INFER_STRATEGY_API_MAX_CONCURRENT`（默认 1） |
| 排队 | 超出并发的任务进 FIFO 队列（`PENDING`） | `INFER_STRATEGY_API_MAX_QUEUED`（默认 32，超出返回 429） |
| 超时 | 子进程超过时限 → SIGTERM → SIGKILL，标记 `TIMEOUT` | `INFER_STRATEGY_API_TASK_TIMEOUT_SEC`（默认 7200） |
| 取消 | 终端任务直接标记；运行中对子进程发 SIGTERM | — |
| 幂等 | 可选 `client_token`，同 token 直接返回已存在的任务 | 可选 |

心跳：API 主进程定期 `poll()` 子进程句柄，回收僵尸并更新状态（也可在查询时惰性回收）。

---

## 9. 持久化与重启恢复

- 任务元数据随产物落盘：`<run_dir>/task.json`、`progress.json`、`run.json`
- 服务启动时扫描 `out_dir`，用 `task.json` 重建内存注册表
- 重启时仍处于 `RUNNING`/`PENDING` 的任务 → 标记为 `FAILED`，
  `error = "service restarted"`，避免"幽灵运行中"任务
- 不引入数据库，保持零运维；如后续需要多实例，再引入 Redis/DB

---

## 10. 部署与配置

新增可选依赖与启动脚本：

```toml
[project.optional-dependencies]
api = ["fastapi>=0.110", "uvicorn>=0.27", "pydantic>=2.5"]

[project.scripts]
infer-strategy-npu-server = "infer_strategy_npu.api.app:main"
```

启动：

```bash
pip install -e .          # fastapi/uvicorn/pydantic 已是主依赖，无需 [api] extra
infer-strategy-npu-server --host 0.0.0.0 --port 8000 \
  --out-dir /workspace/results/infer-strategy-npu \
  --max-concurrent 1 --max-queued 32 --task-timeout 7200
# 或 uvicorn infer_strategy_npu.api.app:app --host 0.0.0.0 --port 8000
# 交互式文档：http://<host>:8000/docs
```

落地位置：`src/infer_strategy_npu/api/`（`app.py` 路由 / `schemas.py` 请求模型与响应封装 /
`tasks.py` 任务管理与子进程 worker）。CLI 路径不导入该包，未装 `.[api]` 也不影响命令行。

环境变量：

| 变量 | 说明 | 默认 |
|---|---|---|
| `INFER_STRATEGY_API_OUT_DIR` | 产物根目录（复用 CLI 的 out-dir） | `/workspace/results/infer-strategy-npu` |
| `INFER_STRATEGY_API_MAX_CONCURRENT` | 并发任务数 | `1` |
| `INFER_STRATEGY_API_MAX_QUEUED` | 队列长度 | `32` |
| `INFER_STRATEGY_API_TASK_TIMEOUT_SEC` | 单任务超时 | `7200` |
| `INFER_STRATEGY_API_CORS_ORIGINS` | 允许的跨域来源（逗号分隔） | `*`（前端 demo 期全放开） |
| `INFER_STRATEGY_LOCAL_MODEL_ROOT` | （已有）msmodeling 本地模型根目录 | `model_configs/msmodeling` |
| `INFER_STRATEGY_PD_BALANCE_GRANULARITY` | （已有）xPyD 卡数粒度 | `16` |

> 鉴权按评审结论**不做**；如需接入网关，再加 `INFER_STRATEGY_API_KEY`。

（Docker 部署：在统一镜像里加 `EXPOSE 8000`，CMD 改为启动 server；msmodeling 需要 NPU → 服务必须跑在 NPU 主机上。）

---

## 11. 安全与运维

- **鉴权**：可选 `X-API-Key` 头；未配置 `INFER_STRATEGY_API_KEY` 时不校验（内网场景）
- **只读接口 vs 写接口**：创建/取消/删除为写操作，建议开启鉴权；查询可放宽
- **产物清理**：可选 TTL（`INFER_STRATEGY_API_RETENTION_DAYS`）定期清理旧任务目录；
  或只提供 `DELETE` 由调用方管理
- **日志**：API 访问日志走 uvicorn；任务自身日志仍在 `<run_dir>/logs/*.log`
- **可观测**：`/api/v1/health` 返回 `{status, version, running, queued, out_dir}`；
  可选 `/metrics`（Prometheus）
- **OpenAPI**：FastAPI 自带 `/docs`、`/openapi.json`

---

## 12. 评审结论（已确认）

| # | 议题 | 结论 |
|---|---|---|
| 1 | HTTP 状态码口径 | **HTTP 状态码与 body `code` 一致** |
| 2 | 结果是否内嵌 | **内嵌在 `GET /tasks/{id}` 的 `data.result`**；`/result` 仍保留为全量入口 |
| 3 | 并发数 | **暂不深究**，先用默认串行/小并发，后续再优化 |
| 4 | 回调通知 | **不做** webhook，客户端轮询 |
| 5 | 鉴权 | **不做** |
| 6 | dry-run | **暂不做** `POST /tasks/preview` |
| 7 | 实施节奏 | **分 4 阶段**（P1 领域层 → P2 任务层 → P3 API 层 → P4 运维） |

**前置任务（优先于 P3）**：aic-npu 与 msmodeling 的返回结果需要先对齐成一套同构模型，
见 `docs/plans/INFER_STRATEGY_NPU_UNIFIED_RESULT_MODEL.md`。
`data.result` 的结构以该文档为准（分层 `core` / `extended` / `raw`，统一返回全量，折叠交给前端）。

## 12.1 仍需补充的点

1. **进度粒度**：子任务级是否够？是否要"当前候选数/总数"更细的进度
2. **任务归属/多租户**：是否需要 `owner` 字段与隔离（当前方案无租户概念）
3. **保留期**：任务产物是否自动清理、保留多久
4. **取消语义**：已完成的子任务结果是否保留（当前设计：保留已落盘产物，标记 CANCELLED）
5. 结果太大时的裁剪开关：`?view=summary`、`?include_raw=false` 是否需要

---

## 13. 实施计划（建议分阶段）

| 阶段 | 内容 | 产出 |
|---|---|---|
| P1 | 领域层改造：`RunArtifacts` 支持外部 `run_id`；`recommend()` 支持 `on_progress`；runner 上报子任务进度 | 仍是 CLI，行为不变 |
| P2 | 任务层：`TaskManager` + 子进程 worker + `task.json`/`progress.json` | 可用 Python 直接驱动，无 HTTP |
| P3 | API 层：FastAPI 路由 + 统一响应封装 + 参数校验 | `/docs` 可调试 |
| P4 | 运维：鉴权、健康检查、清理、Dockerfile/启动脚本、文档 | 可交付 |

P1 是唯一会碰到现有 CLI 代码的阶段，且改动都是"加可选参数"，不影响命令行行为。
