# infer-strategy-npu HTTP 接口文档（现行版）

> 面向调用方（三方平台 / 前端）的**接口参考**。设计背景与演进见
> `docs/plans/INFER_STRATEGY_NPU_HTTP_API_DESIGN.md`，本文件只描述"现在有什么、怎么调"。
>
> 代码位置：`src/infer_strategy_npu/api/{app.py,tasks.py,schemas.py}`
> 交互文档：服务启动后访问 `/docs`（Swagger UI）或 `/openapi.json`。

---

## 1. 启动与配置

```bash
pip install -e ".[api]"

infer-strategy-npu-server --host 0.0.0.0 --port 8000 \
    --out-dir D:/workspace/results/infer-strategy-npu \
    --max-concurrent 1 --max-queued 32 --task-timeout 7200

# 等价写法
uvicorn infer_strategy_npu.api.app:app --host 0.0.0.0 --port 8000
```

| 启动项 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| 结果目录 | `INFER_STRATEGY_API_OUT_DIR` | `/workspace/results/infer-strategy-npu` | 任务目录与产物的根；CLI 用 `--out-dir`，共用同一套命名 |
| 最大并发 | `INFER_STRATEGY_API_MAX_CONCURRENT` | `1` | 同时运行的任务数（每个任务一个 spawn 子进程） |
| 队列长度 | `INFER_STRATEGY_API_MAX_QUEUED` | `32` | 超出返回 `429` |
| 任务超时(s) | `INFER_STRATEGY_API_TASK_TIMEOUT_SEC` | `7200` | 超时置为 `TIMEOUT` |
| CORS 白名单 | `INFER_STRATEGY_API_CORS_ORIGINS` | `*` | 逗号分隔 |

说明：

- **模型路由对调用方透明**：`framework=auto` 时由模型 + 量化自动选 `aic-npu` / `msmodeling`。
- **任务在独立子进程执行**（spawn），崩溃隔离、可取消；服务重启会把遗留的非终态任务标成 `FAILED`。
- 静态界面 `ui/` 由服务自身挂载在 `/`（同源，无需 CORS）。

---

## 2. 统一响应体

所有 JSON 接口返回：

```json
{ "code": 200, "msg": "查询成功", "data": { } }
```

| code | 含义 | 触发场景 |
|---|---|---|
| 200 | 查询成功 | 所有 GET / preview |
| 202 | 任务已创建 / 请求已接受 | `POST /tasks`、`POST /tasks/{id}/cancel` |
| 400 | 参数错误 | 模型/设备不在白名单、`serving_modes`/`pd_ratio_sizes` 非法 |
| 404 | 任务不存在 | task_id 未注册、日志名不存在 |
| 409 | 任务状态冲突 | 对已结束任务调用 cancel |
| 429 | 队列已满 | 排队任务数超过 `max_queued` |

错误体（`HTTPException.detail` 原样返回）：

```json
{ "code": 400, "msg": "参数错误", "data": { "errors": [ { "field": "device", "reason": "unsupported device 'xxx'" } ] } }
```

---

## 3. 接口总表

| 方法 | 路径 | 说明 | 成功码 |
|---|---|---|---|
| GET | `/api/v1/health` | 健康检查（含队列与结果目录信息） | 200 |
| GET | `/api/v1/models` | 模型白名单 | 200 |
| GET | `/api/v1/devices` | 设备清单 | 200 |
| POST | `/api/v1/tasks/preview` | **dry-run**：不下发、不创建任务，返回将要执行的命令与 PD 组合 | 200 |
| POST | `/api/v1/tasks` | 创建任务（异步） | 202 |
| GET | `/api/v1/tasks` | 任务列表（分页 / 过滤） | 200 |
| GET | `/api/v1/tasks/{task_id}` | 任务详情：状态 + 进度 + 结果 | 200 |
| GET | `/api/v1/tasks/{task_id}/result` | 完整结果（`run.json`） | 200 |
| GET | `/api/v1/tasks/{task_id}/pd` | PD 分离专用视图（跨组合排序候选 + 分组摘要） | 200 |
| GET | `/api/v1/tasks/{task_id}/report` | **单文件静态 HTML 报告**（`?download=1` 附件下载） | 200 |
| GET | `/api/v1/tasks/{task_id}/summary` | 人类可读摘要（`summary.txt`） | 200 |
| GET | `/api/v1/tasks/{task_id}/logs` | 日志文件列表 | 200 |
| GET | `/api/v1/tasks/{task_id}/logs/{name}` | 日志内容（`?tail=200`） | 200 |
| POST | `/api/v1/tasks/{task_id}/cancel` | 取消任务 | 202 |
| DELETE | `/api/v1/tasks/{task_id}` | 删除任务记录与产物目录 | 200 |
| GET | `/results/{path}` | 结果目录只读静态文件（`index.json` / `run.json` / `report.html`） | 200 |
| GET | `/` | 静态界面（`ui/`，存在时挂载） | 200 |

---

## 4. 元数据

### 4.1 `GET /api/v1/health`

```json
{
  "code": 200, "msg": "服务正常",
  "data": {
    "status": "ok",
    "version": "1.x.x",
    "running": 1,
    "queued": 0,
    "max_concurrent": 1,
    "out_dir": "D:\\workspace\\results\\infer-strategy-npu",
    "ui": true,
    "results_url": "/results"
  }
}
```

- `ui=true` 表示本服务托管了 `ui/`，可直接从 `http://host:port/` 打开界面。
- `results_url` 是 out_dir 的静态访问前缀，前端据此读取 `index.json` / `run.json` / `report.html`。

### 4.2 `GET /api/v1/models`

返回 `["deepseek-ai/DeepSeek-R1", "ZhipuAI/GLM-5.2", ...]`（ModelScope ID 白名单）。

### 4.3 `GET /api/v1/devices`

返回设备名数组，如 `["ATLAS_800_A2_376T_64G", "ATLAS_950_486T_144G", ...]`。

---

## 5. 创建任务

### 5.1 请求体 `TaskCreateRequest`

`extra="forbid"`，未知字段会被拒绝。

| 字段 | 类型 | 默认 | 必填 | 说明 |
|---|---|---|---|---|
| `model` | string | — | ✅ | 模型白名单内（见 `/api/v1/models`） |
| `device` | string | — | ✅ | 设备名（见 `/api/v1/devices`） |
| `num_devices` | int | — | ✅ | 总卡数，> 0 |
| `input_length` | int | 1024 | | 输入长度 |
| `output_length` | int | 1024 | | 输出长度 |
| `ttft_limit` | int | 1000 | | TTFT 上限（ms） |
| `tpot_limit` | int | 50 | | TPOT 上限（ms） |
| `quantize` | string | `DISABLE` | | `DISABLE` / `ENABLE` |
| `top_n` | int | 10 | | 每类返回候选数 |
| `framework` | string | `auto` | | `auto` / `aic-npu` / `msmodeling` |
| `serving_modes` | string[] | `["agg","disagg"]` | | 可选 `agg`、`disagg`；**仅 msmodeling 侧可控**，aic-npu 一次调用必然同时产出两者 |
| `pd_ratio_sizes` | int[] | `[8]` | | PD 每实例卡数枚举；组合 = **笛卡尔积**（`8 16` → P8D8/P8D16/P16D8/P16D16 四组） |
| `tp_sizes` | int[] | `null` | | TP 候选，留空由 msmodeling 自行决定 |
| `jobs` | int | `null` | | msmodeling 并发度 |
| `out_dir` | string | `null` | | 覆盖服务级结果目录 |
| `quantize_linear_action` | string | `null` | | 专家线性层量化（如 `MXFP4`） |
| `quantize_non_expert_linear_action` | string | `null` | | 非专家线性层 |
| `quantize_attention_action` | string | `null` | | KV cache（如 `INT8` / `FP8`） |
| `speculative_method` / `num_speculative_tokens` / `acceptance_length` / `num_draft_layers` | — | `null` | | 投机解码（msmodeling） |
| `max_search_combinations` | int | `null` | | 搜索组合上限 |
| `compile_model` | bool | `false` | | 图编译，更准但更慢 |
| `client_token` | string | `null` | | 幂等键（当前仅记录，不去重） |

> `pd_ratio_grid` / `pd_skipped` 是服务端回填的只读字段（`exclude=True`），**只能在响应里读**，
> 出现在请求体会被当作未知字段拒绝。

### 5.2 `POST /api/v1/tasks`（创建）— 202

请求：

```bash
curl -X POST http://host:8000/api/v1/tasks -H 'Content-Type: application/json' -d '{
  "model": "ZhipuAI/GLM-5.2",
  "device": "ATLAS_950_486T_96G",
  "num_devices": 32,
  "quantize": "ENABLE",
  "input_length": 4096,
  "output_length": 1024,
  "ttft_limit": 2000,
  "tpot_limit": 80,
  "serving_modes": ["agg", "disagg"],
  "pd_ratio_sizes": [8, 16]
}'
```

响应 `data`：

```json
{
  "task_id": "20261027-114626-ZhipuAI-GLM-5.2-ATLAS_950_486T_96G-32-9639c54a",
  "status": "RUNNING",
  "created_at": "2026-10-27T11:46:26+08:00",
  "started_at": "2026-10-27T11:46:26+08:00",
  "finished_at": null,
  "pid": 49752,
  "error": null,
  "progress": { "stage": null, "kind": null, "label": null, "done": 0, "total": 0, "percent": 0.0, "updated_at": null },
  "request": { "model": "ZhipuAI/GLM-5.2", "...": "..." },
  "result": null,
  "artifacts": {
    "run_dir": "<out_dir>/<task_id>",
    "run_json": "<out_dir>/<task_id>/run.json",
    "summary": "<out_dir>/<task_id>/summary.txt",
    "report": "<out_dir>/<task_id>/report.html",
    "report_url": "/api/v1/tasks/<task_id>/report"
  },
  "pd_ratio_grid": [
    { "label": "P8D8",  "prefill_devices": 8,  "decode_devices": 8 },
    { "label": "P8D16", "prefill_devices": 8,  "decode_devices": 16 },
    { "label": "P16D8", "prefill_devices": 16, "decode_devices": 8 },
    { "label": "P16D16","prefill_devices": 16, "decode_devices": 16 }
  ],
  "pd_skipped": [],
  "warnings": ["PD 分离将执行 4 组 PD-ratio 组合，单组实测约 177s，预计总耗时约 12 分钟"]
}
```

响应头带 `Location: /api/v1/tasks/{task_id}`。

> `result` 在任务完成前恒为 `null`（`run.json` 尚未落盘），**调用方需按此判空**，不要直接取 `result.frameworks`。

### 5.3 `POST /api/v1/tasks/preview`（dry-run）— 200

不创建任务、不执行，秒级返回，用于提交前确认"会跑几组、命令长什么样"。

```json
{
  "code": 200,
  "data": {
    "framework": "msmodeling",
    "reason": "model ZhipuAI/GLM-5.2 with quantize=ENABLE is served by msmodeling ...",
    "pd_ratio_grid": [ { "label": "P8D8", "prefill_devices": 8, "decode_devices": 8 } ],
    "pd_skipped": [ { "label": "P16D16", "prefill_devices": 16, "decode_devices": 16, "reason": "P+D=32 > num_devices=16" } ],
    "warnings": [ "已剔除 P16D16（P+D 超出 num_devices）" ],
    "commands": [ "python -m cli.inference.throughput_optimizer ..." ],
    "total_steps": 4,
    "pd_combo_count": 1
  }
}
```

### 5.4 非阻断告警 `warnings[]`

不拒绝请求，仅提示：

1. PD 分离无可用组合（`num_devices` 装不下任何 P+D）
2. 组合数 ≥ 5：给出预估总耗时（单组实测约 177s）
3. 被剔除的组合（`P+D > num_devices`）
4. 每实例最多 8 卡且未开启量化（大模型通常放不下）
5. `aic-npu` 且 `serving_modes` 少于 2 个：说明"仅过滤输出，不减少计算"

---

## 6. 查询任务

### 6.1 `GET /api/v1/tasks`

| Query | 默认 | 说明 |
|---|---|---|
| `status` | — | 按状态过滤（`PENDING`/`RUNNING`/`SUCCESS`/`FAILED`/`CANCELLED`/`TIMEOUT`） |
| `model` | — | 按模型过滤 |
| `page` | 1 | ≥ 1 |
| `page_size` | 20 | 1–200 |

```json
{ "code": 200, "data": { "total": 33, "page": 1, "page_size": 20,
  "items": [ { "task_id": "...", "status": "SUCCESS", "created_at": "...", "started_at": "...", "finished_at": "...", "pid": null, "error": null, "progress": { } } ] } }
```

### 6.2 `GET /api/v1/tasks/{task_id}`

| Query | 默认 | 说明 |
|---|---|---|
| `include_raw` | `true` | 候选里的 `raw`（48 字段原始行）体积最大，PD-ratio 下可置 `false` |
| `include_variant_candidates` | `true` | `disagg_variants[].pd_ratio_candidates` 与 framework 级重复，可置 `false` |

返回结构与 5.2 相同；任务完成后 `result` 为完整 `run.json`，`artifacts.report_url` 指向静态报告。

### 6.3 状态机与进度

```
PENDING ──(有空闲并发)──> RUNNING ──> SUCCESS
                            │         FAILED
                            ├────────> CANCELLED   (POST /cancel)
                            └────────> TIMEOUT     (超过 task_timeout)
```

`progress` 字段（来自 `<task_id>/progress.json`）：

| 字段 | 说明 |
|---|---|
| `stage` | 原始阶段串，如 `agg` 或 `P8D8/pd_ratio` |
| `kind` | `agg` \| `pd_ratio` |
| `label` | PD 组合标签，如 `P8D8`（agg 阶段为 `null`） |
| `done` / `total` / `percent` | 进度 |
| `updated_at` | 更新时间 |

轮询建议：每 5s 一次 `GET /api/v1/tasks/{task_id}`，直到 `status ∈ {SUCCESS, FAILED, CANCELLED, TIMEOUT}`。
PD 分离每组实测约 177s，`--pd-ratio-sizes 8 16 32`（9 组）约 27 分钟。

### 6.4 `GET /api/v1/tasks/{task_id}/result`

返回裁剪后的 `run.json`（`include_raw` / `include_variant_candidates` 同上）。任务未完成时 `data` 为 `null`。

顶层结构：

```
run_id, generated_at, status, schema_version
request{}            # 入参
routing{}            # framework / reason / commands
frameworks[]         # agg / disagg / disagg_variants / pd_ratio_candidates
recommendation{}     # serving_mode / framework / best / reason
errors[], artifacts{}
```

### 6.5 `GET /api/v1/tasks/{task_id}/pd`

| Query | 默认 | 说明 |
|---|---|---|
| `limit` | — | 只返回 Top-N 候选 |
| `include_raw` | `true` | 是否保留 `raw` |

```json
{
  "available": true,
  "framework": "msmodeling",
  "serving_modes": ["disagg"],
  "pd_ratio_sizes": [8, 16],
  "num_devices": 32,
  "variants": [ { "label": "P8D8", "pd_ratio": {...}, "best": {...}, "prefill": {...}, "decode": {...}, "candidate_count": 10 } ],
  "candidates": [ { "label": "P8D8", "decode_tps_per_instance": 11886.9, "throughput_tokens_per_s": 23747.25, "...": "..." } ],
  "best": { },
  "message": null,
  "errors": []
}
```

### 6.6 `GET /api/v1/tasks/{task_id}/report`

返回**单文件静态 HTML**（CSS/JS/数据全内联，可直接分发、双击打开、离线查看）。

| Query | 默认 | 说明 |
|---|---|---|
| `download` | `false` | `true` 时以附件形式下载，文件名 `<task_id>-report.html` |

- 任务成功时已自动生成 `<run_dir>/report.html`；
- 文件缺失但 `run.json` 存在时**现场渲染**（兼容历史任务）；
- 内容与在线"任务详情"页同构（同一套 `ui/app.js` 渲染，仅数据内嵌）。

### 6.7 `GET /api/v1/tasks/{task_id}/summary`

```json
{ "code": 200, "data": { "text": "infer-strategy-npu - recommend summary\n===== ..." } }
```

### 6.8 日志

```bash
GET /api/v1/tasks/{task_id}/logs              # -> { "code":200, "data": ["msmodeling_P8D8_pd_ratio.log", ...] }
GET /api/v1/tasks/{task_id}/logs/{name}?tail=200  # -> { "code":200, "data": { "name": "...", "text": "..." } }
```

### 6.9 `POST /api/v1/tasks/{task_id}/cancel` — 202

对已结束任务调用返回 `409`。取消是"尽力而为"：终止子进程后状态置 `CANCELLED`。

### 6.10 `DELETE /api/v1/tasks/{task_id}` — 200

删除内存记录**并**删除 `<out_dir>/<task_id>` 整个产物目录（含 `run.json`、`report.html`、日志）。

---

## 7. 结果目录与静态资源

```
<out_dir>/
├── index.json                    # 可选，由 scripts/build_results_index.py 生成（静态列表用）
└── <task_id>/                    # 每次运行一个目录，task_id == run_id
    ├── run.json                  # 完整结果（/result 返回它）
    ├── request.json
    ├── summary.txt               # /summary
    ├── report.html               # /report（任务成功自动生成）
    ├── task.json                 # HTTP 任务状态
    ├── progress.json             # 进度
    ├── done.json                 # 终态
    ├── aic_npu/                  # aic-npu 侧产物
    ├── msmodeling/pd_ratio/<PxD>/
    └── logs/*.log                # /logs
```

- `GET /results/{path}` 以只读方式暴露 out_dir（禁止 `..` 穿越），例如
  `/results/<task_id>/run.json`、`/results/<task_id>/report.html`。
- 历史任务补报告：`python scripts/build_reports.py <结果根> [--force]`。
- CLI（`infer-strategy-npu --recommend --out-dir ...`）同样会产出 `run.json` + `report.html`，
  只是没有 `task.json` / `progress.json` / `done.json`。

---

## 8. 典型调用流程

```bash
BASE=http://host:8000/api/v1

# 1) 先看会跑什么
curl -s -X POST $BASE/tasks/preview -H 'Content-Type: application/json' -d @payload.json

# 2) 提交
TASK=$(curl -s -X POST $BASE/tasks -H 'Content-Type: application/json' -d @payload.json | jq -r .data.task_id)

# 3) 轮询
while :; do
  R=$(curl -s $BASE/tasks/$TASK)
  S=$(echo $R | jq -r .data.status)
  echo "$S $(echo $R | jq -c .data.progress)"
  case $S in SUCCESS|FAILED|CANCELLED|TIMEOUT) break;; esac
  sleep 5
done

# 4) 取结果（或直接转发静态报告给平台）
curl -s "$BASE/tasks/$TASK/result?include_raw=false&include_variant_candidates=false"
curl -s "$BASE/tasks/$TASK/report?download=1" -o report.html
```

---

## 9. 前端对接要点

| 场景 | 用法 |
|---|---|
| 服务托管界面 | 直接访问 `http://host:8000/`（同源，无 CORS 问题，可提交任务） |
| 独立部署前端 | 需放开 `INFER_STRATEGY_API_CORS_ORIGINS`，并把 `API_BASE` 指向服务地址 |
| 纯静态只读 | `python -m http.server` + `?source=static`；结果目录非默认时加 `?dir=<目录名>` |
| 强制数据源 | `?source=api` 或 `?source=static` |
| 三方平台只看结果 | 调 `GET /tasks/{id}/report` 拿 HTML，或 `?download=1` 归档 |

---

## 10. 常见错误排查

| 现象 | 原因 / 处理 |
|---|---|
| `400 unsupported device 'Atlas 800T A2'` | 设备名要用 `/api/v1/devices` 里的枚举值（如 `ATLAS_950_486T_96G`） |
| `400 unsupported serving mode` | 只能用 `agg` / `disagg` |
| `429 队列已满` | 调大 `--max-queued`，或等待空闲 |
| 任务 `SUCCESS` 但某形态无结果 | 查看 `result.errors[]`；常见是卡数/显存放不下，需开量化或增大 `pd_ratio_sizes` |
| PD 分离只有 1 组 | `pd_ratio_sizes` 只填了一个值；组合是笛卡尔积，填 `8 16` 得 4 组 |
| `result` 为 `null` | 任务未结束，`run.json` 还没落盘 |
| 报 404 但任务确实存在 | 服务的 `out_dir` 与产出目录不一致，启动时用 `--out-dir` 指定 |
