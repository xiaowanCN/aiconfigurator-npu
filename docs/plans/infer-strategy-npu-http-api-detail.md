# infer-strategy-npu HTTP 接口详细文档

> 本文档在 `docs/plans/infer-strategy-npu-http-api.md`（现行版接口参考）基础上，**强化各接口的细节**，
> 重点补全「任务创建接口」的全部参数与候选值、「健康状态查询」「模型支持」「设备支持」「结果查询」
> 的请求/返回结构与字段含义，采用类似标准接口文档（请求参数表 + 响应参数表 + 示例 + 补充说明）的格式。
>
> 代码位置：`src/infer_strategy_npu/api/{app.py,tasks.py,schemas.py}`
> 交互文档：服务启动后访问 `/docs`（Swagger UI）或 `/openapi.json`。

---

## 1. 调用方式

### 1.1 统一响应体

所有 JSON 接口返回统一结构：

```json
{ "code": 200, "msg": "查询成功", "data": { } }
```

| 字段 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 业务响应码（即「错误代码」）：**正常响应统一为 `200`**；**错误响应为 `1000X`** 形式（见 1.3）。HTTP 传输层状态码恒为 `200`，业务结果由本字段承载 |
| `msg` | string | 响应信息，成功时为操作描述，失败时为错误原因（对应 `code` 的中文语义） |
| `data` | object | 响应结果主体；失败时一般为错误明细对象（如 `{"errors":[...]}`） |

### 1.2 公共请求/响应消息头

| 名称 | 描述 | 示例 |
|---|---|---|
| `Content-Type` | 指定请求/响应消息体的 MIME 类型 | `application/json` |

### 1.3 错误码

| HTTP 状态码 | 错误代码 | 描述 | 语义 |
|---|---|---|---|
| 200 | 200 | SUCCESS | 请求成功（查询成功 / 任务已创建 / 取消已接受 / 删除成功） |
| 200 | 10001 | INVALID_PARAMETER | 请求参数不正确（模型/设备不在白名单、`serving_modes`/`pd_ratio_sizes` 非法、未知字段 `extra="forbid"`） |
| 200 | 10002 | PERMISSION_ERROR | 没有操作权限（当前接口未鉴权，预留） |
| 200 | 10003 | RESOURCE_NOT_FOUND | 资源不存在（task_id 未注册、日志名不存在） |
| 200 | 10004 | INTERNAL_ERROR | 服务内部错误（未捕获异常 / 服务未就绪） |
| 200 | 10005 | CONFLICT | 任务状态冲突（对已结束任务调用 cancel、任务未结束却请求 report） |
| 200 | 10006 | QUEUE_FULL | 队列已满，请稍后重试 |

> 约定：**HTTP 传输层状态码恒为 `200`**，所有响应都通过 `code` 字段（即「错误代码」）区分业务结果——正常为 `200`，错误为 `1000X` 系列。响应体中的 `msg` 字段为对应语义的中文描述（如 `10001` 对应 `参数错误`）。
>
> 错误体（`HTTPException.detail` 原样返回，结构同统一响应体）：

```json
{ "code": 10001, "msg": "参数错误", "data": { "errors": [ { "field": "device", "reason": "unsupported device 'xxx'" } ] } }
```

---

## 2. 健康状态查询

### 2.1 功能介绍

查询服务存活状态，并附带当前运行/排队任务数、版本、结果目录等运行态信息。通常用于负载均衡探活或前端展示服务健康度。

### 2.2 URI

| 请求方法 | URI | 描述 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查（含队列与结果目录信息） |

### 2.3 请求消息

无请求参数。

### 2.4 响应消息

响应参数说明：

| 名称 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 响应码，成功为 `200` |
| `msg` | string | 响应信息，成功为「服务正常」 |
| `data` | object | 健康详情对象 |
| `data.status` | string | 服务状态，恒为 `ok`（探活成功即为 ok） |
| `data.version` | string | 服务版本号，如 `1.x.x` |
| `data.running` | integer | 当前处于 `RUNNING` 状态的任务数 |
| `data.queued` | integer | 当前处于 `PENDING`（排队中）的任务数 |
| `data.max_concurrent` | integer | 服务允许的最大并发任务数（每个任务一个 spawn 子进程） |
| `data.out_dir` | string | 结果根目录的绝对路径（CLI 用 `--out-dir`，环境变量 `INFER_STRATEGY_API_OUT_DIR`） |
| `data.ui` | boolean | 本服务是否托管了 `ui/` 静态界面；`true` 表示可直接从 `http://host:port/` 打开界面 |
| `data.results_url` | string | out_dir 的静态访问前缀（如 `/results`），前端据此读取 `index.json` / `run.json` / `report.html` |

### 2.5 响应参考示例

```json
{
  "code": 200,
  "msg": "服务正常",
  "data": {
    "status": "ok",
    "version": "1.1.0",
    "running": 1,
    "queued": 0,
    "max_concurrent": 1,
    "out_dir": "D:\\workspace\\results\\infer-strategy-npu",
    "ui": true,
    "results_url": "/results"
  }
}
```

### 2.6 补充说明

- `running` + `queued` 之和可辅助判断是否需要提交新任务或等待空闲；超出 `max_concurrent` 的任务会进入 `queued` 排队，超过 `max_queued` 时新建任务返回 `429`。
- `out_dir` 与启动参数 `--out-dir` 保持一致；若服务重启，挂在 out_dir 下的历史任务目录仍可被 `/results/{path}` 静态访问。

---

## 3. 模型支持接口

### 3.1 功能介绍

返回当前服务支持的模型白名单（ModelScope ID）。调用方在创建任务前应先调用本接口，获取合法的 `model` 取值。

### 3.2 URI

| 请求方法 | URI | 描述 |
|---|---|---|
| GET | `/api/v1/models` | 模型白名单查询 |

### 3.3 请求消息

无请求参数。

### 3.4 响应消息

响应参数说明：

| 名称 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 响应码，成功为 `200` |
| `msg` | string | 响应信息 |
| `data` | array[string] | 模型白名单列表（ModelScope ID）；大小写不敏感，部分模型另含别名 |

### 3.5 响应参考示例

```json
{
  "code": 200,
  "msg": "查询成功",
  "data": [
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen3.8-27B",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "ZhipuAI/GLM-5.3-Flash",
    "ZhipuAI/GLM-5.3",
    "ZhipuAI/GLM-5.2",
    "moonshotai/Kimi-K3",
    "MiniMax/MiniMax-M3"
  ]
}
```

### 3.6 补充说明

- `data` 为字符串数组，元素为 ModelScope ID，作为任务创建接口 `model` 字段的唯一合法取值。
- 部分模型有额外的别名（如 `Qwen/Qwen3.8-27B` 别名 `qwen3_8_27B`、`MiniMax/MiniMax-M3` 别名 `MiniMaxAI/MiniMax-M3`），传入别名也可被识别。
- 不同模型支持的底层框架不同：例如 `deepseek-ai/DeepSeek-R1` 同时支持 aic-npu 与 msmodeling，而 `ZhipuAI/GLM-5.3-Flash` 等仅支持 msmodeling。当 `framework=auto` 时由「模型 + 量化」自动路由到对应框架。

---

## 4. 设备支持接口

### 4.1 功能介绍

返回当前服务支持的设备清单（统一设备名）。设备名作为创建任务接口 `device` 字段的唯一合法取值，且需与模型、量化组合后满足显存/算力约束。

### 4.2 URI

| 请求方法 | URI | 描述 |
|---|---|---|
| GET | `/api/v1/devices` | 设备清单查询 |

### 4.3 请求消息

无请求参数。

### 4.4 响应消息

响应参数说明：

| 名称 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 响应码，成功为 `200` |
| `msg` | string | 响应信息 |
| `data` | array[string] | 设备名列表；命名取自 msmodeling `DeviceProfile`，为两套框架设备命名唯一真源 |

### 4.5 响应参考示例

```json
{
  "code": 200,
  "msg": "查询成功",
  "data": [
    "TEST_DEVICE",
    "ATLAS_800_A2_376T_64G",
    "ATLAS_800_A2_313T_64G",
    "ATLAS_800_A2_280T_64G",
    "ATLAS_800_A2_280T_64G_PCIE",
    "ATLAS_800_A2_280T_32G_PCIE",
    "ATLAS_800_A3_752T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE",
    "ATLAS_800_A3_560T_128G_DIE_ROCE",
    "ATLAS_350_425T_112G",
    "ATLAS_350_425T_84G",
    "ATLAS_850_486T_112G",
    "ATLAS_850_486T_128G",
    "ATLAS_850E_486T_96G",
    "ATLAS_850E_425T_96G",
    "ATLAS_850_486T_112G_ROCE",
    "ATLAS_850_486T_128G_ROCE",
    "ATLAS_850E_486T_96G_ROCE",
    "ATLAS_850E_425T_96G_ROCE",
    "ATLAS_850_486T_112G_FM16",
    "ATLAS_850_486T_128G_FM16",
    "ATLAS_850E_486T_96G_FM16",
    "ATLAS_850E_425T_96G_FM16",
    "ATLAS_950_486T_128G",
    "ATLAS_950_486T_96G",
    "ATLAS_950_486T_144G"
  ]
}
```

### 4.6 补充说明

- `device` 必须与设备清单中的字符串**完全一致**（区分大小写），例如 `ATLAS_950_486T_96G`，传入 `Atlas 800T A2` 等别名会返回 `400 unsupported device`。
- 设备名中数字含义示例：`ATLAS_950_486T_96G` = 950 系列 / 算力 486T / 显存 96G；`A2/A3/850/950` 为不同代次硬件。
- 仅 `ATLAS_800_A2_376T_64G` 在 aic-npu 框架下被支持（映射到 `--system ascend_910b`），其余设备统一走 msmodeling；当 `framework=auto` 且模型仅支持 msmodeling 时，自动路由到 msmodeling，与所选设备无关。

---

## 5. 任务创建接口

### 5.1 功能介绍

提交一次推理策略推荐任务（异步执行）。服务校验参数合法性后创建任务并立即返回成功响应（`code=200`，HTTP 传输层亦为 200），任务在独立子进程（spawn）中执行，调用方需按 `task_id` 轮询结果。提交前可先调用 `POST /api/v1/tasks/preview`（dry-run）确认将要执行的命令组合与 PD 组合数，避免误提交。

### 5.2 URI

| 请求方法 | URI | 描述 |
|---|---|---|
| POST | `/api/v1/tasks` | 创建推理策略推荐任务（异步） |

### 5.3 请求消息

请求参数说明（`extra="forbid"`：出现未知字段将被拒绝）：

| 参数名称 | 是否必选 | 参数类型 | 描述 / 候选值 |
|---|---|---|---|
| `model` | 是 | string | 模型白名单（见 `GET /api/v1/models`），如 `ZhipuAI/GLM-5.2`、`deepseek-ai/DeepSeek-R1` |
| `device` | 是 | string | 设备名（见 `GET /api/v1/devices`），如 `ATLAS_950_486T_96G` |
| `num_devices` | 是 | integer | 总 NPU 卡数，必须 > 0 |
| `input_length` | 否 | integer | 输入序列长度（token），默认 `1024`，> 0 |
| `output_length` | 否 | integer | 输出序列长度（token），默认 `1024`，> 0 |
| `ttft_limit` | 否 | integer | TTFT（首 token 时延）上限，单位 ms，默认 `1000`，> 0 |
| `tpot_limit` | 否 | integer | TPOT（每输出 token 时延）上限，单位 ms，默认 `50`，> 0 |
| `quantize` | 否 | string | 量化总开关，默认 `DISABLE`；候选：`DISABLE`（不量化）/ `ENABLE`（开启量化，具体动作由下方三个 action 参数决定） |
| `top_n` | 否 | integer | 每类部署形态返回的候选数，默认 `10`，> 0 |
| `framework` | 否 | string | 推理框架选择，默认 `auto`；候选：`auto`（模型+量化自动路由）/ `aic-npu` / `msmodeling` |
| `serving_modes` | 否 | list[string] | 要搜索的部署形态，默认 `["agg","disagg"]`；候选：`agg`（PD 聚合）、`disagg`（PD 分离）；可组合 |
| `pd_ratio_sizes` | 否 | list[int] | PD 分离每实例卡数枚举，默认 `[8]`；组合为 **笛卡尔积**（如 `[8,16]` → P8D8/P8D8… 共 4 组 P×D 组合）；每个值必须 > 0；`P+D > num_devices` 的组合会被服务端静默剔除 |
| `tp_sizes` | 否 | list[int] | TP（张量并行）候选；留空由 msmodeling 自适应决定 |
| `jobs` | 否 | integer | msmodeling 并发度（并行搜索进程数），> 0；缺省时由 msmodeling 取默认（约 8） |
| `out_dir` | 否 | string | 覆盖服务级结果目录，用于把本次产物写到指定路径 |
| `quantize_linear_action` | 否 | string | **专家线性层**量化动作；候选：`DISABLED`（默认）、`W8A16_STATIC`、`W8A8_STATIC`、`W4A8_STATIC`、`W8A16_DYNAMIC`、`W8A8_DYNAMIC`、`W4A8_DYNAMIC`、`FP8`、`MXFP4`；`quantize=ENABLE` 且未显式传入时默认 `W8A8_DYNAMIC` |
| `quantize_non_expert_linear_action` | 否 | string | **非专家线性层**（attention 投影 / dense MLP / shared expert 等）量化动作；候选同上（与专家线性层共用取值集）；`ENABLE` 未传时默认 `DISABLED` |
| `quantize_attention_action` | 否 | string | **KV cache** 量化动作；候选：`DISABLED`（默认）、`INT8`、`FP8`；`ENABLE` 未传时默认 `DISABLED` |
| `speculative_method` | 否 | string | 投机解码方法（msmodeling）；候选：`mtp` / `dflash` / `dspark` |
| `num_speculative_tokens` | 否 | integer | 投机 token 数 / 深度 n |
| `acceptance_length` | 否 | number | Decode 吞吐折算用的接受长度（浮点） |
| `num_draft_layers` | 否 | integer | 覆盖 draft 模型的 `num_hidden_layers` |
| `mtp_acceptance_rates` | 否 | list[float] | 指定 MTP token 的接受率列表，默认 `[0.9, 0.6, 0.4, 0.2]` |
| `max_search_combinations` | 否 | integer | 最大搜索空间数量上限 |
| `compile_model` | 否 | boolean | 是否开启图编译，更准但更慢，默认 `false` |
| `batch_range` | 否 | list[int] | msmodeling `--batch-range`：批大小搜索范围 `[min max]` 或 `[max]`；**不传（`null`）则完全不下发**，由 msmodeling 自适应 |
| `max_batched_tokens` | 否 | integer | msmodeling `--max-batched-tokens`：单 DP 副本 Prefill/混合步的 token 预算；不传则自适应（默认 `4*input_length`，OOM 降级 2x/1x） |
| `extra_msmodeling_args` | 否 | list[string] | 原样透传给 msmodeling 的额外 argv token（「逃生舱」，追加在命令末尾，优先级最高，可覆盖上面的结构化字段） |
| `client_token` | 否 | string | 可选幂等键（当前仅记录，不做去重） |

> **服务端只读字段（禁止出现在请求体）**：`pd_ratio_grid`、`pd_skipped` 由服务端根据 `num_devices` 与 `pd_ratio_sizes` 自动回填，出现在请求体会被当作未知字段拒绝。
>
> **msmodeling 专用透传字段**：`batch_range`、`max_batched_tokens` 仅在显式传入（`null` 不会下发）时追加对应参数；`extra_msmodeling_args` 始终原样追加到 msmodeling 命令末尾，最后出现者生效。三者为 `null` 时由 `to_request_dict()` 的 `exclude_none` 剔除，不会进入最终请求体。

### 5.4 请求参考示例

```json
{
  "model": "ZhipuAI/GLM-5.2",
  "device": "ATLAS_950_486T_96G",
  "num_devices": 32,
  "input_length": 4096,
  "output_length": 1024,
  "ttft_limit": 2000,
  "tpot_limit": 80,
  "quantize": "ENABLE",
  "serving_modes": ["agg", "disagg"],
  "pd_ratio_sizes": [8, 16],
  "quantize_linear_action": "MXFP4",
  "quantize_attention_action": "FP8",
  "batch_range": [4, 256],
  "max_batched_tokens": 4096,
  "extra_msmodeling_args": ["--pp-sizes", "8", "--prefix-cache-hit-rate", "0.5"]
}
```

### 5.5 响应消息

响应参数说明（注意 `data` 内 `pd_ratio_grid` / `pd_skipped` / `warnings` 为服务端回填的非阻断信息）：

| 名称 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 响应码，正常为 `200`（不再使用 202；HTTP 传输层亦统一为 200） |
| `msg` | string | 响应信息，成功为「任务已创建」 |
| `data` | object | 任务详情对象 |
| `data.task_id` | string | 任务唯一标识（格式 `{日期}-{时间}-{模型}-{设备}-{卡数}-{8位后缀}`，等于结果目录名） |
| `data.status` | string | 任务状态，创建后通常为 `RUNNING` 或 `PENDING`（排队中） |
| `data.created_at` | string | 任务创建时间（ISO 8601，含时区） |
| `data.started_at` | string | 任务开始执行时间；尚未开始时为 `null` |
| `data.finished_at` | string | 任务完成时间；未完成时为 `null` |
| `data.pid` | integer | 任务子进程 pid；未启动时可能为 null |
| `data.error` | string | 任务错误信息；无错误时为 `null` |
| `data.progress` | object | 任务进度对象（详见 5.7） |
| `data.request` | object | 回显的任务请求数据（同请求参数含义） |
| `data.result` | object | 任务结果数据；**任务完成前恒为 `null`**（run.json 尚未落盘） |
| `data.artifacts` | object | 结果产物元信息（路径与访问 URL） |
| `data.artifacts.run_dir` | string | 任务结果目录根路径 |
| `data.artifacts.run_json` | string | `run.json` 绝对路径 |
| `data.artifacts.summary` | string | `summary.txt` 绝对路径 |
| `data.artifacts.report` | string | `report.html` 绝对路径 |
| `data.artifacts.report_url` | string | `report.html` 的 HTTP 访问路径（`/api/v1/tasks/{task_id}/report`） |
| `data.pd_ratio_grid` | list[object] | 实际会执行的 PD-ratio 组合（已剔除 `P+D > num_devices`） |
| `data.pd_ratio_grid[].label` | string | PD 组合标签，如 `P8D8` |
| `data.pd_ratio_grid[].prefill_devices` | integer | 该组合每个 Prefill 实例的卡数 |
| `data.pd_ratio_grid[].decode_devices` | integer | 该组合每个 Decode 实例的卡数 |
| `data.pd_skipped` | list[object] | 被剔除的 PD-ratio 组合及原因 |
| `data.pd_skipped[].label` | string | 被剔除组合标签，如 `P16D16` |
| `data.pd_skipped[].prefill_devices` | integer | 被剔除组合 Prefill 卡数 |
| `data.pd_skipped[].decode_devices` | integer | 被剔除组合 Decode 卡数 |
| `data.pd_skipped[].reason` | string | 剔除原因，如 `P+D=32 > num_devices=16` |
| `data.warnings` | list[string] | 非阻断告警（不拒绝请求，仅提示，见 5.8） |

### 5.6 响应参考示例

```json
{
  "code": 200,
  "msg": "任务已创建",
  "data": {
    "task_id": "20261027-114626-ZhipuAI-GLM-5.2-ATLAS_950_486T_96G-32-9639c54a",
    "status": "RUNNING",
    "created_at": "2026-10-27T11:46:26+08:00",
    "started_at": "2026-10-27T11:46:26+08:00",
    "finished_at": null,
    "pid": 49752,
    "error": null,
    "progress": { "stage": null, "kind": null, "label": null, "done": 0, "total": 0, "percent": 0.0, "updated_at": null },
    "request": { "model": "ZhipuAI/GLM-5.2", "device": "ATLAS_950_486T_96G", "num_devices": 32, "...": "..." },
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
}
```

> 响应头同时返回 `Location: /api/v1/tasks/{task_id}`，便于调用方直接定位。

### 5.7 进度对象 `progress` 字段说明

| 字段 | 类型 | 描述 |
|---|---|---|
| `stage` | string | 原始阶段串，如 `agg` 或 `P8D8/pd_ratio`；未开始为 `null` |
| `kind` | string | 阶段类型：`agg` \| `pd_ratio`；未开始为 `null` |
| `label` | string | PD 组合标签，如 `P8D8`；agg 阶段为 `null` |
| `done` / `total` | integer | 已完成的子任务数 / 子任务总数 |
| `percent` | number | 完成百分比（0.0–100.0） |
| `updated_at` | string | 进度更新时间（ISO 8601）；未开始为 `null` |

### 5.8 非阻断告警 `warnings[]`

不拒绝请求，仅提示，常见情形：

1. PD 分离无可用组合（`num_devices` 装不下任何 `P+D`）
2. 组合数 ≥ 5：给出预估总耗时（单组实测约 177s）
3. 被剔除的组合（`P+D > num_devices`）
4. 每实例最多 8 卡且未开启量化（大模型通常放不下）
5. `aic-npu` 框架且 `serving_modes` 少于 2 个：说明「仅过滤输出，不减少计算」

### 5.9 补充说明

- **状态机**：`PENDING ──(有空闲并发)──> RUNNING ──> SUCCESS`，失败分支为 `FAILED` / `CANCELLED`（调用 cancel）/ `TIMEOUT`（超过 `task_timeout`）。
- **轮询建议**：每 5s 一次 `GET /api/v1/tasks/{task_id}`，直到 `status ∈ {SUCCESS, FAILED, CANCELLED, TIMEOUT}`。
- **`result` 判空**：任务完成前 `result` 恒为 `null`，调用方须先判空再读取 `result.frameworks`，否则会触发空指针。
- **dry-run 前置**：提交前可调用 `POST /api/v1/tasks/preview`，秒级返回 `framework` / `pd_ratio_grid` / `pd_skipped` / `warnings` / `commands` / `total_steps`，用于确认命令与组合数。

---

## 6. 结果查询接口

### 6.1 功能介绍

根据 `task_id` 查询任务的完整结果（`run.json`）。任务完成后返回裁剪后的完整结果结构；未完成时 `data` 为 `null`。可通过 Query 参数剔除体积最大的 `raw` 明细与重复的变体级候选，降低传输体积（尤其 PD 分离下可从约 397KB 降至约 60KB）。

### 6.2 URI

| 请求方法 | URI | 描述 |
|---|---|---|
| GET | `/api/v1/tasks/{task_id}/result` | 完整结果查询（run.json） |

### 6.3 请求消息

请求参数介绍：

| 参数名称 | 是否必选 | 参数类型 | 描述 |
|---|---|---|---|
| `task_id` | 是 | string | 路径参数，任务唯一标识（即结果目录名） |
| `include_raw` | 否 | boolean | Query 参数，默认 `true`；是否保留各候选的 `raw` 明细（每条 PD 候选挂 48 字段原始行，体积最大） |
| `include_variant_candidates` | 否 | boolean | Query 参数，默认 `true`；是否保留 `disagg_variants[].pd_ratio_candidates`（与 framework 级重复） |

### 6.4 响应消息

响应参数说明（顶层即 `run.json` 结构；任务未完成时整个 `data` 为 `null`）：

| 名称 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 响应码，成功为 `200` |
| `msg` | string | 响应信息 |
| `data` | object \| null | 完整结果对象；任务未完成时为 `null` |
| `data.schema_version` | string | 结果 schema 版本，当前 `1.1` |
| `data.run_id` | string | 运行标识，等于 task_id |
| `data.generated_at` | string | 结果生成时间（ISO 8601，秒级） |
| `data.status` | string | 运行态：`ok`（全部成功）/ `partial`（部分形态有结果）/ `failed`（无可用结果） |
| `data.request` | object | 入参回显（与任务创建请求体字段一致） |
| `data.routing` | object | 框架路由信息 |
| `data.routing.framework` | string | 实际执行的框架：`aic-npu` / `msmodeling` |
| `data.routing.reason` | string | 路由判定原因（如模型+量化为何选该框架） |
| `data.routing.commands` | list[string] | 实际下发的命令列表 |
| `data.routing.aic_model_path` | string \| null | aic-npu 侧模型配置路径（仅 aic-npu 路由时存在） |
| `data.routing.ms_model_source` | string \| null | msmodeling 侧模型来源（本地目录或 ModelScope ID） |
| `data.frameworks` | list[object] | 各框架运行结果数组（通常一项） |
| `data.frameworks[].framework` | string | 框架名 |
| `data.frameworks[].status` | string | 该框架运行态：`ok` / `partial` / `failed` |
| `data.frameworks[].commands` | list[string] | 该框架下发命令 |
| `data.frameworks[].agg` | object \| null | PD 聚合（Aggregated）结果块 |
| `data.frameworks[].disagg` | object \| null | PD 分离（Disaggregated）结果块 |
| `data.frameworks[].disagg_variants` | list[object] | PD-ratio 各组合的变体摘要 |
| `data.frameworks[].pd_ratio_candidates` | list[object] | PD 分离的全局候选列表（跨组合排序；体积大，受 `include_raw` 控制） |
| `data.frameworks[].message` | string \| null | 该框架的提示信息 |
| `data.recommendation` | object \| null | 最终推荐结论 |
| `data.recommendation.serving_mode` | string | 推荐部署形态：`agg` / `disagg` |
| `data.recommendation.mode` | string | 推荐模式细分 |
| `data.recommendation.framework` | string | 推荐框架 |
| `data.recommendation.best` | object | 推荐的最佳候选（含吞吐、时延、TP/PP 等配置） |
| `data.recommendation.reason` | string | 推荐理由 |
| `data.errors` | list[string] | 运行期错误/告警信息数组 |
| `data.artifacts` | object | 产物路径信息 |
| `data.artifacts.run_dir` | string | 结果目录 |
| `data.artifacts.run_json` | string | run.json 路径 |
| `data.artifacts.report_html` | string \| null | report.html 路径（如有） |

### 6.5 响应参考示例（节选）

```json
{
  "code": 200,
  "msg": "查询成功",
  "data": {
    "schema_version": "1.1",
    "run_id": "20261027-114626-ZhipuAI-GLM-5.2-ATLAS_950_486T_96G-32-9639c54a",
    "generated_at": "2026-10-27T11:55:12+08:00",
    "status": "ok",
    "request": { "model": "ZhipuAI/GLM-5.2", "device": "ATLAS_950_486T_96G", "num_devices": 32, "quantize": "ENABLE", "serving_modes": ["agg", "disagg"], "pd_ratio_sizes": [8, 16] },
    "routing": {
      "framework": "msmodeling",
      "reason": "model ZhipuAI/GLM-5.2 with quantize=ENABLE is served by msmodeling",
      "commands": ["python -m cli.inference.throughput_optimizer ..."],
      "aic_model_path": null,
      "ms_model_source": "model_configs/msmodeling/GLM5.2"
    },
    "frameworks": [
      {
        "framework": "msmodeling",
        "status": "ok",
        "commands": ["python -m cli.inference.throughput_optimizer ..."],
        "agg": { "available": true, "best": { "throughput_tokens_per_s": 23890.5, "tps_per_instance": 11945.2, "tp": 4, "pp": 8, "...": "..." }, "candidates": [ { "...": "..." } ] },
        "disagg": { "available": true, "best": { "decode_tps_per_instance": 11886.9, "throughput_tokens_per_s": 23747.25, "...": "..." } },
        "disagg_variants": [ { "label": "P8D8", "pd_ratio": { "...": "..." }, "best": { "...": "..." }, "prefill": { "...": "..." }, "decode": { "...": "..." }, "candidate_count": 10 } ],
        "pd_ratio_candidates": [ { "label": "P8D8", "decode_tps_per_instance": 11886.9, "throughput_tokens_per_s": 23747.25, "raw": { "...": "48 字段原始行" } } ],
        "message": null
      }
    ],
    "recommendation": {
      "serving_mode": "disagg",
      "mode": "pd_ratio",
      "framework": "msmodeling",
      "best": { "label": "P8D16", "throughput_tokens_per_s": 24120.0, "tp": 4, "pp": 8, "...": "..." },
      "reason": "disagg(P8D16) 在满足 TTFT/TPOT 约束下吞吐最高"
    },
    "errors": [],
    "artifacts": {
      "run_dir": "<out_dir>/<task_id>",
      "run_json": "<out_dir>/<task_id>/run.json",
      "report_html": "<out_dir>/<task_id>/report.html"
    }
  }
}
```

### 6.6 响应参考示例（任务未完成）

```json
{ "code": 200, "msg": "查询成功", "data": null }
```

### 6.7 补充说明

- **`data` 为 `null` 即任务未结束**：`run.json` 尚未落盘，调用方应继续轮询 `GET /api/v1/tasks/{task_id}`，不要直接读取 `result.frameworks`。
- **各返回部分含义**：
  - `routing`：说明本次为什么选 aic-npu 或 msmodeling，以及实际执行了哪些命令，便于排查与复现。
  - `frameworks[].agg` / `disagg`：分别对应 PD 聚合、PD 分离两种部署形态的实测结果；`available=false` 表示当前卡数/显存放不下该形态。
  - `frameworks[].pd_ratio_candidates`：PD 分离下跨所有 P×D 组合的全局候选，按指标排序；`raw` 为 48 字段原始行（体积大，可用 `include_raw=false` 剔除）。
  - `recommendation`：综合满足 SLA（TTFT/TPOT 约束）后的最佳推荐结论，含 `serving_mode`、`framework`、`best` 配置与 `reason`。
  - `errors`：非空表示存在运行期问题（如某形态显存放不下），需结合 `status=partial/failed` 解读。
- **相关配套接口**：`GET /api/v1/tasks/{task_id}`（任务详情，含进度与结果）、`GET /api/v1/tasks/{task_id}/pd`（PD 分离专用视图，跨组合排序候选 + 分组摘要）、`GET /api/v1/tasks/{task_id}/report`（`?download=1` 下载单文件 HTML 报告）、`GET /api/v1/tasks/{task_id}/summary`（人类可读摘要）。

---

## 7. 任务概览查询接口（overview）

### 7.1 功能介绍

`GET /api/v1/tasks/{task_id}` 的完整视图包含 `request`、`artifacts`、`pd_ratio_grid`、`pd_skipped`、`warnings` 以及未裁剪的 `result`，体积较大（PD 多组合下达数百 KB）。本接口只保留任务状态与结果中的必要字段，并把候选按 **Top-K** 凝练：

- `frameworks[]` 去掉 `commands`、`artifacts`，只留 `framework` / `status` / `message` / `agg` / `disagg`；
- `agg` 与 `disagg` 统一为 `{available, best, candidates}` 结构；
- `agg.candidates` 为扁平 Top-K 列表；`disagg.candidates` 把三处 PD 候选**整合为按组合标签分组**的字典（如 `{"P8D8": [...], "P8D16": [...]}`），各组各自 Top-K；
- 所有 Top-K 均按整体方案策略的 `throughput_tokens_per_s` **降序**选取。

### 7.2 URI

| 请求方法 | URI | 描述 |
|---|---|---|
| GET | `/api/v1/tasks/{task_id}/overview` | 任务概览视图（字段凝练 + Top-K 候选） |

### 7.3 请求消息

请求参数介绍：

| 参数名称 | 是否必选 | 参数类型 | 描述 |
|---|---|---|---|
| `task_id` | 是 | string | 路径参数，任务唯一标识 |
| `topk` | 否 | integer | 每组候选项保留条数，**默认 `1`**，范围 `1–200`；按候选的 `throughput_tokens_per_s` 降序取前 `topk` 个。`agg.candidates` 与 `disagg.candidates` 下每个 PD 组合标签**各自独立计数** |
| `include_raw` | 否 | boolean | 是否保留候选/`best` 里的 `raw` 明细（48 字段原始行），默认 `false`（概览默认剔除，进一步压缩体积） |

### 7.4 响应消息

响应参数说明：

| 名称 | 类型 | 描述 |
|---|---|---|
| `code` | integer | 响应码，正常为 `200`；任务不存在为 `10003` |
| `msg` | string | 响应信息 |
| `data` | object | 概览任务对象 |
| `data.task_id` | string | 任务唯一标识 |
| `data.status` | string | 任务状态（`PENDING`/`RUNNING`/`SUCCESS`/`FAILED`/`CANCELLED`/`TIMEOUT`） |
| `data.created_at` | string | 任务创建时间 |
| `data.started_at` | string | 任务开始时间；未开始为 `null` |
| `data.finished_at` | string | 任务完成时间；未完成为 `null` |
| `data.progress` | object | 任务进度（同 5.7） |
| `data.error` | string | 任务错误信息；无错误为 `null` |
| `data.pid` | integer | 任务子进程 pid；未运行为 `null` |
| `data.result` | object \| null | 凝练后的结果；任务未完成时恒为 `null` |
| `data.result.run_id` | string | 运行标识（等于 task_id） |
| `data.result.generated_at` | string | 结果生成时间 |
| `data.result.status` | string | 运行态：`ok` / `partial` / `failed`（与 `data.status` 的任务状态不同） |
| `data.result.frameworks` | list[object] | 框架运行结果（精简结构，见下） |
| `data.result.frameworks[].framework` | string | 框架名，如 `msmodeling` |
| `data.result.frameworks[].status` | string | 该框架运行态：`ok` / `partial` / `failed` |
| `data.result.frameworks[].message` | string \| null | 该框架的提示信息 |
| `data.result.frameworks[].agg` | object \| null | PD 聚合结果，固定为 `{available, best, candidates}`；该形态无结果时为 `null` |
| `data.result.frameworks[].agg.available` | boolean | 该形态是否有可用结果 |
| `data.result.frameworks[].agg.best` | object \| null | 该形态最优候选（完整字段，含 `throughput_tokens_per_s` / `ttft_ms` / `tpot_ms` 等） |
| `data.result.frameworks[].agg.candidates` | list[object] | 扁平 Top-K 候选列表，按 `throughput_tokens_per_s` 降序，长度 `min(topk, 原候选数)` |
| `data.result.frameworks[].disagg` | object \| null | PD 分离结果，固定为 `{available, best, candidates}`；该形态无结果时为 `null` |
| `data.result.frameworks[].disagg.available` | boolean | 该形态是否有可用结果 |
| `data.result.frameworks[].disagg.best` | object \| null | 全局最优 PD 候选 |
| `data.result.frameworks[].disagg.candidates` | object | **按 PD 组合标签分组**的候选字典，形如 `{"P8D8": [...], "P8D16": [...], "P16D8": [...], "P16D16": [...]}` |
| `data.result.frameworks[].disagg.candidates.<标签>` | list[object] | 某个 PD 组合（如 `P8D8`）自己的 Top-K 候选，按 `throughput_tokens_per_s` 降序；各标签**独立计数**；顺序按 `disagg_variants` 中的组合顺序 |
| `data.result.recommendation` | object \| null | 最终推荐结论（完整保留；`include_raw=false` 时剔除其中的 `raw`） |
| `data.result.errors` | list[string] | 运行期错误/告警 |

> **不下发的字段**：`data` 层的 `request`、`artifacts`、`pd_ratio_grid`、`pd_skipped`、`warnings`；`result` 层的 `schema_version`、`request`、`routing`、`artifacts`；`frameworks[]` 的 `commands`、`artifacts`、`disagg_variants`、framework 级 `pd_ratio_candidates`；`agg` / `disagg` 的 `source_files`、`label`、`mode`、`pd_ratio`、`prefill`、`decode`、`pd_ratio_candidates` 等冗余字段。
>
> `disagg.candidates` 由原 `framework 级 pd_ratio_candidates`（跨组合候选全集，按 `label` 分组）+ `disagg_variants[]`（决定标签与顺序、补齐缺组）+ `disagg.pd_ratio_candidates`（最优变体副本兜底）**三处整合**而来，不再重复下发。

### 7.5 响应参考示例（默认 `topk=1`）

> 取自真实任务 `20261029-124348-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-7b68a258`（PD 组合 `P8D8 / P8D16 / P16D8 / P16D16`）。候选对象字段较多，示例仅列关键字段。

```json
{
  "code": 200,
  "msg": "查询成功",
  "data": {
    "task_id": "20261029-124348-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-7b68a258",
    "status": "SUCCESS",
    "created_at": "2026-10-29T12:43:48+08:00",
    "started_at": "2026-10-29T12:43:48+08:00",
    "finished_at": "2026-10-29T12:57:17+08:00",
    "progress": { "stage": "P16D16/pd_ratio", "kind": "pd_ratio", "label": "P16D16", "done": 5, "total": 5, "percent": 100.0, "updated_at": "2026-10-29T12:57:17+08:00" },
    "error": null,
    "pid": null,
    "result": {
      "run_id": "20261029-124348-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-7b68a258",
      "generated_at": "2026-10-29T12:43:49+08:00",
      "status": "ok",
      "frameworks": [
        {
          "framework": "msmodeling",
          "status": "ok",
          "message": null,
          "agg": {
            "available": true,
            "best": { "throughput_tokens_per_s": 17612.893, "ttft_ms": 19726.988, "tpot_ms": 71.098, "...": "..." },
            "candidates": [
              { "throughput_tokens_per_s": 17612.893, "ttft_ms": 19726.988, "tpot_ms": 71.098, "...": "..." }
            ]
          },
          "disagg": {
            "available": true,
            "best": { "throughput_tokens_per_s": 23747.2548089199, "...": "..." },
            "candidates": {
              "P8D8":   [ { "label": "P8D8",   "throughput_tokens_per_s": 4865.95,  "...": "..." } ],
              "P8D16":  [ { "label": "P8D16",  "throughput_tokens_per_s": 23747.2548089199, "...": "..." } ],
              "P16D8":  [ { "label": "P16D8",  "throughput_tokens_per_s": 3243.97,  "...": "..." } ],
              "P16D16": [ { "label": "P16D16", "throughput_tokens_per_s": 21710.73118598652, "...": "..." } ]
            }
          }
        }
      ],
      "recommendation": { "serving_mode": "disagg", "mode": "pd_ratio", "framework": "msmodeling", "best": { "...": "..." }, "reason": "..." },
      "errors": []
    }
  }
}
```

### 7.6 补充说明

- **Top-K 选取口径**：所有候选先按「整体方案策略」的 `throughput_tokens_per_s` **降序**排序，再取前 `topk` 个；缺失 / 非数值 / 非有限的吞吐值排到末尾。该口径与 `best` 的选取一致（实测 `agg.best.throughput_tokens_per_s` 即 `agg.candidates` 中的最大值），因此 `topk=1` 时 `agg.candidates[0]` 与 `agg.best` 对应同一方案。
- **agg 与 disagg 对称**：两者都是 `{available, best, candidates}`；区别只在 `candidates` 形态——`agg` 是扁平列表，`disagg` 是按 PD 组合标签分组的字典（因为 PD 分离天然按 `P8D8` 等组合组织）。
- **三处 PD 候选整合**：主源为 framework 级 `pd_ratio_candidates`（跨组合候选全集，每条带 `label`，按 `label` 分组即可还原各组合）；`disagg_variants[]` 决定标签与输出顺序、并在缺组时补齐；`disagg.pd_ratio_candidates`（最优变体副本）作兜底。三者互不重复下发。
- **各组独立计数**：`P8D8`、`P8D16` 等每个标签各自的候选列表都取 `topk` 个，不是全局共用一个配额；`agg.candidates` 同理。
- **`raw` 明细**：默认剔除（每条候选 / `best` 挂着 48 字段原始行），可通过 `?include_raw=true` 保留。
- **体积效果**：以该真实任务（原 `run.json` 约 414KB）为例，概览视图在默认 `topk=1`、`include_raw=false` 下约 12KB（约为原文件的 3%）；`topk` 越大体积线性增长，调用方可按需调节。
- **不改动原接口**：`GET /api/v1/tasks/{task_id}` 与 `/result` 行为不变；本接口为新增旁路，返回前对结果做深拷贝，不会修改磁盘上的 `run.json`。

---

## 8. 典型调用流程

```bash
BASE=http://host:8000/api/v1

# 0) 探活 & 取白名单（可选）
curl -s $BASE/health
curl -s $BASE/models
curl -s $BASE/devices

# 1) 先看会跑什么（dry-run，不创建任务）
curl -s -X POST $BASE/tasks/preview -H 'Content-Type: application/json' -d @payload.json

# 2) 提交任务
TASK=$(curl -s -X POST $BASE/tasks -H 'Content-Type: application/json' -d @payload.json | jq -r .data.task_id)

# 3) 轮询直到终态
while :; do
  R=$(curl -s $BASE/tasks/$TASK)
  S=$(echo $R | jq -r .data.status)
  echo "$S $(echo $R | jq -c .data.progress)"
  case $S in SUCCESS|FAILED|CANCELLED|TIMEOUT) break;; esac
  sleep 5
done

# 4) 取结果（剔除体积最大的 raw 与重复变体）
curl -s "$BASE/tasks/$TASK/result?include_raw=false&include_variant_candidates=false"

# 5) 或直接下载静态 HTML 报告给三方平台归档
curl -s "$BASE/tasks/$TASK/report?download=1" -o report.html
```
