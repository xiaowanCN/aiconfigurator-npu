# HTTP 接口配套 PD-ratio 重构：改动设计方案

> 状态：**已实现**（评审结论：Q1 硬删除 / Q2 只告警不阻断 / Q3 `include_raw` 默认 true /
> Q4 实现 preview / Q5 先做 UI D1–D4 / Q6 加进度字段）
> 前置：`INFER_STRATEGY_NPU_PD_RATIO_REDESIGN.md`（domain 层重构，已完成并通过 whl 真机验证）

## 0.1 实施结果

| 项 | 结果 |
|---|---|
| 请求体 | 旧 `pd_balance_*` 传入即 422；`pd_ratio_grid` / `pd_skipped` / `warnings` 随 202 返回 |
| `POST /tasks/preview` | 已实现；32 卡 + `sizes 8 16` → 5 条命令（1 agg + 4 PD） |
| `GET /tasks/{id}/pd` | 已实现；`limit` 与 `include_raw` 生效 |
| 裁剪 | 真实 run.json **248 KB → 96 KB**（`include_raw=false&include_variant_candidates=false`） |
| 进度 | `progress.json` 新增 `kind` / `label`，读侧有默认值兜底 |
| UI | D1–D4 已做；D5（表单 + preview 联动）按评审结论留作二期 |

---

## 0. 现状盘点

### 0.1 HTTP 层三个文件

| 文件 | 职责 | 本次是否需要改 |
|---|---|---|
| `api/schemas.py` | `TaskCreateRequest` + `envelope()` | **已改**（`serving_modes` / `pd_ratio_sizes`），还需补 A1/A2 |
| `api/app.py` | FastAPI 应用 + 全部路由 | **需改**（新增 2 条路由 + 裁剪参数） |
| `api/tasks.py` | TaskManager，spawn 子进程执行 | 小改（进度字段） |

### 0.2 现有路由清单

| Method | Path | 备注 |
|---|---|---|
| GET | `/api/v1/health` | |
| GET | `/api/v1/models` | |
| GET | `/api/v1/devices` | |
| POST | `/api/v1/tasks` | 创建（202 + Location） |
| GET | `/api/v1/tasks` | 列表（status/model 过滤 + 分页） |
| GET | `/api/v1/tasks/{id}` | 详情，**内嵌整个 run.json** |
| GET | `/api/v1/tasks/{id}/result` | **run.json 原样转发，无裁剪** |
| GET | `/api/v1/tasks/{id}/summary` | summary.txt 文本 |
| GET | `/api/v1/tasks/{id}/logs` | |
| GET | `/api/v1/tasks/{id}/logs/{name}` | |
| POST | `/api/v1/tasks/{id}/cancel` | |
| DELETE | `/api/v1/tasks/{id}` | |

> **设计文档里写了但从未实现的路由**：`POST /api/v1/tasks/preview`（dry-run 返回将要下发的命令）。
> PD-ratio 让"提交前先看会跑多少"变得很关键 —— 单组 177s，`8 16 32` 九组约 27 min。

### 0.3 实测暴露的体积问题（本次重构直接放大）

验证运行（GLM-5.2 / 32 卡 / `--pd-ratio-sizes 8 16`，4 组合）：

| 文件 | 大小 |
|---|---|
| **run.json** | **397 KB** |
| `pd_ratio/P16D16/pd_ratio_raw.csv` | 74 KB |
| `pd_ratio/all_candidates.json` | 45 KB |

根因是**同一批 32 条候选被存了两遍**，且每条还挂着一个 48 字段的 `raw`：

```
frameworks[].pd_ratio_candidates        : 32 条   （跨组合合并，全局排序）
frameworks[].disagg_variants[].pd_ratio_candidates : 10 + 7 + 7 + 8 = 32 条
→ 64 条 × (24 个业务字段 + 48 个 raw 字段)
```

按 9 组合外推，**run.json 约 900 KB**，而 `/tasks/{id}` 还把这个包又内嵌一次。

### 0.4 遗留（不在 HTTP 层，但同批清理）

| 位置 | 内容 |
|---|---|
| `ui/app.js:247` | 卡片标题硬编码 `PD 分离均衡搜索（xPyD）` |
| `ui/app.js:220-240` | 候选表列不含 `decode_tps_per_instance`，也不读 `pd_ratio_candidates` |
| `docs/plans/infer-strategy-npu-user-guide.md:95-111,198-201,350,357,393-394` | **整节仍在文档化 `--pd-balance-*`**（上一轮漏改） |
| `docs/plans/INFER_STRATEGY_NPU_FRONTEND_DESIGN.md:56,68` | `PD 分离均衡搜索（xPyD）` 术语 |
| `scripts/*.sh` | 注释与 `PD_SIZES` 变量语义过时（功能正确） |

---

## 1. 请求体：`TaskCreateRequest`

### A1 新增 `pd_ratio_grid` 预计算并回填到创建响应

`model_validator(mode="after")` 里复用 domain 层的 `pd_ratio_grid()`，把**实际会执行的组合**算出来，随 202 响应返回：

```python
@model_validator(mode="after")
def _resolve_pd_ratio_grid(self) -> "TaskCreateRequest":
    if "disagg" in self.serving_modes:
        self.pd_ratio_grid = [
            {"label": c.label, "prefill_devices": c.prefill_devices, "decode_devices": c.decode_devices}
            for c in pd_ratio_grid(self.num_devices, self.pd_ratio_sizes)
        ]
    else:
        self.pd_ratio_grid = []
    return self
```

POST `/api/v1/tasks` 的 202 响应新增两段：

```jsonc
{
  "code": 202,
  "data": {
    "task_id": "...",
    "progress": {...},
    "pd_ratio_grid": [                       // 新增
      {"label": "P8D8", "prefill_devices": 8, "decode_devices": 8},
      {"label": "P8D16", ...}
    ],
    "warnings": [                            // 新增
      "PD 分离将执行 9 组 PD-ratio 组合，单组实测约 177s，预计总耗时较长",
      "aic-npu 分支一次调用必然同时产出 agg/disagg，--serving-modes 仅过滤输出"
    ],
    "result": null
  }
}
```

**不做硬拒绝**（沿用评审结论 Q7："能满足就让自行搜索，可能没结果而已"）。
空网格时 `pd_ratio_grid: []` + `warnings: ["PD 分离：32 卡装不下任何 P+D 组合…"]`，调用方自行决定要不要删任务。

### A2 跨字段告警（不阻断）

创建时按路由结果与参数组合生成 `warnings[]`：

| 条件 | 告警文案 |
|---|---|
| PD 网格为空 | `PD 分离：--num-devices=N 装不下任何 P+D 组合（最小 P+D=2×min(sizes)）` |
| 组合数 ≥ 5 | `PD 分离将执行 N 组 PD-ratio，单组实测约 177s，预计总耗时 ~N×3min` |
| 路由到 aic-npu 且 `serving_modes` 只给一个 | `aic-npu 一次调用必然同时产出 agg/disagg，serving_modes 仅过滤输出，不减少计算` |
| 大模型 + 小每实例卡数 + 不量化 | `大模型在 N 卡/实例下不量化通常无法放下，建议 --quantize ENABLE` |

> 前 3 条无需执行即可判断；第 4 条需要路由结果，与 A1 同在 `create_task` 里算。

### A3 请求体字段最终形态

```jsonc
{
  "model": "ZhipuAI/GLM-5.2",
  "device": "ATLAS_950_486T_96G",
  "num_devices": 32,
  "input_length": 4096, "output_length": 1024,
  "tpot_limit": 80, "ttft_limit": 2000,
  "quantize": "ENABLE",
  "quantize_linear_action": "MXFP4",
  "quantize_attention_action": "FP8",
  "serving_modes": ["disagg"],        // 默认 ["agg","disagg"]，可选 agg / disagg
  "pd_ratio_sizes": [8],              // 默认 [8]
  "tp_sizes": null,                   // null = 不下发，由 msmodeling 自行推导
  "speculative_method": "dflash",
  "client_token": "optional"
}
```

旧字段 `pd_balance_search` / `pd_balance_max` / `pd_balance_sizes` 已删除；
因为 `model_config = ConfigDict(extra="forbid")`，**旧客户端会直接 422**，这是有意的破坏性变更（见 §7-Q1）。

---

## 2. 路由改动

### B1 新增 `POST /api/v1/tasks/preview`（补齐设计文档欠账）

同步返回，不创建任务：

```jsonc
{
  "code": 200,
  "data": {
    "framework": "msmodeling",
    "reason": "...",
    "serving_modes": ["disagg"],
    "pd_ratio_grid": [{"label": "P8D8", ...}, ...],   // 实际会跑的组
    "skipped": [{"label": "P16D16", "reason": "16+16 > 32"}],
    "commands": ["python -m cli.inference.throughput_optimizer ... --enable-optimize-prefill-decode-ratio ..."],
    "total_steps": 4,
    "warnings": [...]
  }
}
```

内部直接调 `routing.route(request)`（dry-run 路径，**不 import msmodeling**，秒级返回）。
用途：前端提交前先展示"会跑几组、命令长什么样"，避免盲等 27 分钟。

### B2 新增 `GET /api/v1/tasks/{id}/pd`（PD 分离专用视图）

不用让调用方在 397 KB 的 run.json 里自己翻：

```jsonc
{
  "code": 200,
  "data": {
    "available": true,
    "serving_modes": ["disagg"],
    "pd_ratio_sizes": [8, 16],
    "variants": [                       // 每组一条摘要
      {"label": "P8D8", "pd_ratio": {"prefill_devices": 8, "decode_devices": 8},
       "best": {...}, "candidate_count": 10}
    ],
    "candidates": [ ... 32 条，已按 decode_tps_per_instance 降序 ... ],
    "best": {...},                      // 集群吞吐最优的那组
    "skipped": [...],
    "message": null                     // framework.message（含无结果时的可操作提示）
  }
}
```

支持 `?limit=N` 只取 Top-N 候选，`?include_raw=false` 去掉 48 字段 raw。

### B3 `/result` 与 `/tasks/{id}` 增加裁剪开关

```
GET /api/v1/tasks/{id}/result?include_raw=false&include_variant_candidates=false
```

| 参数 | 默认 | 效果 |
|---|---|---|
| `include_raw` | `true` | 去掉各候选的 `raw`（48 字段/条） |
| `include_variant_candidates` | `true` | 去掉 `disagg_variants[].pd_ratio_candidates`（与 framework 级重复的那 32 条） |

两个都关 → 397 KB 降到约 **60 KB**（实测外推，主要是 raw 与重复候选）。

### B4 路由最终清单（新增 2 条）

```
POST   /api/v1/tasks/preview          # 新增
GET    /api/v1/tasks/{id}/pd          # 新增
```

---

## 3. 进度上报

`progress.json` 现状：`{stage, done, total, percent, updated_at}`。
PD-ratio 下 stage 形如 `"P8D8/pd_ratio"`，调用方无法区分阶段类型。

**C1 增加两个可选字段**（纯增量，旧调用方不受影响）：

```jsonc
{
  "stage": "P8D8/pd_ratio",
  "kind": "pd_ratio",              // "agg" | "pd_ratio"
  "label": "P8D8",
  "done": 2, "total": 5, "percent": 40.0,
  "updated_at": "..."
}
```

`total` 在 PD-ratio 下 = `(1 if 含 agg else 0) + len(grid)`，保持不变。

---

## 4. UI 配套（`ui/app.js`）

| # | 位置 | 改动 |
|---|---|---|
| D1 | `:247` | 标题 `PD 分离均衡搜索（xPyD）` → **`PD 分离（PD-ratio 每实例卡数组合）`** |
| D2 | `:220-240` 候选表 | 补两列：**`单 D 实例 decode 吞吐`**、**`PD 组合`**（取 `decode_tps_per_instance` / `label`） |
| D3 | 新增卡片 | 读 `frameworks[].pd_ratio_candidates`，渲染**全局排序候选表**（P×D / P实例数 / D实例数 / 每实例卡数 / decode tps / 集群 tokens/s / req/s / 用卡） |
| D4 | `variantBody` | P/D 双栏补 **`每实例卡数`** 与 **`实例数`**（来源 `extra.devices_per_instance` / `extra.instances`），并显示 `devices_used/num_devices` |
| D5 | 提交表单 | 新增 `serving_modes` 多选、`pd_ratio_sizes` 输入；提交前调 `/tasks/preview` 展示组合数 |

> D1–D4 只改渲染层：UI 的 api 模式（`j.data.result`）与静态模式（直接 fetch run.json）拿到的结构一致，改一次两边都生效。

---

## 5. 文档清理

| 文件 | 改动 |
|---|---|
| `infer-strategy-npu-user-guide.md` | **补上漏改的 `--pd-balance-*` 整节**（95-111 / 198-201 / 350 / 357 / 393-394） |
| `INFER_STRATEGY_NPU_HTTP_API_DESIGN.md` | 请求体样例（已改）+ 路由表补 preview/pd + 进度字段 |
| `INFER_STRATEGY_NPU_FRONTEND_DESIGN.md` | xPyD 术语 → PD-ratio |
| `INFER_STRATEGY_NPU_TEST_CASES.md` | 258-259 行 xPyD 说明 |
| `scripts/*.sh`（4 个） | 注释与 `PD_SIZES` 语义说明（功能已正确） |

---

## 6. 改动计划（评审通过后执行）

| # | 文件 | 内容 |
|---|---|---|
| 1 | `api/schemas.py` | A1 model_validator 计算 grid；`pd_ratio_grid` 字段；warnings 生成函数 |
| 2 | `api/app.py` | B1 preview 路由、B2 pd 路由、B3 裁剪参数；create 响应回填 grid + warnings |
| 3 | `api/tasks.py` | C1 progress.json 增加 `kind`/`label` |
| 4 | `ui/app.js` | D1–D5 |
| 5 | `docs/plans`（5 个文件） | §5 文档清理 |

`scripts/build_results_index.py` **无需改动**（不读 PD 字段）。

---

## 7. 需要你拍板

### Q1. 旧请求体字段是硬删除还是兼容？
`extra="forbid"` + 已删 `pd_balance_*` ⇒ 旧客户端 `POST /tasks` 直接 **422**。
- **建议：硬删除**（这些字段语义已完全失效，留着只会误导）。若你有在跑的调用方，我可改为先接受并忽略、打 warning。

### Q2. PD 网格为空 / 过大时，是拒绝还是只告警？
- **建议：只告警 + 回填 `pd_ratio_grid`**（沿用 Q7 结论，不阻断）。
- 备选：网格为空时 400；组合数 > 6 时 400。

### Q3. `/result` 的 `include_raw` 默认值？
- **建议：默认 `true`（保持现有行为），调用方按需传 `false`**。
- 备选：默认 `false`（新 API 更轻量，但会让现有取 raw 的调用方拿不到）。

### Q4. 是否实现 `POST /tasks/preview`？
- **建议实现**（设计文档欠账，且 PD-ratio 下价值最大：27 分钟的任务不该盲提交）。
- 它不需要 import msmodeling，秒级返回，风险低。

### Q5. UI 改到什么程度？
- D1+D2+D3+D4（渲染层，改动集中、风险低）——**建议全做**。
- D5（表单 + preview 联动）工作量明显更大，可拆成二期。**是否先做 D1–D4？**

### Q6. progress.json 加字段是否必要？
- 只加 `kind`/`label` 两个可选字段，向后兼容。**建议加**；若你希望保持接口稳定，也可不改（调用方可从 `stage` 字符串解析）。
