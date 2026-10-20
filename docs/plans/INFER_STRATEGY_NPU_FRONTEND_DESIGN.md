# 前端 Demo 设计

> 参考原型：`infer-strategy-ui(1).html`（侧边栏 + 任务列表 + 任务详情 + 新建弹窗）。
> 本设计保留原型的交互骨架，但**数据来源改为从 `results/` 读取**，并把"新建任务"改为后续接入 HTTP 接口时的占位。

## 1. 目标与边界

- **Demo 期**：静态页面直接读取 `results/infer-strategy-npu/*/run.json`，无需后端。
- **正式期**：把数据源从 `ResultsSource` 切换为 `ApiSource`（指向已实现的 `infer-strategy-npu-server`），UI 渲染逻辑不变。
- 不在本阶段实现真正的"提交任务"（需要后端任务编排），仅保留入口占位。

## 2. 目录结构

```
ui/
  index.html      页面骨架（侧边栏 / 列表视图 / 详情视图）
  styles.css      样式（沿用原型的深蓝侧栏 + 卡片风格，新增 tab / PD 侧 / 候选表）
  app.js          渲染逻辑 + 数据源抽象
scripts/
  build_results_index.py   扫描 results 生成 index.json（列表用）
results/
  infer-strategy-npu/
    index.json                  # 由脚本生成，列表视图读取
    <task_id>/run.json         # 详情视图按需读取
```

## 3. 数据源抽象（关键）

`app.js` 暴露统一接口，切换数据源只改一处：

```js
const DataSource = {
  listTasks() {},   // -> 任务摘要数组
  getTask(id) {},   // -> 单任务完整 run.json
};
```

| 视图 | `ResultsSource`（当前） | `ApiSource`（后续，`?source=api`） |
|---|---|---|
| 列表 | `GET results/infer-strategy-npu/index.json` | `GET /api/v1/tasks?page_size=200` |
| 详情 | `GET results/infer-strategy-npu/<id>/run.json` | `GET /api/v1/tasks/<id>` → `data.result` |

> 两者返回的"单任务结构"完全一致（都是 `run.json`），因此 `renderDetail` 不用改。

## 4. 页面与视图

### 视图一：任务列表
- 筛选：`模型`（下拉，自动从数据聚合）、`状态`、`run_id 关键字`。
- 表格列：`run_id`(截断) · `模型` · `设备×卡数` · `量化` · `框架` · `状态` · `创建时间` · `操作`。
- 行点击 → 进入详情。

### 视图二：任务详情
1. **任务信息**（meta-grid）：模型 / 设备 / 卡数 / 量化 / in-out 长度 / TTFT·TPOT 上限 / 框架 / 路由原因。
2. **推荐结论卡**（reco）：最佳部署模式（agg / disagg）、集群吞吐、单卡吞吐、TTFT、TPOT、并发、批大小、并行策略串。
3. **聚合模式（agg）卡** + 聚合候选表（可折叠）。
4. **PD 分离均衡搜索（xPyD）**：以 `disagg_variants`（1P1D / 2P1D / 3P1D …）作 tab，每个 tab 展示组合吞吐、Prefill 侧 / Decode 侧并行策略与各自卡数吞吐、候选表。
5. **执行命令**（可折叠）：`routing.commands`。
6. **错误信息**（失败时）：`errors[]`。

## 5. 字段映射（run.json → UI）

| UI 字段 | run.json 路径 |
|---|---|
| 模型/设备/卡数/量化/长度/上限 | `request.*` |
| 框架 + 路由原因 | `routing.framework` / `routing.reason` |
| 推荐结论 | `recommendation.best.{throughput_tokens_per_s, throughput_tokens_per_s_per_device, ttft_ms, tpot_ms, concurrency, batch_size, parallel_str}` |
| agg 最优 / 候选 | `frameworks[].agg.best` / `frameworks[].agg.candidates[]` |
| disagg 各 xPyD 组 | `frameworks[].disagg_variants[]`（`label`, `best`, `prefill`, `decode`, `candidates`） |
| 并行策略串 | `*.parallel_str`（已含 TP/PP/DP/EP/MOE-TP/MOE-DP/DFlash） |
| 执行命令 | `routing.commands[]` |

> 注意：`ttft_ms` 在 disagg 场景下为 `null`（Prefill 侧才有意义），UI 显示为 `—`。

## 6. 运行方式

```bash
# 在仓库根目录启动静态服务（浏览器 file:// 直接打开会因 CORS 无法 fetch 本地文件）
python -m http.server 8000
# 浏览器访问
open http://localhost:8000/ui/index.html

# 后续切到接口数据源（需先启动 infer-strategy-npu-server）
http://localhost:8000/ui/index.html?source=api
```

## 7. 后续接入接口时的改动清单

1. `app.js` 中 `useApi` 分支已实现，启动 server 后加 `?source=api` 即可验证。
2. `ApiSource.listTasks` 的字段映射（API 列表项是精简视图）可按需对齐 `index.json` 摘要。
3. 真正实现"新建任务"：把占位按钮接到 `POST /api/v1/tasks`，再轮询 `GET /api/v1/tasks/<id>`（进度走 `data.progress`）。
