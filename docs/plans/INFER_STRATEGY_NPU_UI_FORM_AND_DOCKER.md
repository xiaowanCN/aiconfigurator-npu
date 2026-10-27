# UI 表单提交 + 容器内交互：改动方案

> 状态：**UI 部分已实现**（表单 + FastAPI 托管静态界面）；**Docker 部分待评审**（涉及镜像入口变更，需你拍板）

---

## 0. 问题根源

之前 UI 只能用 `python -m http.server` 静态打开，因为：

1. 浏览器 `file://` 直接打开 → CORS 拒绝读取本地 `results/index.json`；
2. 静态模式下 `/api/v1` 不存在 → 表单无从提交。

**解法**：让 FastAPI 自己托管 `ui/`。这样 UI 与 API **同源**，两个问题一起消失，也不再需要 `python -m http.server`。

---

## 1. UI 部分（已实现）

### 1.1 `api/app.py`：托管静态界面

```python
UI_DIR = Path(__file__).resolve().parents[3] / "ui"   # src/infer_strategy_npu/api/app.py -> <repo>/ui
if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
```

- 挂载位置在**所有 API 路由之后**，因此不会吞掉 `/api/v1/*`（已验证）。
- `ui/` 不存在时（wheel 安装场景）自动跳过。
- `GET /api/v1/health` 新增 `ui: true|false` 便于排查。

访问方式：

```bash
infer-strategy-npu-server --host 0.0.0.0 --port 8000
# 浏览器打开 http://<host>:8000/
```

### 1.2 `ui/index.html`：真实表单

基础区（8 项）：模型 / 设备 / 卡数 / 量化 / 输入长度 / 输出长度 / TTFT 上限 / TPOT 上限

模式选择：`PD 混部（agg）` + `PD 分离（disagg）` 双勾选框，默认全选 → 对应 `serving_modes`

高级区（`<details>` 折叠）：

| 字段 | 对应 API 字段 |
|---|---|
| PD 每实例卡数（空格分隔，默认 `8`） | `pd_ratio_sizes` |
| TP 候选（留空 = 自动推导） | `tp_sizes` |
| 并发度 jobs | `jobs` |
| top-n（aic-npu） | `top_n` |
| 专家线性层量化动作 | `quantize_linear_action` |
| KV cache 量化动作 | `quantize_attention_action` |
| 开启图编译 | `compile_model` |

按钮：`预览命令` / `提交并运行` / `关闭`

### 1.3 `ui/app.js`：提交 + 轮询

| 函数 | 行为 |
|---|---|
| `isApiMode()` | **自动探测** `GET /api/v1/health`；可用 `?source=api` / `?source=static` 强制 |
| `fillFormOptions()` | 用 `/api/v1/models` + `/api/v1/devices` 填充下拉；静态模式下禁用提交并给出启动提示 |
| `buildPayload()` | 组装请求体，空值剔除（让服务端走默认） |
| `validatePayload()` | 前端预校验（模型/设备/卡数/形态/正整数） |
| `previewNewTask()` | `POST /tasks/preview` → 展示 PD 组合、`warnings`、待下发命令 |
| `submitNewTask()` | `POST /tasks` → 拿 `task_id` |
| `pollAndOpen()` | 每 5s 轮询 `/tasks/{id}`，显示 `进度 label(kind) done/total`；终态时刷新列表并打开详情 |

静态模式下的错误提示已改为明确的两种解法（启动服务 / 用 http.server）。

### 1.4 验证结果

```
GET /            -> 200  含「新建任务」
GET /app.js      -> 200  含 pdCandidatesTable
GET /styles.css  -> 200  含 nf-grid
GET /api/v1/health|models|devices -> 200 application/json（未被静态挂载吞掉）
POST /api/v1/tasks/preview        -> 200  32卡 sizes=8 16 -> 4 条 PD 命令
缺 model                          -> 422
```

---

## 2. Docker 部分（待评审）

`docker/Dockerfile.unified` 是 CLI 镜像（`CMD ["/bin/bash"]`），当前**没有** `ui/`，也没有暴露端口。

### 2.1 必须做的改动

| # | 改动 | 说明 |
|---|---|---|
| D1 | `COPY ui/ ./ui/` | 当前遗漏，容器内根本没有界面文件 |
| D2 | `EXPOSE 8000` | 声明端口 |
| D3 | 设置 `INFER_STRATEGY_API_OUT_DIR` | 让服务端产物目录与 CLI 默认一致：`/workspace/results/infer-strategy-npu` |
| D4 | 启动命令 | 见 §2.2（**需要你拍板**） |
| D5 | 结果持久化 | `docker run -v /data/results:/workspace/results` |

`fastapi` / `uvicorn` / `pydantic` 已在 `pyproject.toml` 的**主依赖**里（不是 optional），镜像 `pip install -e .` 后即具备，无需新增依赖。

### 2.2 D4 三种入口方案（需拍板）

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A（推荐）** | 保留 `CMD ["/bin/bash"]`，文档给出 `docker run -it ... infer-strategy-npu:0.1.0 infer-strategy-npu-server --host 0.0.0.0 --port 8000` | 不破坏现有 CLI 用法 | 每次要手打命令 |
| B | `CMD` 直接起服务 + `EXPOSE 8000` | 开箱即用 | 破坏现有「进容器跑 CLI」的用法 |
| C | 新增 `docker/Dockerfile.server`，`FROM infer-strategy-npu:0.1.0` + `CMD` 起服务 | 两种用法互不干扰 | 多一个 Dockerfile |

### 2.3 运行示例（方案 A）

```bash
docker build -f docker/Dockerfile.unified -t infer-strategy-npu:0.1.0 .

docker run -it --rm \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data/results:/workspace/results \
  -p 8000:8000 \
  infer-strategy-npu:0.1.0 \
  infer-strategy-npu-server --host 0.0.0.0 --port 8000
```

浏览器打开 `http://<host>:8000/` → 填表提交 → 轮询 → 看结果。

### 2.4 容器内需要注意的几点

| 项 | 说明 |
|---|---|
| **NPU** | msmodeling 需要真实 NPU；必须 `--device /dev/davinci*` 挂载驱动，服务跑在 NPU 主机上 |
| **并发** | `INFER_STRATEGY_API_MAX_CONCURRENT` 默认 `1`；NPU 主机建议保持 1，避免多任务抢卡 |
| **超时** | `INFER_STRATEGY_API_TASK_TIMEOUT_SEC` 默认 7200s；9 组 PD-ratio 约 27 min，够用 |
| **CORS** | UI 由 FastAPI 同源托管，**不需要**跨域；仅当前端独立部署（如 vite 5173）时才配 `INFER_STRATEGY_API_CORS_ORIGINS` |
| **结果目录** | 服务端 `out_dir` 与 CLI 默认一致时，静态模式下 `results/infer-strategy-npu/index.json`（`scripts/build_results_index.py` 生成）与 API 列表可共存 |
| **重启恢复** | `TaskManager._recover()` 会重建注册表，但残留 `RUNNING/PENDING` 会被标为 `FAILED`；长任务中断不会自动续跑 |
| **产物体积** | 9 组 PD-ratio 的 `run.json` 约 900 KB；前端详情默认会拉全量，建议详情页请求加 `include_raw=false`（当前 UI 未加，可后续优化） |

### 2.5 建议的落地顺序

1. D1 + D2 + D3（无争议，直接改）
2. D4 按你选的方案
3. 验证：容器内 `curl http://localhost:8000/api/v1/health`，浏览器开 `/` 提交一个 1 组 PD-ratio 的小任务

---

## 3. 需要你拍板

| # | 问题 | 建议 |
|---|---|---|
| **Q1** | 容器入口用 A / B / C 哪种？ | **A**（不改 `CMD`，用 `docker run ... <cmd>` 起服务），不破坏现有 CLI 用法 |
| **Q2** | 是否现在就改 Dockerfile？ | D1/D2/D3 无争议可直接改；D4 等你选 |
| **Q3** | UI 详情是否默认请求 `include_raw=false`？ | 建议加（9 组时 900 KB → 约 200 KB），但会让「raw 明细」在 UI 上不可见 |
| **Q4** | 是否要「取消任务」按钮？ | API 已有 `POST /tasks/{id}/cancel`，UI 未接；长任务（27 min）建议接上 |
