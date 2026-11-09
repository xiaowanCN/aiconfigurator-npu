# infer-strategy-npu-task-enter.py 扩展指南（新增模型方案 / 设备 / PD 组合）

> 适用对象：仓库根目录一次性任务入口 `infer-strategy-npu-task-enter.py`，及其离线单测 / 演示脚本。
> 契约依据：`docs/plans/INFER_STRATEGY_NPU_TASK_ENTER_DESIGN.md`（v3 定稿）与接口文档 4.1.2 / 4.1.3。
> 本文只说明「后续怎么扩展」，不改变现有契约；文中一律用符号名定位，不写易漂移的行号。

---

## 1. 先判断要加的是哪一类

| 要新增的内容 | 配置驱动 | 主要改动位置 | 是否要改内核 `src/infer_strategy_npu/*` |
| --- | --- | --- | --- |
| 新的「模型 + 量化方案」组合（`--model` 新白名单值，复用已注册 `model_id`） | 是 | 入口 `MODEL_PLANS` 加 1 项 | 不需要 |
| 新模型（全新 `model_id`） | 是 | 入口 `MODEL_PLANS` + 框架 `MODELS` + 模型配置目录 | 需要补注册表项 |
| 新设备代号 `--device` | 是 | `DEVICE_SPECS` 加 1 项 | 不需要 |
| 新的**对称** PD 组合（如 P64D64） | 是 | `PD_RATIO_SIZES` 加 1 个值 | 不需要 |
| **非对称** PD 组合（如 P8D16） | 否 | 需要改 `pd_ratio_grid()` / runner 及测试 | **需要**，且属于契约变更，须重新评审 |
| 新 CLI 参数、扩大 4.1.2 值域、改 `topk` | 否 | 契约变更 | 须重新评审 |

---

## 2. 新增「模型 + 量化方案」组合（最常见）

### 2.1 三个前置检查（先做，否则运行期会失败）

1. **框架是否认识这个 `model_id`**
   `build_request()` 把 `config.plan.model_id` 原样传给 `RecommendRequest.model`；框架 `route()` → `resolve_model()` 会查 `src/infer_strategy_npu/models.py` 的 `MODELS` 注册表（大小写不敏感）。未注册会抛 `unsupported --model`，任务变成全局 `1 / failed`，异常栈进日志。
   - 已注册的 9 个 `model_id`：`deepseek-ai/DeepSeek-R1`、`deepseek-ai/DeepSeek-V4-Pro`、`deepseek-ai/DeepSeek-V4.1-Flash`、`ZhipuAI/GLM-5.2`、`ZhipuAI/GLM-5.3`、`ZhipuAI/GLM-5.3-Flash`、`Qwen/Qwen3.8-27B`、`moonshotai/Kimi-K3`、`MiniMax/MiniMax-M3`。
   - 新 `model_id` 需在 `MODELS` 补一条 `ModelSpec(model_id=..., ms_model_id=..., local_dir=...)`；`supports_msmodeling` 由 `ms_model_id` 是否非空决定，别名放 `aliases`。`aic_model_path` 可留空（本入口强制 `framework="msmodeling"`，不走 aic-npu 路由）。`supported_models()` 会自动包含新项。
2. **镜像里是否有本地模型配置目录**
   `msmodeling_model_source()` 优先返回 `<local_model_root>/<ModelSpec.local_dir>`（默认 `model_configs/msmodeling`，容器由 `ENV INFER_STRATEGY_LOCAL_MODEL_ROOT=model_configs/msmodeling` 指定）；目录不存在会回退到 `ms_model_id` 联网拉取，而镜像设了 `HF_HUB_OFFLINE=1`，离线环境必然失败或超时。
   - 现有目录：`DeepSeek-R1/`、`DeepSeek-V4-Pro/`、`DeepSeek-V4.1-Flash/`、`GLM5.2/`、`GLM5.3/`、`GLM5.3-Flash/`、`Kimi-K3/`、`MiniMax-M3/`、`Qwen3.8-27B/`（各含 `config.json`）。
   - 新模型请补 `model_configs/msmodeling/<local_dir>/config.json`。`docker/Dockerfile.unified` 已有 `COPY model_configs/ ./model_configs/`，**无需改 Dockerfile，但需要重新构建镜像**。
3. **量化动作是否在 4.1.3 值域内**
   `parse_and_validate()` 用 `_LINEAR_VALUE_DOMAIN`（`DISABLED, W8A16_STATIC, W8A8_STATIC, W4A8_STATIC, W8A16_DYNAMIC, W8A8_DYNAMIC, W4A8_DYNAMIC, FP8, MXFP4, HIF8`）与 `_ATTENTION_VALUE_DOMAIN`（`DISABLED, INT8, FP8, HIF8`）自检，越界直接判 `1 / failed`。这两个集合已是 4.1.3 全集，**不要扩**；要用新枚举属于契约变更，须先改接口文档。
   - `quantize` 固定 `"ENABLE"`；BF16 方案用三项 `DISABLED` 表达，不能用 `DISABLE`（框架会强制覆盖显式 action）。

### 2.2 改动清单

| # | 文件 | 位置 / 符号 | 改动 | 必须 |
| --- | --- | --- | --- | --- |
| 1 | `infer-strategy-npu-task-enter.py` | `MODEL_PLANS` | 增 1 项 `ModelPlan(...)` | 是 |
| 2 | `src/infer_strategy_npu/models.py` | `MODELS` | 仅当 `model_id` 未注册时补 `ModelSpec` | 条件 |
| 3 | `model_configs/msmodeling/<local_dir>/` | `config.json` | 仅当新 `local_dir` 不存在时补，并重建镜像 | 条件 |
| 4 | `scripts/run_infer_strategy_task_enter_tests.py` | `test_model_plans` | `len(mod.MODEL_PLANS) == 8` 改为新数量；若新方案是 MXFP4，同步更新 `mxfp4` 集合断言（当前 2 个） | 是 |
| 5 | 同上 | 新增用例 | 建议补 1 个 `build_request_echo()` 回显 + `build_request()` 下发断言 | 建议 |
| 6 | `scripts/demo_infer_strategy_task_enter_responses.py` | `CASES` | 可选加 1 个演示案例（沿用 `serving_modes=0/1`） | 可选 |
| 7 | `docs/plans/INFER_STRATEGY_NPU_TASK_ENTER_DESIGN.md` | §3 参数表、§4 方案表、§10 用例 1 | 白名单数量与方案表行同步 | 是 |
| 8 | `docker/Dockerfile.unified` | — | 通常不改；仅当新量化能力需要新 msmodeling whl 时才改离线安装包 | 通常否 |

### 2.3 `ModelPlan` 六个字段

| 字段 | 含义 | 约束 / 影响 |
| --- | --- | --- |
| `model_id` | 传给 `RecommendRequest.model` 的 ModelScope ID | 必须在框架 `MODELS` 注册（§2.1 第 1 项） |
| `local_dir` | 入口日志里显示的 msmodeling 本地目录名 | **仅用于日志追溯**；真实解析用框架 `ModelSpec.local_dir`，两者建议保持一致 |
| `linear` | `--quantize-linear-action` | 必须在 `_LINEAR_VALUE_DOMAIN`；同时回显到 `request.quantize-linear-action` |
| `non_expert_linear` | `--quantize-non-expert-linear-action` | 同上 |
| `attention` | `--quantize-attention-action` | 必须在 `_ATTENTION_VALUE_DOMAIN` |
| `mxfp4_group_size` | 可选，默认 `None` | 填了才经 `extra_msmodeling_args=["--mxfp4-group-size", 值]` 下发；**不填也会在 `request` 回显默认 `"32"`**（`DEFAULT_MXFP4_GROUP_SIZE`），仅回显、不下发 |

### 2.4 示例：给已注册模型加一个新量化方案

假设框架已注册 `Qwen/Qwen3.8-27B`、镜像已有 `model_configs/msmodeling/Qwen3.8-27B/`，现在新增方案 `Qwen3.8-27B-W8A8`：

```python
# infer-strategy-npu-task-enter.py · MODEL_PLANS
"Qwen3.8-27B-W8A8": ModelPlan(
    "Qwen/Qwen3.8-27B", "Qwen3.8-27B", "W8A8_DYNAMIC", "W8A8_DYNAMIC", "INT8"
),
```

随后只需：

- `test_model_plans` 的 `8` 改成 `9`（不是 MXFP4 方案，`mxfp4` 集合断言不变）；
- 设计文档 §4 方案表加 1 行，`--mxfp4-group-size` 列填 `不下发、**回显默认 32**`；§3 与 §10 用例 1 的「8 个」改成「9 个」；
- 其余代码、测试、Dockerfile 都不动。

### 2.5 新增表项后自动生效、无需改的地方

- `--model` 白名单校验与错误提示（`parse_and_validate()` 直接遍历 `MODEL_PLANS`）；
- `request` 回显：成功 `build_request_echo()`、失败 `best_effort_request_echo()`、字段顺序 `_REQUEST_KEY_ORDER`、默认 `mxfp4-group-size`；
- `RecommendRequest` 下发与等价命令（`build_request()`、`_log_plan()`）；
- 状态机、`result` 组装、结果文件原子落盘、日志格式；
- `docker/Dockerfile.unified` 的 `COPY infer-strategy-npu-task-enter.py ./`。

---

## 3. 新增设备代号

1. `DEVICE_SPECS` 增 1 项 `DeviceSpec(ms_device, gpus_per_node)`；`ms_device` 必须是 msmodeling 认识的 `ATLAS_*` 名。
2. `gpus_per_node` 决定 `num_nodes = num_devices // gpus_per_node`，`parse_and_validate()` 要求整除且 `>0`；`num_devices` 仍限定「8 的倍数、`[8,256]`」。若新设备单机卡数不是 8，需要先确认该值域是否仍成立。
3. `test_device_specs` 补断言（当前逐项校验 3 个设备与 `gpus_per_node == 8`）。
4. 设计文档 §3 参数表、§5 设备表补行。
5. 入口强制 `framework="msmodeling"`，不改 `routing.py` 的设备分流逻辑。

---

## 4. 新增 / 调整 PD 组合

- `PD_RATIO_SIZES` 当前为 `(8, 16, 32)`。`run_all()` 中 agg 用 `PD_RATIO_SIZES[0]` 调一次；disagg 对每个值各调一次 `pd_ratio_sizes=[P]`，天然只生成 `(P,P)` 对称组合，卡数不足由框架 `pd_ratio_grid()` 自动过滤。
- 只允许在**对称集合**内增删。例如改成 `(8, 16, 32, 64)` 时：
  - `test_only_three_symmetric_calls` 的 `len(calls) == 4`、`sizes == [8,16,32]` 改为 5 / `[8,16,32,64]`；
  - `test_model_plans` 的 `PD_RATIO_SIZES` 断言同步；
  - 设计文档 §4 要点 6、§6.2、§9、§10 用例 7/14、§11 Q6 中「三种组合」的表述同步；
  - 注意 `num_devices` 上限 256，新组合要有足够卡数才有实际运行结果。
- **非对称组合（P8D16 等）不能靠改常量实现**：`pd_ratio_grid()` 对多值列表做笛卡尔积，正是当前「逐组单值调用」规避了它。要支持非对称必须改 `src/infer_strategy_npu/runners/msmodeling.py` 及相关测试，属于「内核改动 + 契约变更」，须重新评审。

---

## 5. 红线（不要随扩展一起改）

| 项 | 当前约定 |
| --- | --- |
| 4.1.2 / 4.1.3 字段、类型、值域 | 以接口文档参数说明表为最严格约束；4.1.3 示例里的字符串数值不作准 |
| `topk` | 内部固定 `1`，不开放外部指定 |
| `serving_modes` | 仅 `0/1/2` |
| 状态机 | 全局 `code` 只有 `0/1`；内层 `0/1/3` 落盘，`2/running` 仅日志 |
| `result` 形态 | `1/failed` 与 `3/empty` 时统一 `{"agg": {}, "disagg": {}}`；`0/success` 按 `serving_modes` 填候选 |
| 时间戳 | ISO8601 秒精度 `+08:00`；不做人为顺延，瞬时失败允许三值相同 |
| 结果文件 | 只在全局 `code` 确定为 `0/1` 后原子写一次；运行期只写日志 |
| 内核 | `src/infer_strategy_npu/*` 保持零改动（例外：§2.1 新增模型注册表项） |

---

## 6. 变更后验证清单

本地 Windows（不需要 msmodeling / NPU）：

```powershell
.\.venv\Scripts\python.exe scripts\run_infer_strategy_task_enter_tests.py
.\.venv\Scripts\python.exe scripts\demo_infer_strategy_task_enter_responses.py --out-dir D:\workspace\results\task_enter_responses --increment
```

`--increment` 会自动生成 `rNN` 标记的产物并保留历史轮次；不加该参数则覆盖同名文件。

逐项检查：

- 单元测试 `Ran 19 tests ... OK`（新增方案应补对应用例）；
- 新方案日志 `[mapping] model=... -> model_id=..., local_dir=...`、三项量化动作、`group-size` 的「下发 / 仅回显默认 32」标注正确；
- 等价命令中 `--quantize-linear-action` 等三项与方案表一致；只有 MXFP4 方案带 `--mxfp4-group-size`；
- 响应 JSON：`request` 14 个字段齐全且类型正确（含 `mxfp4-group-size="32"`）、外层/内层 `code/status` 组合、`result` 形态、秒精度时间戳。

真实 NPU 冒烟（容器内，需要 msmodeling）：

```bash
python infer-strategy-npu-task-enter.py --model <新方案> --device 950DT --num_devices 32 \
  --input-length 4096 --output-length 1024 --tpot-limit 100 --ttft-limit 20 \
  --serving_modes 0 --log-file /apps/suyantest/outputs/infer/infer_strategy_npu/new.log \
  --result-file /apps/suyantest/outputs/infer/infer_strategy_npu/new.json
```

- 日志出现 `unsupported --model` → 回到 §2.1 第 1 项检查框架注册表；
- 模型加载超时 / 联网失败 → 回到 §2.1 第 2 项检查 `model_configs/msmodeling/<local_dir>/config.json`。

---

## 7. 常见误操作

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 只加了 `MODEL_PLANS`，入参校验通过但执行报 `unsupported --model` | 框架 `MODELS` 未注册该 `model_id` | §2.1 第 1 项 |
| 容器里模型加载超时 / 联网失败 | `local_dir` 目录不存在，`msmodeling_model_source()` 回退到 ModelScope ID，而 `HF_HUB_OFFLINE=1` | §2.1 第 2 项 |
| 新方案立刻 `1/failed`，提示「量化动作超出 4.1.3 预留值域」 | 用了值域外枚举 | 只能选 `_LINEAR_VALUE_DOMAIN` / `_ATTENTION_VALUE_DOMAIN` 内的值 |
| `test_model_plans` 失败 | `len(mod.MODEL_PLANS) == 8` 与 MXFP4 集合是硬编码断言 | §2.2 第 4 项 |
| 非 MXFP4 方案没下发 `--mxfp4-group-size` | 正常：`mxfp4_group_size=None` 时只回显默认 `"32"`，不下发 | 无需处理 |
| `case_12` 只有 `.txt` 没有 `.json` | 它是「`--result-file` 非 .json」的失败案例，内容仍是完整契约 JSON | 见演示脚本 `CASES` 的 `result_suffix` |
| 只改了 `PD_RATIO_SIZES` 但单测报 `sizes == [8,16,32]` | `test_only_three_symmetric_calls` 硬编码了组合列表 | §4 |
