# infer-strategy-npu-task-enter.py 整体设计（v3 · 定稿，未开发）

| 项 | 内容 |
| --- | --- |
| 状态 | **设计 v3 定稿**（评审意见与 3 个决定已全部合并，无待确认项）；确认后进入开发 |
| 目标 | 以《推理工具链 V1.0.0 接口文档 01》第 4 章「推理策略寻优模块」为**唯一对外契约**，实现容器一次性任务入口脚本 `infer-strategy-npu-task-enter.py` |
| 交付形态 | 仓库根目录单文件脚本，随 `msmodeling` 工具镜像分发；容器常驻、外部 `docker exec` 触发一次任务，跑完写日志/结果文件并退出 |
| 契约范围 | 仅 4.1「推理策略寻优任务创建」：4.1.1 容器启动命令、4.1.2 参数说明、4.1.3 返回值与参数说明。**不做**第 5 章推理仿真 |
| 硬约束 | 入参/返回值字段名、类型、取值范围、默认值一律以 4.1.2 / 4.1.3 为准；不新增文档外的请求/响应字段，不扩大取值域，不遗漏参数说明表里的字段（含 `topk`、`num_nodes`）；`mxfp4-group-size` 为必返字段，方案表未解析到配对值时按默认 `32` 回显（仅回显，不下发） |
| 调整依据 | 《开发调整事项》一（`--model` 白名单 + 固定量化方案）、二（`--device` 映射 + 单机卡数 + `num_nodes`）、三（结果获取方式与值域约束） + v2 评审意见（PD 三种对称组合、结果文件写入时机、9 条确认结论） |

---

## 1. 现状调研结论（复用基础）

仓库已有完整的 `src/infer_strategy_npu/` 分层，本设计**复用编排与结果归一化能力，不重写寻优算法**：

| 能力 | 位置 | 作用 |
| --- | --- | --- |
| `recommend(RecommendRequest)` | `src/infer_strategy_npu/service.py:45` | 归一化 → `RunArtifacts.create()` → `route()` → runner → 写 `run.json/request.json/summary.txt/report.html` |
| `MsmodelingRunner` | `src/infer_strategy_npu/runners/msmodeling.py:164` | **进程内**调 `cli.inference.throughput_optimizer.arg_parse()` + `ParallelRunner.run_agg()/run_disagg()`，拿 `OptimizerSummary` 结构化对象，而不是 `subprocess` 解析 stdout |
| `Candidate` / `PdRatioCandidate` / `PdRatioSide` | `src/infer_strategy_npu/schema.py:91/175/123` | 归一化后的 agg 候选 / disagg 候选 / P、D 两侧指标（含文档所需全部字段） |
| `RunArtifacts` | `src/infer_strategy_npu/output.py:30` | `<out_dir>/<run_id>/{run.json,request.json,summary.txt,report.html,logs/*.log,msmodeling/...}` |
| 结构化设备/模型注册表 | `devices.py:12`、`models.py:43` | 26 个 `ATLAS_*` 设备名、9 个 ModelScope 模型 ID + 本地目录 |
| 量化动作解析 | `quantize.py:73` | `DISABLE` 强制三项 `DISABLED`，`ENABLE` 时以显式 action 为准 |
| 镜像 | `docker/Dockerfile.unified` | 已装 msmodeling 离线 whl、`pip install -e .`、`WORKDIR /workspace`、`CMD ["/bin/bash"]`、无 ENTRYPOINT |

**仓库内不存在** `*-task-enter.py`、不存在一次性 ENTRYPOINT；现有 HTTP 服务是常驻 uvicorn，与文档要求的「常驻容器 + 一次 `docker exec`」不同。因此新脚本是**新的最外层适配入口**，复用内层 `service.recommend()`。

---

## 2. 总体架构与运行形态

### 2.1 运行形态（对应 4.1.1）

文档容器启动命令是 `docker run -dit ... tail -f /dev/null`，服务由外部一次性调用：

```
docker exec infer_strategy_npu_optimize /bin/bash -lc 'python infer-strategy-npu-task-enter.py \
  --model DeepSeek-V4.1-Flash-MXFP4 --device 950DT --num_devices 32 \
  --input-length 4096 --output-length 1024 --tpot-limit 100 --ttft-limit 20 \
  --serving_modes 2 \
  --log-file /apps/suyantest/outputs/infer/infer_strategy_npu/01_infer_strategy_log.log \
  --result-file /apps/suyantest/outputs/infer/infer_strategy_npu/01_infer_strategy_result.json'
```

同时兼容第二种（一次性容器）：`docker run --rm -v <目标目录>:<目标目录> <image> python infer-strategy-npu-task-enter.py ...`。
脚本无状态、无端口、无守护进程，执行完退出，退出码 0/1。

### 2.2 分层与数据流

```
CLI 入参（4.1.2 全量 + 别名；ttft/tpot 有默认值）
  → ① 解析与校验层：白名单/值域/单位/路径校验，失败即定全局 code=1 → 写结果文件与日志
  → ② 映射层：model/device/量化方案/SLO/serving_modes/num_nodes → RecommendRequest
       （默认 PD 空间常量 (8,16,32)，按原始 P8D8 方式逐组 pd_ratio_sizes=[P]）
  → ③ 执行层：强制 framework=msmodeling，逐组调用 recommend()
       └ agg 一次（serving_modes 含 agg 时） + P8D8/P16D16/P32D32 各一次（含 disagg 时）
         每组卡数不足由 pd_ratio_grid() 自动过滤
  → ④ 结果层：合并各组 pd_ratio_candidates → 按 D 侧吞吐顺序 → topk=1 → 4.1.3 字段裁剪/状态机
  → ⑤ 落盘层：全局 code 确定为 0 或 1 后才写 result-file；log-file 全程写；
              默认产物落 /workspace/results/infer-strategy-npu/<run_id>/（每次调用一个）
  → 退出码
```

**关键决策：结果获取方式（对应开发调整事项三）**

| 方案 | 做法 | 结论 |
| --- | --- | --- |
| A（采用） | 复用 `service.recommend()`；其 `MsmodelingRunner` 进程内调用 msmodeling 的 `arg_parse()` / `ParallelRunner`，直接得到 `OptimizerSummary`，再归一化为 `Candidate` / `PdRatioCandidate` | **采用**：4.1.3 所需字段（吞吐量、ttft_ms、tpot_ms、parallel tp/pp/dp/ep、P/D 实例数、单实例/单卡/集群吞吐）在现有数据模型里已全部存在且语义一致；无需解析终端表格 |
| B（不采用） | `subprocess` 跑 `python -m cli.inference.throughput_optimizer ...`，解析 stdout 文本 | 不采用：msmodeling 本身**不落盘结构化结果**，stdout 是展示用表格，字段/精度/列顺序易变，且 `pd_ratio` 的 P/D 双侧指标无法可靠还原 |

A 与「直接分发于 msmodeling 工具」不冲突：进程内分发与 4.1.1/《调整事项》给出的等价命令参数完全一致；脚本会把等价命令写入主日志、`command.txt`、`run.json`，保证可复现。

---

## 3. 入参规格与校验（严格按 4.1.2 + v2 确认）

| 参数（文档名） | 必选 | 类型 | 取值范围 / 默认 | 兼容别名 | 内部/下发处理 |
| --- | --- | --- | --- | --- | --- |
| `--model` | 是 | string | 8 个白名单值（精确匹配、大小写敏感），见 §4 | 无 | 拆为「简化模型名 + 量化方案」→ 仓库 `model_id` + 量化动作 |
| `--device` | 是 | string | `910B2` / `910B4` / `950DT` | 无 | 映射为 `ATLAS_*`，见 §5 |
| `--num_devices` | 是 | integer | 8 的倍数、`[8,256]` | `--num-devices` | 直接作为 msmodeling `--num-devices`；`num_nodes = num_devices // gpus_per_node` |
| `--input_length` | 是 | integer | `[128,131072]` | `--input-length` | `--input-length` |
| `--output_length` | 是 | integer | `[128,131072]` | `--output-length` | `--output-length` |
| `--ttft_limit` | 否（默认 20） | float | `[0.001,60]`，单位**秒**；默认 `20` | `--ttft-limit` | `floor(Decimal(str(v)) * 1000)` → 毫秒整数；默认 `20 → 20000`，`0.001 → 1` |
| `--tpot_limit` | 否（默认 100） | integer | `[1,1000]`，单位**毫秒**；默认 `100` | `--tpot-limit` | 原值作为 `--tpot-limit` |
| `--serving_modes` | 否 | integer | **仅 `{0,1,2}`**，默认 `1`；`0=agg`、`1=disagg`、`2=agg+disagg`；3 及以上/负数/非整数一律判失败 | `--serving-modes` | `0→["agg"]`、`1→["disagg"]`、`2→["agg","disagg"]`；原值回显 |
| `--log-file` | 是 | string | 合法文件路径 | `--log_file` | 父目录自动 `mkdir -p`；UTF-8；同时 tee 到 stdout；**运行期持续写** |
| `--result-file` | 是 | string | 合法文件路径，必须 `.json` | `--result_file` | 父目录自动创建；**仅在全局 `code` 确定为 0 或 1 后写一次**（原子写） |

补充约定：

- 文档 4.1.2 把 `ttft_limit` / `tpot_limit` 标为必选，按 v2 确认给默认值 `20`（秒）/ `100`（毫秒），**传入即覆盖**；值域校验不变，回显本次实际生效值；`request` 各字段的输出类型严格按 4.1.3 参数说明表（见 §6.1）。
- 文档示例混用 `--num_devices` 与 `--input-length` 两种命名，故**同一参数同时注册下划线/连字符两种 option string**，不视为新增参数。
- **不暴露** 4.1.2 以外的 CLI 参数。`topk` 为**内部保留参数**、取值范围固定 `{1}`（当前固定 1，不开放外部指定）；`jobs=32`、`batch-range=4 512`、`max-batched-tokens=8192`、投机参数不下发，全部内置。
- 所有越界、非白名单、非整除、路径非法 → 不启动寻优，全局 `code=1`，直接落失败结果文件与日志，退出码 1。
- `num_nodes` 必须是 `>0` 的整数；当前 3 种设备单机 8 卡且 `num_devices` 为 8 的倍数，必然整除；未来设备若 `gpus_per_node` 变化且不能整除则判失败。

---

## 4. `--model` 白名单与固定量化方案（对应开发调整事项一）

格式：`{简化模型名}-{量化}`，简化模型名取原「厂商/模型名」中的模型名。方案表**配置驱动**，后续扩充只增表项。

| `--model` | 简化模型名 / 仓库 `model_id` | msmodeling 模型目录 | `--quantize-linear-action` | `--quantize-non-expert-linear-action` | `--quantize-attention-action` | `--mxfp4-group-size` |
| --- | --- | --- | --- | --- | --- | --- |
| `GLM-5.2-w4a4c8-mxfp4` | `GLM-5.2` / `ZhipuAI/GLM-5.2` | `GLM5.2` | `MXFP4` | `FP8` | `FP8` | `32`（下发 + 回显） |
| `GLM-5.2-BF16` | `GLM-5.2` / `ZhipuAI/GLM-5.2` | `GLM5.2` | `DISABLED` | `DISABLED` | `DISABLED` | 不下发、**回显默认 32** |
| `GLM-5.3-BF16` | `GLM-5.3` / `ZhipuAI/GLM-5.3` | `GLM5.3` | `DISABLED` | `DISABLED` | `DISABLED` | 不下发、**回显默认 32** |
| `GLM-5.2-FP8` | `GLM-5.2` / `ZhipuAI/GLM-5.2` | `GLM5.2` | `FP8` | `DISABLED` | `FP8` | 不下发、**回显默认 32** |
| `GLM-5.3-FP8` | `GLM-5.3` / `ZhipuAI/GLM-5.3` | `GLM5.3` | `FP8` | `DISABLED` | `FP8` | 不下发、**回显默认 32** |
| `DeepSeek-R1-FP8` | `DeepSeek-R1` / `deepseek-ai/DeepSeek-R1` | `DeepSeek-R1` | `FP8` | `FP8` | `FP8` | 不下发、**回显默认 32** |
| `DeepSeek-V4.1-Flash-MXFP4` | `DeepSeek-V4.1-Flash` / `deepseek-ai/DeepSeek-V4.1-Flash` | `DeepSeek-V4.1-Flash` | `MXFP4` | `FP8` | `DISABLED` | `32`（下发 + 回显） |
| `DeepSeek-V4-Pro-W8A8` | `DeepSeek-V4-Pro` / `deepseek-ai/DeepSeek-V4-Pro` | `DeepSeek-V4-Pro` | `W8A8_DYNAMIC` | `W8A8_DYNAMIC` | `INT8` | 不下发、**回显默认 32** |

要点：

1. **映射链**：白名单 → `(仓库 model_id, 量化动作, 是否下发 group-size；request 的 group-size 缺省回显 32)`；`model_id` 交给 `models.msmodeling_model_source()` 解析为本地模型目录。容器内默认 `model_configs/msmodeling/<目录>`（即案例中 `models\GLM5.2` 的等价路径；可用 `INFER_STRATEGY_LOCAL_MODEL_ROOT=/workspace/models` 对齐案例写法）。**不再把平台白名单当 ModelScope ID 直接传给 msmodeling**。
2. **量化值域**：以 4.1.3 参数说明为准——linear / non-expert 允许 `DISABLED,W8A16_STATIC,W8A8_STATIC,W4A8_STATIC,W8A16_DYNAMIC,W8A8_DYNAMIC,W4A8_DYNAMIC,FP8,MXFP4,HIF8`；attention 允许 `DISABLED,INT8,FP8,HIF8`。脚本对表内取值做白名单校验，当前 8 项全部落在文档值域内。现有 `quantize.py:38/50` 的枚举缺 `HIF8`，当前方案不涉及；后续扩充到 `HIF8` 时再补枚举。
3. **必须 `quantize="ENABLE"`**：`resolve_ms_quantize_actions()` 在 `DISABLE` 时会**忽略显式 action 并强制三项 `DISABLED`**。因此统一构造 `RecommendRequest(quantize="ENABLE", 三个 action=表内值)`，BF16 方案用三个 `DISABLED` 表达，避免语义被覆盖。
4. **`--mxfp4-group-size 32`**：已确认 msmodeling 支持（参考案例均可跑出候选），**不做探测/降级**。仓库未建模该参数，仅 2 个 MXFP4 方案经 `extra_msmodeling_args=["--mxfp4-group-size","32"]` 逃生舱透传；非 MXFP4 方案**不下发该参数，但 `request` 仍按默认值 `"32"` 回显**（本轮调整，覆盖 v2 Q4 原结论；下发行为同 Q8）。
5. **固定下发参数**（平台入参中没有，按《调整事项》PD 分离案例固定）：`--jobs 32`、`--max-batched-tokens 8192`、`--batch-range 4 512`，以及每个 PD 组合的 `--enable-optimize-prefill-decode-ratio --prefill-devices-per-instance P --decode-devices-per-instance D`；不下发 `--compile`、投机参数、`--max-search-combinations`、`--tp-sizes`。
6. **PD 寻优空间不推导**（v2 Q6）：固定 **P8D8 / P16D16 / P32D32** 三种对称组合，与模型、卡数无关；三个组合都参与寻优，最后选其中结果最高的（详见 §6.2）。案例只是「能跑出结果的示例」，不作为每实例卡数的推导依据。
7. **案例只说明量化参数分发逻辑**；`ttft/tpot` 入参默认 `20s / 100ms`，传入即覆盖，案例数值不覆盖入参。
8. 等价命令示例（`--serving_modes 2`、32 卡时的 P8D8 子任务；SLO 参数因 `slo_flags()` 追加在 argv 末尾，顺序不同但语义等价）：

```
python -m cli.inference.throughput_optimizer model_configs/msmodeling/DeepSeek-V4.1-Flash \
  --device ATLAS_950_486T_96G --num-devices 32 --input-length 4096 --output-length 1024 \
  --quantize-linear-action MXFP4 --quantize-non-expert-linear-action FP8 \
  --quantize-attention-action DISABLED --mxfp4-group-size 32 \
  --enable-optimize-prefill-decode-ratio --prefill-devices-per-instance 8 \
  --decode-devices-per-instance 8 --max-batched-tokens 8192 --jobs 32 --batch-range 4 512 \
  --tpot-limit 100 --ttft-limit 20000
```

同一次运行还会产生 P16D16 的等价命令（32 卡时 P32D32 被自动过滤）；agg 子任务另有一条不带 PD 三参数的命令。

---

## 5. `--device` 映射与 `num_nodes`（对应开发调整事项二）

| `--device` | msmodeling `--device` | 单机卡数 `gpus_per_node` | 说明 |
| --- | --- | --- | --- |
| `910B2` | `ATLAS_800_A2_376T_64G` | 8 | 属性表记录，供后续扩展 |
| `910B4` | `ATLAS_800_A2_280T_64G` | 8 | 同上 |
| `950DT` | `ATLAS_950_486T_96G` | 8 | 同上 |

- `DEVICE_SPECS` 以表驱动保存 `{平台名, ATLAS 名, gpus_per_node}`，新增设备只改表；`gpus_per_node` 同时写入主日志和内部 `platform_request.json`（**不写入 4.1.3 契约 JSON**，避免新增文档外字段）。
- `num_nodes = num_devices // gpus_per_node`，整数且 `>0`（当前范围 `1..32`）；写入 `common.infer_strategy_npu.data.request.num_nodes`。
- 强制 `framework="msmodeling"`，不进入 aic-npu 路由（`route()` 的 `auto` 会把部分模型/设备组合分流到 aic-npu，与本脚本契约不符）。

---

## 6. 结果组装与状态机（严格按 4.1.3）

### 6.1 字段全集校验清单（「不能缺少实现」）

- 外层：`code`、`status`、`message`、`common`
- 内层 `common.infer_strategy_npu`：`code`、`status`、`message`、`data`
- `data`：`created_at`、`started_at`、`finished_at`、`request`、`result`
- `request` 字段与类型**严格按 4.1.3 参数说明表**（示例 JSON 把数值写成字符串，不作准）：

| 字段 | 类型 | 取值/默认 |
| --- | --- | --- |
| `model` | string | 平台白名单值 |
| `device` | string | `910B2` / `910B4` / `950DT` |
| `num_devices` | integer | 8 的倍数、`[8,256]` |
| `num_nodes` | integer | `num_devices // gpus_per_node`，`>0` |
| `input_length` | integer | `[128,131072]` |
| `output_length` | integer | `[128,131072]` |
| `ttft_limit` | float | 单位秒，`[0.001,60]`，默认 `20` |
| `tpot_limit` | integer | 单位毫秒，`[1,1000]`，默认 `100` |
| `serving_modes` | integer | `{0,1,2}`，默认 `1` |
| `quantize-linear-action` | string | 值域见 §4 要点 2 |
| `quantize-non-expert-linear-action` | string | 值域见 §4 要点 2 |
| `mxfp4-group-size` | string | 默认 `32`；**必返字段**：方案表解析到配对值则回显该值，否则回显默认 `"32"`；所有模型与失败响应都带该键（仅回显，不下发） |
| `quantize-attention-action` | string | `DISABLED` / `INT8` / `FP8` / `HIF8` |
| `topk` | integer | 内部固定 `1`，取值范围 `{1}`，不开放外部指定 |

- `result.agg.candidates[]`：`throughput_tokens_per_s`、`throughput_tokens_per_s_per_device`、`throughput_req_per_s`、`ttft_ms`、`tpot_ms`、`meets_slo`、`concurrency`、`batch_size`、`num_devices`、`parallel{tp,pp,dp,ep}`
- `result.disagg.candidates[]`：`prefill` 与 `decode` 各含 `instances`、`devices_per_instance`、`devices_total`、`parallel{tp,pp,dp,ep}`、`batch_size`、`concurrency`、`ttft_ms`、`tpot_ms`、`tokens_per_s_per_instance`、`tokens_per_s_per_device`、`tokens_per_s_cluster`
  - 按示例：`prefill.tpot_ms = null`、`decode.ttft_ms = null`（现有 `PdRatioSide` 天然如此）。

**「不能跳出去」**：结果 JSON 只输出上述字段；`framework`、`mode`、`quantize`、`extra`、`raw`、`qps_*`、`parallel_str`、`label`、`pd_ratio`、`run_id` 等内部字段一律不进契约 JSON（仅存在于内部产物目录）。`request` 按参数说明表类型输出原生 JSON 类型（integer/float/string），`result` 指标保持 float/int/bool/null；字段缺失值统一 `null`，`parallel` 四字段即使为 `null` 也必须存在。

### 6.2 PD 寻优空间、自动过滤与 topk 语义（v3 定稿）

**默认只有三种对称搜索组合**：脚本常量 `PD_RATIO_SIZES = (8, 16, 32)` → P8D8 / P16D16 / P32D32；不做任何推导、不引入非对称组合。

> 实现方式（按「原始 P8D8 方式」，**零内核改动**）：`routing.py:137-161` 的 `pd_ratio_grid()` 是 `sizes × sizes` 笛卡尔积，传 `[8,16,32]` 会生成 9 组（含 P8D16 等）。因此脚本对每种卡数分别构造 `RecommendRequest(pd_ratio_sizes=[P])`，各自只生成 `(P,P)` 一组，天然严格只有三种对称组合；无需改 `schema.py` / `routing.py` / `runners/msmodeling.py`，现有 CLI/HTTP/回归行为完全不变。

**调用编排**：agg 单独调一次 `recommend(serving_modes=["agg"])`（仅当 `serving_modes` 含 agg，避免重复跑）；disagg 对 `P=8/16/32` 各调一次 `recommend(serving_modes=["disagg"], pd_ratio_sizes=[P])`。每次调用各自生成一个 `run_id` 与一份默认产物，主日志汇总所有 `run_id` 及对应关系；结果层合并各次 `FrameworkRun.pd_ratio_candidates` 后统一组装契约 JSON。

**卡数不够时框架自动过滤（已核实）**：`pd_ratio_grid()` 剔除 `p + d > num_devices` 及 `p/d > total`；`runners/msmodeling.py:218-223` 在网格为空时只记一条 failure 而不抛异常；`_run_pd_ratio_combo()` 单组合无候选也只记 failure 并继续下一个组合。

| `num_devices` | 实际运行的 PD 组合 | 说明 |
| --- | --- | --- |
| 8 | 无 | P8D8 需 16 卡，三组全被过滤，disagg 无结果（记入日志） |
| 16 | P8D8 | 其余两组被过滤 |
| 32 | P8D8、P16D16 | P32D32 被过滤 |
| 64 / 128 / 256 | P8D8、P16D16、P32D32 | 三组全跑 |

**topk 语义与 PD 排序（按 v3 决定）**：

1. 每个 PxDy 组合**各自独立**跑一次 `run_disagg()`；msmodeling 的 `OptimizerSummary._prepare_pd_ratio_results()` 在**组合内部**做 SLO 过滤、按 `(parallel_p, parallel_d)` 与 `balanced_qps` 去重、排序，产出该组合自己的候选集（「各个 PxDy 下各自保留候选」）。脚本**不在子任务层把每个组合截断到 1 条**。
2. 各组 `FrameworkRun.pd_ratio_candidates` 由框架按 **`decode_tps_per_instance`（单 D 实例 decode 吞吐，即 D 节点吞吐）** 降序排好；脚本合并三组候选后**直接沿用 D 侧吞吐排序**，不做集群吞吐重排。
3. 在跨组合汇总之后取 `topk=1`，即 D 侧吞吐最高的那一条；`agg.candidates` 仍按 `throughput_tokens_per_s` 降序取 `topk=1`。
4. 某组合没结果不影响其他组合；三组都没结果 → 内层 `3 / empty`，`result` 两侧统一 `{}`（不再输出 `{"candidates": []}`）。

### 6.3 字段映射

| 契约字段 | 取自 `UnifiedRunResult` | 处理 |
| --- | --- | --- |
| `agg.candidates` | `frameworks[0].agg.candidates`（`Candidate.to_dict()`） | 按 `throughput_tokens_per_s` 降序 → `[:topk]` → 裁剪到 10 个标量字段 + `parallel` 四字段（补齐缺失键为 `null`） |
| `disagg.candidates` | 三次调用各自的 `frameworks[0].pd_ratio_candidates`（`PdRatioCandidate.to_dict()`） | 合并后直接沿用 **D 侧吞吐 `decode_tps_per_instance`** 降序 → `[:topk]` → 取 `prefill`/`decode` 子对象 → 裁剪到 12 个标量字段 + `parallel` 四字段（补齐缺失键为 `null`） |
| 空形态 | `serving_modes` 未包含该形态 | 保留 key，值为 `{}`（4.1.3：`0` 时 `disagg={}`，`1` 时 `agg={}`） |
| 无候选 | 请求了该形态但无可行解（内层 `3 / empty`） | 整个 `result` 统一为 `{"agg": {}, "disagg": {}}`，不再出现 `{"candidates": []}` |
| 数值 | `schema.to_float/to_int` | NaN/缺失 → `null`；bool 保持 bool；不四舍五入、不改单位（ttft/tpot 统一 ms） |

`serving_modes=2` 时 agg 跑一次、三组 PD 各跑一次，两个形态各取 1 条候选，与 4.1.3 示例结构一致。

### 6.4 时间、类型与状态

- `created_at` / `started_at` / `finished_at`：`datetime.now().astimezone().isoformat(timespec="seconds")`，容器设 `TZ=Asia/Shanghai` 保证形如 `2026-11-04T09:06:16+08:00`；三值分别在建任务、进入执行、写结果文件前取。
- `request` 严格按 4.1.3 参数说明表输出原生类型（§6.1）：`num_devices/num_nodes/input_length/output_length/tpot_limit/serving_modes/topk` 为 integer、`ttft_limit` 为 float（秒，如 `20.0`，未传时默认 `20.0`；内部/下发用毫秒 `20000`）、其余为 string（`mxfp4-group-size` 输出 `"32"`）。4.1.3 示例里的字符串数值不作准。
- 全局 `code` **只有 `0` / `1`**（v2 Q3）；内层 `code` 保留 4.1.3 定义的 `0/1/2/3`。

### 6.5 状态机、结果文件写入时机与退出码

| 场景 | 全局 `code/status/message` | 内层 `code/status/message` | 退出码 | `result` |
| --- | --- | --- | --- | --- |
| 成功找到结果 | `0 / success / 寻优任务完成` | `0 / success / 寻优策略任务完成，已找到结果` | 0 | 按 `serving_modes` 填 candidates |
| 正常运行但无任何候选（含卡数不足、SLO 过严） | `0 / success / 寻优任务完成（无结果）` | `3 / empty / 未找到满足 SLO 的部署策略` | 0 | `{"agg": {}, "disagg": {}}`，两侧统一 |
| 入参非法 / 路径不可写 | `1 / failed / 具体校验错误` | `1 / failed / 入参不对：…` | 1 | 完整骨架，`request` 尽力回显、`result` 两侧统一 `{}` |
| msmodeling 导入/执行失败、异常 | `1 / failed / 寻优任务执行失败：…` | `1 / failed / 任务报错失败：…` | 1 | 同上 |
| 运行中 | **不写结果文件、不写任何状态文件** | 枚举 `2 / running` 仅体现在日志阶段记录 | — | — |

**写入时机（v2 修改）**：`--result-file` 在**全局 `code` 确定为 `0` 或 `1` 之后才写，且只写一次**（原子写）；运行期间只持续写 `--log-file`（及内部产物日志）。内层 `2 / running` 不作为终态写入任何文件。失败路径同样写出完整 4.1.3 JSON 骨架（全局 `code=1` 已确定），保证平台侧总能解析。

判定依据：以「agg / disagg 是否至少有一条候选」区分 `0` 与 `3`，**不能只看 `result.status`**——卡数不足时框架会把 `run.status` 置为 `failed` 并附 failure 说明，但语义上是「无结果」（`3 / empty`）；真正的执行异常（导入失败、设备检查失败、未捕获异常）才判 `1 / failed`。

---

## 7. 日志与数据落盘

1. `--result-file`：最终契约 JSON，`ensure_ascii=False`、缩进 2、UTF-8；**仅终态写一次**（全局 `code` 确定为 0/1 后），先写同目录临时文件再 `os.replace` 原子替换；同内容另存一份到默认产物目录便于追溯。
2. `--log-file`：主日志（同时输出到 stdout，便于 `docker exec` 实时查看），格式 `[ISO时间] [级别] [阶段] 消息`，**全程持续写**，内容至少包含：
   - 原始入参、别名归一化结果、校验结论、`ttft/tpot` 是否取默认值；
   - 解析出的模型方案 / 设备属性 / `gpus_per_node` / `num_nodes` / 量化动作 / 是否下发 group-size；
   - 内部 `RecommendRequest` 摘要与**等价 msmodeling 命令**（agg 一条 + 实际运行的每个 PxDy 一条），以及被自动过滤掉的 PD 组合；
   - 每个 PD 组合的候选数、**D 侧吞吐**排序结果、`topk` 截断结果；
   - 各次 `recommend()` 的 `run_id` 与其对应的 agg / PxDy 子任务、默认产物目录、框架 stdout 日志路径；
   - 阶段进度（`running`）、各阶段耗时、异常堆栈（失败时）、最终状态码与退出码。
3. **默认产物路径按默认值**（v2 Q9）：每次调用 `recommend()` 时 `out_dir` 不指定（或显式 `DEFAULT_OUT_DIR`），**每次调用各生成一个 `<run_id>`**，产物落在 `/workspace/results/infer-strategy-npu/<run_id>/`，内含 `run.json`、`request.json`、`platform_request.json`（含 `num_nodes`、`gpus_per_node`、平台原始入参）、`summary.txt`、`report.html`、`logs/msmodeling_*.log`（由 `capture_stdout()` 生成）、`msmodeling/agg/*`、`msmodeling/pd_ratio/P8D8|P16D16|P32D32/*` 的 csv/json/command.txt。
   - 即：**契约 JSON 与主日志落在 `--result-file` / `--log-file` 指定的目标文件夹**；额外的默认产物仍按框架默认路径保存，不写到目标文件夹、也不改变契约 JSON 字段。
4. 目录不存在自动 `mkdir -p`；写失败要降级到 stdout 报错并退出 1。
5. 单实例假设：同一容器同一时刻只跑一次 `docker exec`；`result-file` 原子写避免半文件；不做并发队列（平台侧调度）。

---

## 8. msmodeling「对不上的参数逻辑」清单（实现时逐项处理）

| # | 差异 | 处理 |
| --- | --- | --- |
| 1 | 平台 `--num_devices/--input_length/...` 下划线 vs 仓库 `--num-devices/--input-length/...` | 同参数双 option string 注册 |
| 2 | TTFT 单位秒 vs msmodeling 毫秒 | `floor(s*1000)`；`request.ttft_limit` 按参数说明表回显为 float 秒值 |
| 3 | `serving_modes` 数字 `0/1/2` vs 仓库 `serving_modes: list[str]` | 显式映射 + `{0,1,2}` 范围限制；2 → 两者都跑；未跑形态输出 `{}` |
| 4 | 平台模型白名单 vs 仓库 ModelScope ID | 方案表映射到 `model_id`，再走 `msmodeling_model_source()` 本地目录 |
| 5 | 平台设备代号 vs `ATLAS_*` | `DEVICE_SPECS` 映射（含 `gpus_per_node=8`） |
| 6 | 仓库 `--quantize DISABLE` 会覆盖显式 action | 统一 `ENABLE` + 显式三 action |
| 7 | `--mxfp4-group-size` 仓库未建模（msmodeling 已确认支持） | 仅 MXFP4 方案经 `extra_msmodeling_args` 透传；argv 位于 `slo_flags()` 之前 |
| 8 | 平台无 PD 分离参数 | 常量 `PD_RATIO_SIZES=(8,16,32)`，逐组 `pd_ratio_sizes=[P]`（原始 P8D8 方式）→ 三种对称组合；每组派生 `--enable-optimize-prefill-decode-ratio` 等三项 |
| 9 | 平台无 `jobs/batch-range/max-batched-tokens` | 固定 `32` / `4 512` / `8192` 下发 |
| 10 | `cli.py` 默认投机方法 `dflash` vs 案例命令无投机参数 | 直接构造 `RecommendRequest`（不复用 `cli.build_parser()` 默认值），投机字段全 `None` |
| 11 | `top_n` 默认 10 且 runner 不用于截断 | 不在子任务层截断；跨组合汇总后按 `topk=1` 截断两类 candidates |
| 12 | 等价命令中 SLO 参数位置与案例不同 | 仅顺序不同，argparse 语义等价；日志打印每个子任务的可复现命令 |
| 13 | 代码量化枚举缺 `HIF8`、值域比 4.1.3 窄 | 以文档值域为校验真源；当前 8 方案不受影响；扩充时补 `quantize.py` |
| 14 | `route()` 的 `auto` 可能分流到 aic-npu | 强制 `framework="msmodeling"` |
| 15 | 案例模型路径 `models\GLM5.2` vs 容器 `model_configs/msmodeling/GLM5.2` | 通过 `INFER_STRATEGY_LOCAL_MODEL_ROOT` 对齐；目录名一致、离线可用 |
| 16 | msmodeling `check_device_targets()` 依赖 NPU 运行环境 | 仅在容器内运行；失败按状态机落 `failed` |
| 17 | 4.1.2 标 `ttft/tpot` 必选，v2 要求默认值 | 默认 `20s / 100ms`，传入覆盖，值域校验不变 |
| 18 | `pd_ratio_grid()` 是笛卡尔积，含非对称组合 | 按原始 P8D8 方式逐组单值调用 `pd_ratio_sizes=[P]`，天然只有 3 种对称组合，**零内核改动** |
| 19 | 框架 `run.status=failed` 也可能是「卡数不足无 PD 组合」 | 以候选是否为空判 `empty`，不直接映射为 `failed`（§6.5） |
| 20 | 内部产物默认路径 vs 平台目标文件夹 | 默认产物保留 `/workspace/results/infer-strategy-npu/<run_id>/`（每次调用一个），契约 JSON/主日志写平台指定路径 |
| 21 | 4.1.3 示例把 `request` 数值写成字符串，参数说明表写 integer/float | **以参数说明表为最严格约束**输出原生 JSON 类型；示例不作准（§6.1） |
| 22 | 框架 PD 候选默认排序键 vs 集群吞吐 | PD 分离**直接按 D 节点吞吐 `decode_tps_per_instance`** 排序，不做集群吞吐重排 |

---

## 9. Docker 集成与文件改动清单（评审通过后才执行）

| 动作 | 文件 | 内容 |
| --- | --- | --- |
| 新增 | `infer-strategy-npu-task-enter.py`（仓库根） | 单文件入口：配置表（含 `PD_RATIO_SIZES=(8,16,32)`）+ 解析/校验 + 映射 + 逐组执行 + 跨组合汇总 + 落盘 + 状态机；顶部做 `sys.path` 兜底与依赖自检 |
| 修改 | `docker/Dockerfile.unified` | `COPY infer-strategy-npu-task-enter.py ./`、`ENV TZ=Asia/Shanghai`，保留 `CMD ["/bin/bash"]`（与 4.1.1 常驻 + `docker exec` 模式一致） |
| **不改** | `src/infer_strategy_npu/*`（`cli.py` / `service.py` / `routing.py` / `schema.py` / `runners/*` / `api/*`） | **内核零改动**：用原始 P8D8 方式逐组 `pd_ratio_sizes=[P]` 调用，天然只生成 `(P,P)`，现有 CLI / HTTP / 回归用例完全不受影响 |
| 不改 | `pyproject.toml` | 不额外注册 console script，避免与文档 `python infer-strategy-npu-task-enter.py` 调用方式混淆 |
| 新增 | `scripts/run_infer_strategy_task_enter_tests.py`（或 `tests/`） | 映射表、校验、跨组合汇总与结果组装的单测 + 需要 msmodeling 环境的集成冒烟 |

脚本内 `PD_RATIO_SIZES=(8,16,32)` 为**默认且唯一的 PD 搜索空间**；agg 单独调一次（仅当请求含 agg），disagg 逐组调三次，每次 `recommend()` 各生成一个 `run_id` / 一份默认产物，主日志汇总对应关系。若后续要求「一次调用、单一 `run_id`」，可另行确认后新增默认关闭的 `symmetric_only` 开关（改 `schema.py` / `routing.py` / `runners/msmodeling.py` 3 处），当前**不采用**。

镜像沿用 `docker/Dockerfile.unified`：含 msmodeling whl、`pip install -e .`、`model_configs/msmodeling/*`、`HF_HUB_OFFLINE=1`。运行时沿用 4.1.1 的 NPU 设备与卷挂载（`-v /apps/suyantest:/apps/suyantest` 等），`--log-file` / `--result-file` 必须落在已挂载目录内；默认产物落在镜像内 `/workspace/results/infer-strategy-npu/`（如需持久化可另挂该卷）。

---

## 10. 验收用例（开发后执行）

| # | 用例 | 期望 |
| --- | --- | --- |
| 1 | 8 个白名单 × 3 设备等价命令核对（内部 dry-run 调试，不新增 CLI 参数） | 量化动作、group-size 是否下发、jobs/batch-range 与方案表一致；每个 disagg 子任务 P=D |
| 2 | 文档 4.1.1 示例命令全量执行（950DT / 32 卡 / ttft 20 / tpot 100 / modes 2） | 内层 `code=0`；`num_nodes=4`(int)、`ttft_limit=20.0`(float)、`serving_modes=2`(int)，agg 与 disagg 各 1 条候选 |
| 3 | 值域边界：`num_devices=8/256`、`input/output=128/131072`、`ttft=0.001/60`、`tpot=1/1000` | 通过；`ttft 0.001 → --ttft-limit 1` |
| 4 | 不传 `--ttft_limit/--tpot_limit` | 默认生效，等价命令 `--ttft-limit 20000 --tpot-limit 100`，`request` 回显 `20.0`(float) / `100`(int) |
| 5 | 非法值：`num_devices=12`、`model=GLM-5.2-FP16`、`device=950X`、`ttft=0`、`serving_modes=3`、`result-file=.txt` | 退出 1，全局 `code=1`，写出完整 failed JSON 与日志 |
| 6 | `serving_modes=0/1/2` | 分别得到 `disagg={}` / `agg={}` / 两者都有 |
| 7 | 三种 PD 组合与自动过滤：8 / 16 / 32 / 64 卡 | 分别为 0 / 1 / 2 / 3 组实际运行；被过滤组合记入日志；8 卡时 disagg 为空但全局仍 `code=0` |
| 8 | topk 语义与 D 侧排序（构造每组多条候选的结果） | 各组候选不被提前截断；最终 `disagg.candidates` 仅 1 条，且为三组中 `decode_tps_per_instance` 最高者 |
| 9 | Q4：MXFP4 方案 vs 非 MXFP4 方案 | 所有模型与失败响应的 `request` 都含 `mxfp4-group-size="32"`(string)；仅 MXFP4 方案下发该参数 |
| 10 | 运行中观察文件 | `--log-file` 持续有阶段日志，`--result-file` 在任务结束前不存在；结束后只写一次 |
| 11 | 无可行解（极端 SLO） / msmodeling 导入失败 | 前者内层 `3 / empty`、全局 0、`result` 两侧统一 `{}`；后者内层 `1 / failed`、全局 1，异常栈进日志 |
| 12 | 落盘路径与产物 | `--log-file`/`--result-file` 目录自动创建；默认产物在 `/workspace/results/infer-strategy-npu/<run_id>/`，逐组调用各一个 |
| 13 | 契约 JSON 字段与类型断言（脚本化） | 与 §6.1 参数说明表一致，无多余/缺失字段；integer/float/string 类型逐一校验，示例的字符串数值不被沿用 |
| 14 | 内核零改动与空间校验 | `src/infer_strategy_npu/*` 无改动；日志中只出现 P8D8 / P16D16 / P32D32，不出现 P8D16 等非对称组合 |
| 15 | `docker exec` 一次调用 → 退出 → 再次调用 | 无状态残留，两次 `run_id` 不同，结果互不覆盖 |

---

## 11. 评审确认结论（v3 定稿，无待确认项）

| # | 确认结论 | 落地位置 |
| --- | --- | --- |
| Q1 | `request` 字段**严格按 4.1.3 参数说明表输出原生 JSON 类型**：`model/device/quantize-*-action/mxfp4-group-size` 为 string；`num_devices/num_nodes/input_length/output_length/tpot_limit/serving_modes/topk` 为 integer；`ttft_limit` 为 float。**4.1.3 示例中的字符串数值不作准** | §6.1 / §6.4 / §8 |
| Q2 | `serving_modes` 接受 `0/1/2`、默认 `1`，**做范围限制，不扩展到 3 及以上** | §3 |
| Q3 | 全局 `code` **只有 0、1**；无结果时全局 `0/success` + 内层 `3/empty` 且 `result` 两侧统一 `{}`；执行失败全局 `1/failed` + 内层 `1/failed` 且同样两侧 `{}` | §6.4 / §6.5 |
| Q4 | **所有方案都回显 `mxfp4-group-size`**：方案表解析到配对值则回显该值，否则回显默认 `"32"`(string)；仅 2 个 MXFP4 方案下发该参数（本轮调整，覆盖 v2 Q4 原结论） | §4 / §6.1 |
| Q5 | `topk` 不开放外部指定，**内部保留参数、取值范围固定 `{1}`**（当前固定 1） | §3 / §6.1 / §6.2 |
| Q6 | **PD 每实例卡数不推导**：默认只有 P8D8 / P16D16 / P32D32 三种对称组合，卡数不足由框架自动过滤 | §4 / §6.2 / §9 |
| Q7 | 案例只说明量化参数分发逻辑；`ttft/tpot` 虽标必选，但给默认值 **20s / 100ms**，传入即覆盖 | §3 / §4 |
| Q8 | `--mxfp4-group-size` **已确认支持**（参考案例可跑出候选），直接下发，不做探测/降级 | §4 / §8 |
| Q9 | result-file 之外**保留默认产物**，路径按框架默认值 `/workspace/results/infer-strategy-npu/<run_id>/` | §7 / §9 |

### 本轮 3 个决定

| # | 决定 | 实现 |
| --- | --- | --- |
| 决定 1 | **默认只有 3 种对称搜索组合** | 采用**内核零改动**方案：脚本常量 `PD_RATIO_SIZES=(8,16,32)`，按原始 P8D8 方式逐组 `pd_ratio_sizes=[P]` 调用 `recommend()`（agg 单独一次、disagg 三次）；`pd_ratio_grid()` 单值只生成 `(P,P)`，天然无 P8D16 等非对称组合，卡数不足仍由框架自动过滤。`src/infer_strategy_npu/*` 完全不改 |
| 决定 2 | **PD 分离直接按 D 节点吞吐排序** | 沿用框架 `pd_ratio_candidates` 的 `decode_tps_per_instance`（单 D 实例 decode 吞吐）降序，合并三组后**不做集群吞吐重排**，直接取 `topk=1`；agg 仍按 `throughput_tokens_per_s` 取 1 条 |
| 决定 3 | **4.1.3 示例可能不对，以参数说明表为最严格约束** | `request` 全部字段按参数说明表类型输出（见 Q1）；示例 JSON 的字符串数值不沿用 |

---

**结论**：v3 已合并全部评审意见与 3 个决定——`request` 类型严格按 docx 参数说明表；默认只有三种对称 PD 组合且**内核零改动**；PD 分离按 D 节点吞吐排序、`topk=1` 在跨组合汇总后生效；结果文件仅在全局 `code` 确定为 0/1 时写一次，运行期只写日志；默认产物按框架默认路径保存。设计已定稿，确认后即可进入开发。
