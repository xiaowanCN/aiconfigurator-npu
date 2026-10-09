# 6 接口文档（CLI）

> 本章描述 aiconfigurator-npu（AIConfigurator Ascend NPU 版）**default 模式**的 CLI 接口规格。
>
> 与传统 API 接口文档不同，本系统以命令行方式对外提供能力：**命令行选项（Options）对应"请求参数"，标准输出表格与落盘结果对应"响应参数"，进程退出码对应"响应码"**。
>
> - 主要参考：`docs/ARCHITECTURE_AND_CALCULATION_FRAMEWORK.md`
> - 参数定义位置：`src/aiconfigurator_npu/cli/main.py`（`_add_default_mode_arguments` / `_build_common_cli_*`）
> - 输出与落盘位置：`src/aiconfigurator_npu/cli/report_and_save.py`（`log_final_summary` / `save_results`）
> - 术语约定：文中 "Qwen3.8" / `qwen_3_8_config.json` 指 **Qwen3-8B**（约 8B 参数的 dense 模型），下同。

---

## 6.1 访问方式

> 对应原 API 文档中的"访问域名"：CLI 接口无 HTTP 域名，通过可执行入口 `aic-npu` 在本机或容器内访问。

### 6.1.1 命令入口

| 项目 | 说明 |
|------|------|
| 可执行命令 | `aic-npu` |
| 入口定义 | `pyproject.toml` → `aic-npu = aiconfigurator_npu.cli.main:main` |
| 子命令 | `default`（聚合 agg 与分离 disagg 对比推荐） |
| 等价写法 | `aic-npu cli default ...`（入口会剥离 `cli` 前缀，语义相同） |
| 运行环境 | Linux（Ascend 910B 容器/主机）；需已安装本包及对应性能数据库 |
| 认证鉴权 | **无**。CLI 为本地进程调用，不涉及账号、Token 或签名 |

### 6.1.2 接口调用格式

```bash
aic-npu default [公共选项] --model-path <模型> --total-gpus <N> --system <系统> [其他选项]
```

**本文档的基准示例命令**：

```bash
aic-npu default \
  --model-path /workspace/model_configs/qwen_3_8_config.json \
  --total-gpus 8 \
  --system ascend_910b \
  --backend vllm-ascend \
  --database-mode HYBRID \
  --isl 128 \
  --osl 128 \
  --ttft 1600 \
  --tpot 60 \
  --top-n 10 \
  --save-dir results
```

> 说明：设计稿中 `--isl <> --osl <>` 为占位符；上例取 `128/128`，实际使用时替换为业务真实输入/输出序列长度即可。

---

## 6.2 公共请求参数

> 对应原 API 文档"公共请求头参数"。公共请求参数是 **default 等实验类子命令均支持** 的选项，可在任意示例命令中附加；本节之外的 default 专属参数见 [6.6](#66-default-接口请求参数详表)。

表 6 1 公共请求参数说明

| 名称 | 描述 | 是否必填 | 类型/取值 | 默认值 | 示例 |
|------|------|----------|-----------|--------|------|
| `total-gpus` | 自身总计 NPU 资源数（卡），作为部署 GPU 预算参与选型 | 是 | 正整数；agg 要求 ≥0，disagg 要求 ≥2（`<2` 时跳过 disagg） | 无 | `8` |
| `isl` | 输入序列长度（token 数），决定 prefill 计算量与 KV 占用 | 否 | 正整数 | `4000` | `128` |
| `osl` | 输出序列长度（token 数），决定 decode 步数与 `request_latency` | 否 | 正整数 | `1000` | `128` |
| `ttft` | 首 Token 时延上限（ms），作为 **SLA 硬过滤上界** | 否 | 浮点数（ms） | `2000` | `1600` |
| `tpot` | 每输出 Token 时延上限（ms），作为 **SLA 硬过滤上界** | 否 | 浮点数（ms） | `30` | `60` |
| `top-n` | 保留方案数上限（个），agg/disagg 各保留 top-n 条 | 否 | 正整数 | `5` | `10` |
| `save-dir` | 结果落盘根目录；不传则仅打印不落盘 | 否 | 目录路径 | `None` | `results` |
| `model-path` / `model` | 模型定义：HF model id、含 `config.json` 的目录，或 `config.json` 文件路径 | 是 | 字符串 | 无 | `/workspace/model_configs/qwen_3_8_config.json` |
| `system` | 硬件系统名，对应 `systems/{system}.yaml`；本仓库内置 `ascend_910b` | 是 | 字符串（运行时校验数据库存在） | 无 | `ascend_910b` |
| `backend` | 推理后端，决定调度模型与性能数据目录 | 否 | `trtllm` / `sglang` / `vllm` / `vllm-ascend` / `auto` | `trtllm` | `vllm-ascend` |
| `backend-version` | 后端性能数据库版本；缺省取该 backend 最新版本 | 否 | 版本号字符串 | `latest`（如 `0.18.0`） | `0.18.0` |
| `database-mode` | 性能估算取数模式 | 否 | `SILICON` / `HYBRID` / `EMPIRICAL` / `SOL` | `SILICON` | `HYBRID` |
| `decode-system` | disagg 模式下 decode 侧硬件系统；缺省同 `--system` | 否 | 字符串 | 同 `--system` | `ascend_910b` |
| `request-latency` | 端到端请求延迟目标（ms）；设置后启用 request-latency 优化轴 | 否 | 浮点数（ms） | `None` | `4000` |
| `prefix` | 前缀缓存命中长度；有效 prefill token = `(isl-prefix)×bs` | 否 | 非负整数 | `0` | `0` |
| `nextn` | MTP 投机解码 draft token 数，`0` 表示关闭 | 否 | 非负整数 | `0` | `0` |
| `nextn-accept-rates` | MTP 各 draft 位接受率，5 个逗号分隔浮点数 | 否 | 字符串 | `0.85,0.3,0,0,0` | `0.85,0.3,0,0,0` |
| `enable-chunked-prefill` | 启用 chunked prefill，细化 `ctx_tokens` 扫描粒度 | 否 | 开关（flag） | 关闭 | （加上该 flag 即启用） |
| `systems-paths` | 系统规格/性能数据搜索路径，逗号分隔；`default` 表示内置路径 | 否 | 路径列表 | 内置 | `default,/opt/aic/systems` |
| `debug` | 开启调试日志 | 否 | 开关（flag） | 关闭 | （加上该 flag 即启用） |
| `no-color` | 关闭 ANSI 彩色输出 | 否 | 开关（flag） | 关闭 | （加上该 flag 即启用） |
| `generator-dynamo-version` | 生成部署配置时使用的 Dynamo 版本（后端版本由其映射） | 否 | 版本字符串 | `None` | `1.2.0` |
| `generated-config-version` | 生成部署配置使用的后端版本（覆盖默认） | 否 | 版本字符串 | `None` | `0.18.0` |
| `generator-set` | 生成器键值覆盖，可重复传入 | 否 | `KEY=VALUE` | 无 | `--generator-set K8sConfig.k8s_pvc_name=my-pvc` |

**参数书写约定**：

1. 全部选项使用 argparse 长选项形式 `--名称 值`（开关类选项只写 `--名称`）。
2. 选项名与上表"名称"列一致（文档省略前导 `--`）；例如 `total-gpus` 对应命令行 `--total-gpus 8`。
3. **无请求头、无 Body、无鉴权字段**；认证鉴权不适用于本 CLI 接口。

---

## 6.3 公共响应参数

> 对应原 API 文档"公共响应头参数"。CLI 无 HTTP 响应头，**公共响应参数**指每次成功执行后，标准输出摘要与推荐结果中固定出现的回显/结果字段（agg、disagg 两张结果表及 `best_config_topn.csv` 均携带）。

表 6 2 公共响应参数说明

| 名称 | 描述 | 示例 |
|------|------|------|
| Model | 回显模型路径/ID（请求 `--model-path`） | `/workspace/model_configs/qwen_3_8_config.json` |
| Total GPUs | 回显集群总卡数预算（请求 `--total-gpus`） | `8` |
| Agg Top Configurations | **PD 聚合场景下分布式策略推荐结果表**（agg 模式 top-n，按 `tokens/s/gpu_cluster` 降序） | 见 6.4.1 |
| Disagg Top Configurations | **PD 分离场景下分布式策略推荐结果表**（disagg 模式 top-n，按 `tokens/s/gpu_cluster` 降序） | 见 6.4.2 |
| Best Experiment Chosen | 全局最优实验选择（agg 与 disagg 吞吐对比结论） | `agg at 123.45 tokens/s/gpu (disagg 1.05x better)` |
| Best Throughput / Per-GPU / Per-User | 总吞吐、每卡吞吐（`tokens/s/gpu_cluster`）、单用户吞吐（`tokens/s/user`） | `987.6` / `123.45` / `33.33` |
| TTFT / TPOT / Request Latency | 最优方案预测延迟（ms）：`request_latency = ttft + tpot×(osl-1)` | `1520.0` / `58.7` / `7451.0` |
| concurrency / request_rate | 推荐方案并发度与请求速率（req/s） | `64` / `12.3` |
| total_gpus / replicas / parallel | 方案占用总卡数、副本数、并行策略串 | `8` / `1` / `tp8pp1dp1etp1ep1` |
| Pareto Frontier | Pareto 前沿图（x=`tokens/s/user`，y=`tokens/s/gpu_cluster`；终端 plotext 图，落盘时另存 PNG） | 字符图 / `pareto_frontier.png` |
| Exit Code | 进程退出码，见 [6.5](#65-公共响应码参数) | `0` |

### 6.3.1 响应传递方式

| 响应通道 | 内容 | 说明 |
|----------|------|------|
| 标准输出（stdout/stderr 日志） | Summary 框 + Agg/Disagg Top Configurations 表 + Pareto 图 | 由 `log_final_summary` 打印；受 `--no-color` / `--debug` 影响 |
| 结果文件（`--save-dir` 指定时） | CSV / YAML / PNG | 由 `save_results` 落盘，目录结构见 [6.4.3](#643-落盘文件结构响应体) |
| 进程退出码 | `0` / `1` / `2` | 见 [6.5](#65-公共响应码参数) |

---

## 6.4 响应内容详述

### 6.4.1 聚合场景响应：Agg Top Configurations

PD 聚合（agg）场景：同一实例同时承担 prefill 与 decode。推荐结果表标题形如：

```text
agg Top Configurations: (Sorted by tokens/s/gpu)
```

表 6 3 Agg 推荐结果字段说明

| 字段 | 描述 | 示例 |
|------|------|------|
| Rank | 方案排名（1 起，按 `tokens/s/gpu_cluster` 降序） | `1` |
| backend | 推理后端 | `vllm-ascend` |
| tokens/s/gpu | 每卡吞吐（表头简称，计算列为 `tokens/s/gpu_cluster`，含副本与装箱折算） | `123.45` |
| tokens/s/user | 单用户体感吞吐 `1000/tpot` | `16.95` |
| req/s | 集群请求速率 `request_rate` | `12.30` |
| TTFT | 预测首 Token 时延（ms），须 ≤ 请求 `ttft` | `1520.0` |
| request_latency | 端到端请求时延（ms） | `7451.0` |
| concurrency | 并发请求数 | `64` |
| total_gpus (used) | 方案实际占用总卡数 `replicas × gpus/replica` | `8` |
| replicas | 副本数 | `2` |
| gpus/replica | 单副本卡数 `tp×pp×dp` | `4` |
| gpus/worker | 单 worker 卡数 | `4` |
| parallel | 并行策略缩写 `tp{T}pp{P}dp{D}etp{ET}ep{EP}` | `tp4pp1dp1etp1ep4` |
| bs | 批大小 | `32` |
| power_w | 平均功率（W，有功耗数据时输出） | `380.0` |

**过滤规则**：先按 `tpot ≤ 请求 tpot`（若指定 `--request-latency` 则改按 `request_latency ≤ 目标`）过滤，再按 `tokens/s/gpu_cluster` 降序取 `top-n`。无满足约束的方案时输出：

```text
No configurations for agg met the tpot constraint.
```

### 6.4.2 分离场景响应：Disagg Top Configurations

PD 分离（disagg）场景：prefill 与 decode 使用独立 worker 池。表标题形如：

```text
disagg Top Configurations: (Sorted by tokens/s/gpu)
```

表 6 4 Disagg 推荐结果字段说明（在 Agg 公共列基础上）

| 字段 | 描述 | 示例 |
|------|------|------|
| (p)workers / (d)workers | prefill / decode worker 数 | `1` / `1` |
| (p)gpus/worker / (d)gpus/worker | prefill / decode 单 worker 卡数 | `4` / `4` |
| (p)parallel / (d)parallel | prefill / decode 并行策略 | `tp4pp1dp1etp1ep4` / `tp4pp1dp1etp1ep4` |
| (p)bs / (d)bs | prefill / decode 批大小 | `32` / `64` |
| 其余 Rank、tokens/s、TTFT、concurrency、total_gpus、replicas、power_w 等 | 同 Agg 表公共语义 | — |

> 要求：`--total-gpus ≥ 2` 才会构建并输出 disagg 结果；`< 2` 时告警并跳过 disagg。

### 6.4.3 落盘文件结构（响应体）

指定 `--save-dir results` 时，结果写入如下目录（后缀为 **0～1000000 随机整数**，非时间戳）：

```text
results/
└── {model名}_{system}_{backend}_isl{isl}_osl{osl}_ttft{ttft}_tpot{tpot}_{随机数}/
    ├── pareto_frontier.png              # 全局 Pareto 对比图
    ├── agg/
    │   ├── best_config_topn.csv         # 聚合场景 top-n 推荐（≤ top-n 行）
    │   ├── pareto.csv                   # 完整 Pareto 前沿
    │   ├── exp_config.yaml              # 本次实验 TaskConfig 快照
    │   └── top1/ … topN/                # 每个推荐方案
    │       └── generator_config.yaml    # 可下发的部署生成配置
    └── disagg/
        └── （结构同 agg/）
```

**`best_config_topn.csv` / `pareto.csv` 主要列（公共列 + 场景列）**：

| 场景 | 主要列 |
|------|--------|
| agg | `model, isl, osl, prefix, concurrency, request_rate, bs, global_bs, ttft, tpot, request_latency, seq/s, seq/s/gpu, tokens/s, tokens/s/gpu, tokens/s/user, tokens/s/gpu_cluster, num_total_gpus, tp, pp, dp, moe_tp, moe_ep, parallel, memory, backend, version, system, power_w, ...` |
| disagg | 在 agg 列基础上，将并行/内存/后端拆为 `(p)*` / `(d)*` 前缀，并增加 `(p)workers, (d)workers, (p)seq/s/worker, (d)seq/s/worker` 等 |

> 完整列定义见 `sdk/common.py` 中 `ColumnsAgg` / `ColumnsDisagg`。

**成功响应示例（标准输出节选）**：

```text
********************************************************************************
*                       aiconfigurator Final Results                          *
********************************************************************************
  Input Configuration & SLA Target:
    Model: /workspace/model_configs/qwen_3_8_config.json (is_moe: False)
    Total GPUs: 8
    Best Experiment Chosen: agg at 123.45 tokens/s/gpu (disagg 1.05x better)
  ...
  agg Top Configurations: (Sorted by tokens/s/gpu)
  +------+---------+--------------+---------------+ ...
  | Rank | backend | tokens/s/gpu | tokens/s/user | ...
  +------+---------+--------------+---------------+ ...
  |  1   | vllm-ascend | 123.45  | 16.95         | ...
  +------+---------+--------------+---------------+ ...
  ...
  disagg Top Configurations: (Sorted by tokens/s/gpu)
  ...
********************************************************************************
```

---

## 6.5 公共响应码参数

> 对应原 API 文档"公共响应码"：CLI 以 **进程退出码（Exit Code）** 作为所有调用方均需判断的公共响应码。

表 6 5 公共响应码参数说明

| 响应码 | 含义 | 备注 |
|--------|------|------|
| `0` | 成功 | 参数解析、搜索与（可选）落盘均成功；即使个别实验无可行方案，只要仍有成功实验，也返回 0 并附警告 |
| `1` | 业务失败 | 如：backend/system 不匹配、性能数据库缺失、`--systems-paths` 非法、全部实验无结果等 |
| `2` | 参数/语法错误 | argparse 解析失败（缺必填项、choices 非法、`--model-path` 校验失败）；generator 版本解析失败亦为 2 |

---

## 6.6 错误信息

### 6.6.1 错误码

表 6 6 错误码说明

| 错误码（退出码） | 错误信息（关键片段） | 描述 | 抛出模块 |
|------------------|----------------------|------|----------|
| 2 | `the following arguments are required: --model-path/--model, ...` | 缺少必填参数 | argparse |
| 2 | `unrecognized arguments: ...` | 出现 default 模式不支持的选项（如 `--target-concurrency`） | argparse |
| 2 | `invalid choice: '...' (choose from 'trtllm', 'sglang', 'vllm', 'vllm-ascend', 'auto')` | `--backend` 取值非法 | argparse |
| 2 | `invalid choice: '...' (choose from 'SILICON', 'HYBRID', 'EMPIRICAL', 'SOL')` | `--database-mode` 取值非法 | argparse |
| 2 | `'...' is not a valid HuggingFace model path or local path with config.json` | `--model-path` 校验失败 | `cli/main.py`（`_validate_model_path`） |
| 2 | `Directory '...' does not contain a config.json file.` | 模型目录缺少 `config.json` | `cli/main.py` |
| 1 | `Invalid --systems-paths: each entry must be an existing directory.` | 系统搜索路径含不存在目录 | `sdk/perf_database.py` |
| 1 | `Backend {backend} is not supported for system {system}. Supported backends: ...` | 当前 system 不支持该 backend | `cli/main.py`（`_ensure_backend_version_available`） |
| 1 | `No perf database for system=... backend=... version=...` + `Available versions: ...` | 指定版本无性能数据库 | `cli/main.py` |
| 1 | `No backends available for system ...` | `--backend auto` 展开后无可用后端 | `cli/main.py` |
| 1 | `total_gpus must be greater than 2 for disagg, got ...` | 构建 disagg 任务时 GPU 数不足 | `sdk/task.py` |
| 1 | `No successful experiment runs to compare.` | 全部实验失败/无结果（SILICON 模式可能附带建议改用 HYBRID 的提示） | `cli/main.py` |
| 1 | `Unsupported mode: ...` | mode 非法（内部保护） | `cli/main.py` |
| 2 | `Failed to resolve backend version for generator_dynamo_version=...` | 生成配置时版本映射失败 | `cli/report_and_save.py` |

**相关警告（不改变退出码）**：

| 警告信息（关键片段） | 含义 |
|----------------------|------|
| `Skipping disagg since it requires at least 2 GPUs.` | `--total-gpus < 2`，跳过 disagg，仅输出 agg |
| `Experiment {name} returned no results. Possible causes: (1) TTFT/TPOT constraints are too tight ...` | 单实验无满足 SLA 的可行配置（约束过紧 / 显存不足 / 性能数据缺失） |
| `No configurations for {agg\|disagg} met the {tpot\|request latency} constraint.` | 结果表阶段仍无满足约束的行 |
| `Failed to save results` | 落盘异常（打印堆栈，进程仍可能以 0 退出） |

### 6.6.2 错误信息返回格式

> CLI 无统一 JSON 错误体；错误信息按固定文本结构写入标准错误/日志流。

表 6 7 错误信息返回格式

| 参数名（结构） | 描述 |
|----------------|------|
| 日志级别前缀 | 如 `ERROR:` / `WARNING:` / argparse 的 `error:` |
| 错误描述 | 一句话说明失败原因（见 6.6.1 "错误信息"列） |
| 修复建议 / 上下文 | 可选：`Fix:` 建议、`Available versions:`、`Supported backends:` 等枚举提示 |
| 失败明细列表 | 全部实验失败时逐条列出：`-> {exp_name}: {reason}` |
| 退出码 | 进程结束时的 Exit Code（0/1/2），见 6.5 |

argparse 参数类错误额外打印 `usage:` 用法块。

### 6.6.3 错误返回示例

**示例 1：性能数据库缺失（退出码 1）**

```text
ERROR: No perf database for system=ascend_910b backend=vllm version=0.14.0
Fix: check --backend / --backend-version, or run with --backend-version omitted to use latest.
Available versions: 0.18.0
```

```bash
echo $?   # 1
```

**示例 2：模型路径非法（退出码 2）**

```text
usage: aic-npu default [-h] --model-path MODEL_PATH --total-gpus TOTAL_GPUS --system SYSTEM ...
aic-npu default: error: argument --model-path/--model: '/bad/path' is not a valid HuggingFace model path or local path with config.json.
```

```bash
echo $?   # 2
```

**示例 3：SLA 过紧导致无结果（业务失败，退出码 1）**

```text
WARNING: Experiment agg returned no results. Possible causes: (1) TTFT/TPOT constraints are too tight (ttft=1600, tpot=60) ... (2) model does not fit ... (3) no perf data ...
ERROR: No successful experiment runs to compare.
-> agg: no feasible configuration under constraints
-> disagg: no feasible configuration under constraints
```

```bash
echo $?   # 1
```

---

## 6.7 default 接口请求参数详表

> 单列 **default 子命令专属参数** 的完整规格，便于与 6.2 公共参数对照查阅。定义于 `_add_default_mode_arguments`（`cli/main.py`）。

表 6 8 default 专属请求参数

| 名称 | 类型 | 必填 | 默认值 | 取值约束 | 接口作用 |
|------|------|------|--------|----------|----------|
| `model-path` / `model` | string | 是 | 无 | 本地 `config.json` 目录/文件，或 HF id（白名单/可解析） | 加载模型算子图（如 Qwen3-8B → dense/MOE 结构） |
| `total-gpus` | int | 是 | 无 | agg ≥0；disagg ≥2 | 集群卡数预算；`tokens/s/gpu_cluster` 整机折算 |
| `system` | string | 是 | 无 | 须存在对应 YAML 与数据目录；本仓库为 `ascend_910b` | 硬件规格（显存/带宽/FLOPS/HCCL 开销） |
| `decode-system` | string | 否 | 同 `system` | 同 `system` | disagg decode 侧异构硬件 |
| `backend` | enum | 否 | `trtllm` | `trtllm` \| `sglang` \| `vllm` \| `vllm-ascend` \| `auto` | 调度步分解与并行过滤规则；示例取 `vllm-ascend` |
| `backend-version` | string | 否 | latest | 须存在于 `{data_dir}/{backend}/{version}` | 选择算子性能数据版本（如 `0.18.0`） |
| `database-mode` | enum | 否 | `SILICON` | `SILICON` \| `HYBRID` \| `EMPIRICAL` \| `SOL` | 取数策略；示例 `HYBRID`＝实测+SOL 兜底 |
| `isl` | int | 否 | `4000` | 正整数 | 输入长度；prefill token、KV、搜索网格 |
| `osl` | int | 否 | `1000` | 正整数 | 输出长度；decode 步数、`request_latency` |
| `ttft` | float | 否 | `2000` | ms | SLA 上界：过滤 `ttft ≤ 目标` |
| `tpot` | float | 否 | `30` | ms | SLA 上界：过滤 `tpot ≤ 目标`（内部仍按网格搜索） |
| `request-latency` | float | 否 | `None` | ms | 启用端到端延迟轴，展开多组 (ttft,tpot) 约束 |
| `prefix` | int | 否 | `0` | 非负整数 | 前缀缓存，减少有效 prefill token |
| `nextn` | int | 否 | `0` | 非负整数 | MTP draft 数；>0 时激活内存 ×(nextn+1) |
| `nextn-accept-rates` | string | 否 | `0.85,0.3,0,0,0` | 5 个逗号分隔 float | MTP 接受率，折算加速比 |
| `enable-chunked-prefill` | flag | 否 | 关 | — | 细粒度 `ctx_tokens` 扫描 |
| `top-n` | int | 否 | `5` | 正整数 | agg/disagg 各保留条数（示例 `10`） |
| `save-dir` | string | 否 | `None` | 可写目录 | 结果落盘根目录（示例 `results`） |

**SLA 语义（接口行为约定）**：

1. `--ttft` / `--tpot` 是搜索的 **硬过滤上界**，不是单点目标；内部会把 tpot 展开为网格并对每组约束搜索一轮，最终只保留满足用户上界的方案。
2. 若指定 `--request-latency L`，选型主轴切换为 `request_latency`，并按 `tpot_i = (L - ttft_i) / (osl - 1)` 展开约束。
3. default 模式 **始终同时** 尝试构建 `agg` 与 `disagg`（disagg 需 `total-gpus ≥ 2`），对应两张 Top Configurations 响应表。

---

## 6.8 关键接口的作用和说明

### 6.8.1 接口定位

`aic-npu default` 是系统的 **一站式配置推荐接口**：输入模型、硬件、后端、SLA 与卡数预算，输出满足约束的 **PD 聚合（Agg）与 PD 分离（Disagg）Top-N 分布式部署方案**。不执行真实推理，属于静态性能估算与 Pareto 选型。

### 6.8.2 处理流程（接口内部时序）

```text
CLI argv
  → argparse 解析（6.2 / 6.7 请求参数）
  → build_default_task_configs
       · 校验 system × backend × backend-version 性能库
       · 加载 model-path → 模型算子图
       · 构建 TaskConfig("agg") / TaskConfig("disagg")
  → _execute_task_configs → TaskRunner
       · 并行枚举 (tp,pp,dp,moe_tp,moe_ep)
       · batch × ctx_tokens 扫描，逐算子查 PerfDatabase
       · OOM 剪枝 + ttft/tpot SLA 硬过滤
  → pick_default（固定 GPU 预算下最大化吞吐）
       · tokens/s/gpu_cluster 折算 → Pareto 前沿 → 按并行分组取 top-n
  → 响应输出
       · stdout：Summary + Agg/Disagg Top Configurations + Pareto 图（6.3 / 6.4）
       · --save-dir：CSV / YAML / PNG（6.4.3）
       · 退出码（6.5）
```

### 6.8.3 关键参数对结果的影响

| 参数 | 对推荐结果的影响 |
|------|------------------|
| `total-gpus` | 决定可容纳的副本数与整机折算吞吐；过小可能无 disagg 方案 |
| `isl` / `osl` | 改变 prefill/decode 负载配比与 `balance_score`，影响最优 `bs`、`parallel` |
| `ttft` / `tpot` | 直接决定可行域大小：越紧方案越少，甚至返回"无满足约束配置" |
| `backend` | 改变并行合法域（如 vllm 禁止 `moe_tp>1 且 moe_ep>1`）与步延迟公式 |
| `database-mode` | `SILICON` 结果可复现但覆盖有限；`HYBRID` 对新模型/未测点更稳（示例命令采用 HYBRID） |
| `top-n` | 仅控制输出条数，不改变搜索深度（搜索内部另有固定 `top_k`） |
| `save-dir` | 控制是否落盘；不传时仅 stdout 响应 |

### 6.8.4 使用示例

**（1）基准示例 — Qwen3-8B + Ascend 910B + vllm-ascend**

```bash
aic-npu default \
  --model-path /workspace/model_configs/qwen_3_8_config.json \
  --total-gpus 8 \
  --system ascend_910b \
  --backend vllm-ascend \
  --database-mode HYBRID \
  --isl 128 \
  --osl 128 \
  --ttft 1600 \
  --tpot 60 \
  --top-n 10 \
  --save-dir results
```

预期：退出码 `0`；stdout 输出 Agg/Disagg 两张 Top Configurations 表；`results/` 下生成带随机后缀的结果目录。

**（2）仅查看帮助（接口自描述）**

```bash
aic-npu default --help
```

**（3）不落盘、关闭颜色（便于脚本采集 stdout）**

```bash
aic-npu default --model-path /workspace/model_configs/qwen_3_8_config.json \
  --total-gpus 8 --system ascend_910b --backend vllm-ascend \
  --database-mode HYBRID --isl 128 --osl 128 --ttft 1600 --tpot 60 \
  --top-n 10 --no-color
echo $?    # 判断 6.5 公共响应码
```

### 6.8.5 接口约束与兼容性说明

1. **与上游 GPU 版差异**：入口名为 `aic-npu`（非 `aiconfigurator`）；本仓库 `--system` 实际可用值为 `ascend_910b`；`--backend` 示例为 `vllm-ascend`（性能数据目录 `systems/data/ascend_910b/vllm-ascend/0.18.0`）。
2. **未暴露的编程参数**：如 `target_concurrency`（load-match 选型）目前仅存在于内部 API `_execute_task_configs(...)`，**不是** default 子命令的 CLI 选项；误传会得到退出码 `2` 的 `unrecognized arguments`。
3. **幂等性**：相同请求参数可重复执行；`--save-dir` 下因随机目录后缀不会覆盖历史结果。
4. **性能数据依赖**：接口正确性依赖 `--system × --backend × --backend-version` 对应算子性能库存在，否则在任务构建阶段以退出码 `1` 失败。
