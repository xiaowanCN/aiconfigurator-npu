# aiconfigurator-npu 仓库架构与计算框架总览

> 本文档以如下命令为线索，梳理整个仓库架构，以及命令背后完整的**配置搜索 / 性能估算计算框架**：
>
> ```bash
> aic-npu default --model-path /workspace/model_configs/qwen_3_8_config.json  --total-gpus 8 --system ascend_910b --backend vllm-ascend --database-mode HYBRID --isl <> --osl <> --ttft 1600 --tpot 60 --top-n 10 --save-dir results 
> ```
>
> 相关文档：
> - `AIC_NPU_DEFAULT_COMMAND_ARCHITECTURE.md` — default 命令分阶段执行细节
> - `DISAGG_PARETO_ARCHITECTURE.md` — 分离式（prefill/decode）Pareto 搜索
> - `HCCL_ALLREDUCE_PERF_DATA_GUIDE.md` — 通信算子性能数据采集
> - `GLM5_ADAPTATION_DESIGN.md` / `GLM5_CONFIG_SEARCH_PLAN.md` — GLM-5 适配

---

## 目录

- [0. 命令与本仓库现状的差异（必读）](#0-命令与本仓库现状的差异必读)
- [1. 仓库整体架构](#1-仓库整体架构)
- [2. 计算框架总览（一图流）](#2-计算框架总览一图流)
- [3. 五层计算框架详解](#3-五层计算框架详解)
  - [3.1 输入层：SLA / 模型 / 系统 / 后端](#31-输入层sla--模型--系统--后端)
  - [3.2 建模层：模型算子图 + 硬件规格 + 性能数据库](#32-建模层模型算子图--硬件规格--性能数据库)
  - [3.3 搜索层：并行 × batch × ctx_tokens 三维枚举](#33-搜索层并行--batch--ctx_tokens-三维枚举)
  - [3.4 估算层：延迟 / 吞吐 / 内存 / 功耗公式](#34-估算层延迟--吞吐--内存--功耗公式)
  - [3.5 选型层：SLA 过滤 + Pareto + top-n / load-match](#35-选型层sla-过滤--pareto--top-n--load-match)
- [4. 端到端数据流](#4-端到端数据流)
- [5. 关键符号索引（file:line）](#5-关键符号索引fileline)
- [6. 术语与输出列说明](#6-术语与输出列说明)

---

## 0. 命令与本仓库现状的差异（必读）

本仓库是 **NVIDIA Dynamo AIConfigurator 的 Ascend NPU fork**，系统与性能数据面向 `ascend_910b`。上述命令是**上游 GPU 版写法**，直接在本仓库执行会遇到 4 处不匹配：

| 项 | 命令写法 | 本仓库现状 | 影响 |
|---|---|---|---|
| 入口名 | `aiconfigurator` | console script 为 **`aic-npu`**（`pyproject.toml:33`） | `main()` 会剥掉 `cli` 前缀（`cli/main.py`），`aiconfigurator cli default ...` 与 `aic-npu default ...` 语义等价 |
| 系统 | `--system ascend_910b` | 系统 YAML 仅有 **`ascend_910b`**（`src/aiconfigurator_npu/systems/ascend_910b.yaml`）。`SupportedSystems` 含 `a100_sxm`，无 `a800_sxm`（`sdk/common.py:328-337`） | `--system` 是自由字符串，argparse 能通过；但 `get_supported_databases()` / `get_database()` 找不到 `a800_sxm.yaml`，会在 `_ensure_backend_version_available` 或 DB 加载处失败 |
| backend + version | `--backend vllm --backend-version 0.14.0` | 性能数据目录为 **`systems/data/ascend_910b/vllm-ascend/0.18.0`**。无 `vllm/0.14.0` | 即使 system 正确，仍会因 `{data_dir}/{backend}/{version}` 不存在而 `SystemExit(1)` |
| `--target-concurrency 4` | 当作 CLI 旗标 | **不是 default 模式 argparse 选项**。只存在于 `_execute_task_configs` / `process_experiment_result` / `pick_load_match` 的函数参数中 | argparse 直接报 `unrecognized arguments` |

`main()` 在 default 模式下的实际调用（`cli/main.py:1487-1491`）**未转发** `target_concurrency`：

```python
_, best_configs, pareto_fronts, _, _ = _execute_task_configs(
    task_configs, args.mode, top_n=args.top_n,
)
# 未传 target_request_rate / target_concurrency / max_total_gpus
```

因此 default 模式目前**只会走 `pick_default`**（固定 GPU 预算下最大化吞吐），不会走 `pick_load_match`（满足目标并发下最小化 GPU）。

### 本仓库等价可跑命令

```bash
aic-npu default \
  --model Qwen/Qwen3-30B-A3B --total-gpus 8 \
  --system ascend_910b --backend vllm_ascend \
  --isl 128 --osl 512 --ttft 1000 --tpot 30 \
  --top-n 10 --backend-version 0.18.0 \
  --database-mode HYBRID --save-dir results
```

> 说明：
> - `--model` 是 `--model-path` 的别名（`cli/main.py:106-109`）。
> - `Qwen/Qwen3-30B-A3B` 在 `DefaultHFModels` 中（`sdk/common.py:295`）；本地 `model_configs/` 只有 `Qwen3-235B-A22B` 与 `GLM-5`，30B-A3B 会走 HF cache / 下载。
> - `--target-concurrency` 的**完整计算语义**在 [3.5 节](#35-选型层sla-过滤--pareto--top-n--load-match) 说明（load-match 选型），只是 CLI 尚未暴露该旗标，需通过编程 API `_execute_task_configs(..., target_concurrency=4)` 触发。

下文按「命令修正为本仓库可跑形态」描述完整计算框架；框架本身对上游 GPU 版与本 NPU fork 是同构的。

---

## 1. 仓库整体架构

```
aiconfigurator-npu/
├── pyproject.toml                  # 包名 aiconfigurator_npu；入口 aic-npu = aiconfigurator_npu.cli.main:main
├── src/aiconfigurator_npu/         # ★ 主 Python 包
│   ├── cli/                        # 入口层：参数解析、实验编排、结果打印/落盘
│   │   ├── main.py                 #   configure_parser / main / build_default_task_configs / _execute_task_configs
│   │   ├── api.py                  #   程序化 API（generate_naive_config 等）
│   │   ├── utils.py                #   process_experiment_result / merge_experiment_results_by_mode
│   │   ├── report_and_save.py      #   log_final_summary（表格 / Pareto 图）、save_results
│   │   └── example.yaml            #   exp 模式示例
│   ├── sdk/                        # ★ 核心计算 SDK
│   │   ├── task.py                 #   TaskConfig / TaskConfigFactory / TaskRunner
│   │   ├── config.py               #   ModelConfig / RuntimeConfig
│   │   ├── models.py               #   BaseModel / GPT / LLAMA / MOE / DEEPSEEK ... 算子图
│   │   ├── operations.py           #   GEMM / Attention / MoE / MoEDispatch / AllReduce 等 op
│   │   ├── perf_database.py        #   PerfDatabase：按 system/backend/version 加载插值表
│   │   ├── inference_session.py    #   InferenceSession / DisaggInferenceSession
│   │   ├── inference_summary.py    #   单点/搜索结果摘要、OOM 检查
│   │   ├── backends/               #   trtllm / sglang / vllm 后端：调度步分解 + 指标公式
│   │   ├── pareto_analysis.py      #   agg_pareto / disagg_pareto / get_pareto_front / load-match
│   │   ├── picking.py              #   pick_default / pick_load_match / pick_autoscale
│   │   ├── performance_result.py   #   结果容器
│   │   ├── common.py               #   枚举、列 schema、DefaultHFModels、SupportedSystems
│   │   └── utils.py                #   并行枚举、HF config 解析、模型配置加载
│   ├── generator/                  # naive 配置生成（无 sweep）、本地 generator bridge
│   ├── systems/                    # ★ 内置系统规格 + 算子性能数据库
│   │   ├── ascend_910b.yaml        #   910B 硬件规格
│   │   ├── support_matrix.csv      #   模型 × 量化 × 后端支持矩阵
│   │   └── data/ascend_910b/vllm-ascend/0.18.0/   # 实测/插值算子数据
│   └── logging_utils.py
├── systems/                        # 外部/工作区系统数据（--systems-paths 可指向这里）
│   └── ascend_910b_aiconfigurator/ascend_910b.yaml
├── model_configs/                  # 预下载 HF config.json（离线）
│   ├── Qwen--Qwen3-235B-A22B_config.json
│   └── zai-org--GLM-5_config.json
├── collector/                      # ★ 真机算子微基准采集（GEMM / Attention / MoE / MLA）
├── tools/                          # 数据转换、兼容性/算子诊断脚本
├── docs/                           # 设计与架构文档
├── results/                        # 历史结果归档
├── docker/                         # 容器构建（arm64 / Ascend 910B）
└── backup/                         # 旧工作区快照
```

### 分层职责

| 层 | 目录 | 职责 |
|---|---|---|
| 入口层 | `src/.../cli` | 解析 SLA/模型/系统/backend → 构造 `TaskConfig` → 跑 `TaskRunner` → 挑 top-n → 打印/落盘 |
| 计算层 | `src/.../sdk` | 模型算子图、性能数据库查询、后端搜索、Pareto、选型 |
| 数据层 | `src/.../systems` + 根 `systems/` | 硬件规格 YAML + `{data_dir}/{backend}/{version}` 算子性能表 |
| 模型层 | `model_configs/` | 离线 HF `config.json`（`_iter_model_config_dirs` 优先包内再 CWD） |
| 采集层 | `collector/` | NPU 真机算子延迟采集 → 灌进 perf DB |
| 工具层 | `tools/` | 采集后处理、诊断、补丁 |

**数据生产闭环**：`collector/` 在 NPU 上跑微基准 → CSV/`*_perf.txt` → `tools/convert_to_aiconfigurator.py` → `systems/data/{system}/{backend}/{version}/` → `PerfDatabase` 插值查询 → 配置搜索。

---

## 2. 计算框架总览（一图流）

整个系统本质是一个 **「算子级性能数据库 + 模型算子图 + 多维配置枚举 + SLA 硬过滤 + Pareto 选型」** 的静态分析器，**不跑真实推理**。

```
                 ┌──────────────────────────────────────────────────────────┐
   输入层        │  SLA: isl/osl/ttft/tpot(/request-latency/prefix)          │
                 │  模型: model_path → HF config → MOEModel 算子图           │
                 │  硬件: system YAML (mem_capacity/BW/FLOPS/nccl_mem)       │
                 │  后端: backend + backend-version → 性能数据目录           │
                 │  预算: total_gpus / top_n (/target_concurrency)           │
                 └───────────────────────────┬──────────────────────────────┘
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
   建模层        │  PerfDatabase = system_spec + gemm/attn/moe/nccl 插值表    │
                 │  MOEModel    = Embedding + Attn + Router + Dispatch + FFN │
                 │  Backend     = trtllm / vllm / sglang 步分解与指标公式    │
                 └───────────────────────────┬──────────────────────────────┘
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
   搜索层        │  并行枚举 (tp,pp,dp,moe_tp,moe_ep)                         │
                 │    × batch_size  b ∈ [1..1024] 非均匀网格                 │
                 │    × ctx_tokens  (chunked prefill 步粒度)                  │
                 │  [disagg 再 × (prefill worker, decode worker) 配比]       │
                 └───────────────────────────┬──────────────────────────────┘
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
   估算层        │  对每个候选点：                                            │
                 │    延迟: 逐 op 查 DB → step 分解 → ttft / tpot            │
                 │    吞吐: seq/s, tokens/s, tokens/s/user, request_latency  │
                 │    内存: weights/TP + act + KV + HCCL + reserved → OOM?   │
                 │    功耗: Σ(op energy) / Σ(op latency)                     │
                 └───────────────────────────┬──────────────────────────────┘
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
   选型层        │  SLA 硬过滤 (ttft≤target, tpot≤target) + OOM 剔除          │
                 │  → 按 seq/s 排序取 top_k                                  │
                 │  → tokens/s/gpu_cluster 重标定（整机副本数）              │
                 │  → Pareto 前沿 (tokens/s/user ↔ tokens/s/gpu_cluster)     │
                 │  → pick_default: 固定 GPU 预算最大化吞吐，取 top_n         │
                 │    (或 pick_load_match: 满足 target_concurrency 最少 GPU) │
                 └───────────────────────────┬──────────────────────────────┘
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
   输出层        │  summary box + prettytable(top-n) + Pareto 曲线            │
                 │  (--save-dir 时落盘 CSV / 图 / 生成部署配置)              │
                 └──────────────────────────────────────────────────────────┘
```

---

## 3. 五层计算框架详解

### 3.1 输入层：SLA / 模型 / 系统 / 后端

#### 3.1.1 CLI 参数（default 模式）

定义于 `_add_default_mode_arguments`（`cli/main.py:105-183`）：

| 参数 | 默认 | 进入计算的位置 | 语义 |
|---|---|---|---|
| `--model-path` / `--model` | 必填 | `get_model_config_from_model_path` → `MOEModel` | HF id / 本地 config.json |
| `--total-gpus` | 必填 | `pick_default` 的集群折算、`num_gpu_per_worker` | 部署总卡数预算 |
| `--system` | 必填 | `get_database(system, ...)` | 硬件规格 + 数据目录键 |
| `--decode-system` | =system | disagg decode 侧 DB | 异构 prefill/decode |
| `--backend` | trtllm | `get_backend` + 数据目录 + 并行过滤 | 推理框架 |
| `--backend-version` | latest | 数据目录 `{data_dir}/{backend}/{version}` | 性能数据版本 |
| `--database-mode` | SILICON | `PerfDatabase` 查询策略 | SILICON/HYBRID/EMPIRICAL/SOL |
| `--isl` / `--osl` | 4000 / 1000 | prefill token 数、KV、`request_latency` | 输入/输出序列长度 |
| `--ttft` / `--tpot` | 2000 / 30 | **SLA 硬过滤**（`ttft<=x and tpot<=y`） | 延迟上界（ms） |
| `--request-latency` | None | `enumerate_ttft_tpot_constraints` 展开多组 (ttft,tpot) | 端到端延迟 SLA |
| `--prefix` | 0 | 有效 prefill token = `(isl-prefix)*b` | 前缀缓存命中长度 |
| `--nextn` / `--nextn-accept-rates` | 0 / `0.85,0.3,0,0,0` | MTP 缩放因子、激活内存 ×(nextn+1) | 投机解码 |
| `--enable-chunked-prefill` | off | `ctx_tokens` 枚举粒度 | 分块预填充 |
| `--top-n` | 5 | `pick_*` 最终保留条数 | 输出配置条数 |

**SLA 语义要点**：
- `--tpot 30` **不是单点测量**，而是搜索的**上界约束**。`TaskRunner.run_agg` 会把 tpot 展开成搜索网格 `list(range(1,20)) + list(range(20,300,5))`，对每组 `(ttft, tpot)` 约束各搜一轮；用户给的 `ttft/tpot` 只用于**过滤**结果。
- 若给 `--request-latency L`，则调用 `enumerate_ttft_tpot_constraints(osl, L, ttft)`（`sdk/utils.py:188`）展开多组约束：`tpot_i = (L - ttft_i) / (osl - 1)`，picking 改用 `request_latency` 轴。

#### 3.1.2 模型：Qwen3-30B-A3B 如何被建模

加载链（`sdk/utils.py`）：

```
get_model_config_from_model_path(model_path)          # utils.py:964 @cache
  → _load_model_config_from_model_path                # utils.py:909 @cache
      优先级:
      1) 本地目录 + config.json
      2) 直接指向 *config.json 文件
      3) HF cache 命中
      4) DefaultHFModels 预下载目录 (model_configs/)
      5) _download_hf_config (联网)
      + 推断 quant 字段 / hf_quant_config
  → _parse_hf_config_json(raw)                        # utils.py:490
  → {"architecture","layers","n","n_kv","d","hidden_size",
     "inter_size","vocab","context","topk","num_experts",
     "moe_inter_size","extra_params","raw_config"}
```

对 `Qwen/Qwen3-30B-A3B`：

| 步骤 | 结果 | 依据 |
|---|---|---|
| DefaultHFModels 命中 | `"Qwen/Qwen3-30B-A3B"` | `sdk/common.py:295` |
| HF architecture | `Qwen3MoeForCausalLM` | HF config（同类 235B 见 `model_configs/Qwen--Qwen3-235B-A22B_config.json`） |
| family 映射 | `"Qwen3MoeForCausalLM": "MOE"` | `sdk/common.py:359` |
| MoE 字段解析 | `topk=num_experts_per_tok`；`num_experts=num_experts|num_local_experts|n_routed_experts`；`moe_inter_size=moe_intermediate_size` | `sdk/utils.py:543-545` |
| Qwen3 额外参数 | `extra_params = {"architecture", "use_qk_norm": True}` | `sdk/utils.py:635-637` |
| 实例化模型类 | `get_model(...) → MOEModel` | `sdk/models.py:160` / `MOEModel` :884 |

`MOEModel` 的并行约束与算子图（`sdk/models.py:884-943`）：

```python
assert tp_size * attention_dp_size == moe_tp_size * moe_ep_size   # 全局 EP 域一致
assert num_experts >= moe_ep_size
_mtp_scale_factor = (1/(1+calc_expectation(nextn, rates)) * (nextn+L)/L) if nextn>0 else 1.0
_power_law_alpha = 1.2     # 负载不均衡：power_law_1.2 → query_moe
# 算子序列: Embedding → ContextAttention/GenerationAttention (QK-norm)
#          → router GEMM → MoEDispatch (All2All) → MoE (gated FFN)
#          → logits GEMM ...
```

MoE 权重（`sdk/operations.py:269-279`）：

```
weights = hidden * moe_inter * num_experts * quant_bytes * num_gemms
          // moe_ep_size // moe_tp_size
num_gemms = 3 if is_gated else 2    # gate/up/down
```

#### 3.1.3 系统：`ascend_910b.yaml`

`src/aiconfigurator_npu/systems/ascend_910b.yaml`（根 `systems/ascend_910b_aiconfigurator/` 下有同内容副本）：

```yaml
data_dir: data/ascend_910b
gpu:
  mem_bw: 2e12                          # 2 TB/s
  mem_bw_empirical_scaling_factor: 0.85
  mem_empirical_constant_latency: 3e-6
  mem_capacity: 68719476736             # 64 GiB
  float16_tc_flops: 256e12              # BF16 256 TFLOPS
  int8_tc_flops: 512e12
  fp8_tc_flops: 512e12
  power: 400
node:
  num_gpus_per_node: 8
  inter_node_bw: 25e9                   # 100Gb/s RoCE
  intra_node_bw: 56e9                   # HCCS
  pcie_bw: 32e9
  p2p_latency: 10e-6
misc:
  nccl_mem: {1: 0, 2: 342MB, 4: 392MB, 8: 392MB}   # HCCL 通信库开销
  other_mem: 3.5GB                      # runtime / driver 预留
  nccl_version: '2.26.0'
```

加载入口（`sdk/perf_database.py`）：
- `set_systems_paths`（:49）— 支持 `default` + 额外路径
- `get_supported_databases`（:93）— 扫描 `{systems_root}/*.yaml` × `data_dir/{backend}/{version}`（跳过 `INCOMPLETE.txt`）
- `get_latest_database_version`（:149）
- `get_database(system, backend, version)`（:236）— 读 YAML → `PerfDatabase`，全局 cache

#### 3.1.4 后端与版本：两层影响

**(A) PerfDatabase 数据选择**（延迟/能量插值表）

`PerfDatabase.__init__`（`sdk/perf_database.py:2102-2137`）：

```
system_spec = yaml(systems/{system}.yaml)
data_dir    = systems_root / system_spec["data_dir"] / backend / version
nccl_dir    = systems_root / data_dir_root / "nccl" / system_spec["misc"]["nccl_version"]
load_*_data(gemm, context_attention, generation_attention, moe, nccl, mla, ...)
```

本仓库实际数据布局：

```
src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/
├── context_attention_perf.txt / generation_attention_perf.txt
├── gemm_perf.txt / moe_perf.txt
├── MatMulV2.csv / QuantBatchMatmulV3.csv
├── GroupedMatmul_MoE_BF16.csv / GroupedMatmul_MoE_W8A8.csv
├── FusedInferAttentionScore.csv / FusedInferAttentionScore_Decode.csv
```

查询 API（二维/多维插值）：`query_gemm` :3541、`query_context_attention` :3809、`query_generation_attention` :3958、`query_moe` :4748、`query_nccl` :4636、`query_mem_op` :5238、`query_p2p` :5531。

`database_mode` 决定数据来源策略：

| 模式 | 含义 |
|---|---|
| `SILICON`（默认） | 只用真机实测数据，完全可复现 |
| `HYBRID`（推荐新模型） | SILICON 覆盖不到的点用 SOL+经验系数补齐 |
| `EMPIRICAL` | 仅 SOL + 经验系数 |
| `SOL` | 仅理论 Speed-of-Light |

**(B) Backend 类**（搜索调度与指标公式）

`backends/factory.get_backend(name)` → `TRTLLMBackend` / `SGLangBackend` / `VLLMBackend`。并行枚举的后端过滤（`sdk/utils.py:154-169`）：

| 后端 | 过滤规则 |
|---|---|
| trtllm | 拒绝 `dp>1 and tp>1` |
| sglang | wideep 只要 EP；非 wideep 不要 EP |
| **vllm / vllm_ascend** | **拒绝 `moe_tp>1 and moe_ep>1`**（MoE 权重不能同时 TP+EP 切） |

此外：激活内存经验系数 `c_dict` 随 `model_family` 与 `tp_size` 变化；mix/genonly 步延迟合成、chunked prefill 的 `ctx_tokens` 粒度均按 backend 特化。

---

### 3.2 建模层：模型算子图 + 硬件规格 + 性能数据库

三类核心对象：

1. **`BaseModel` / `MOEModel`**（`sdk/models.py`）— 把 HF 结构翻译成**有序算子列表** `context_ops` / `generation_ops`。每个 op 自己知道：
   - `get_latency(database, tokens, ...)` → 查插值表
   - `get_weights()` → 参数字节数（按 tp/pp/moe_ep/moe_tp 分片）
   - `get_energy(...)`
2. **`PerfDatabase`**（`sdk/perf_database.py`）— 算子延迟/能量的多维查表 + 插值 + SOL 兜底。
3. **`BaseBackend` 子类**（`sdk/backends/`）— 把算子序列组织成**推理步**（context step / generation step / mix step / genonly step），并给出吞吐/并发/内存公式。

> 关键思想：**性能 = Σ(算子延迟)**，算子延迟来自离线微基准插值；框架只负责把「模型 × 并行 × batch × 序列长」映射成算子形状，再去查表。

---

### 3.3 搜索层：并行 × batch × ctx_tokens 三维枚举

#### 3.3.1 并行枚举

`enumerate_parallel_config`（`sdk/utils.py:101-185`）：

```
for tp, pp, dp, moe_tp, moe_ep in product(tp_list, pp_list, dp_list, moe_tp_list, moe_ep_list):
    if is_moe:
        keep 条件:
          dp*tp*pp ∈ num_gpu_list          # 每 replica 卡数合法
          dp*tp == moe_tp*moe_ep            # MoE 全局专家域一致
          + 后端过滤规则（见 3.1.4-B）
        → [tp, pp, dp, moe_tp, moe_ep]
    else:
        if tp*pp ∈ num_gpu_list: → [tp, pp, 1, 1, 1]
```

`num_gpu_list` 来自 `TaskConfigFactory` 的 `num_gpu_per_worker`（由 `--total-gpus` 与 replica 规则推导）。

对 `--total-gpus 8` + Qwen3-30B-A3B（MOE）+ vllm，合法并行点形如：
`tp8pp1dp1etp1ep8`、`tp4pp1dp1etp1ep4`、`tp2pp1dp2etp1ep4`、`tp1pp1dp1etp1ep1` …
（受限于 `moe_tp=1` 或 `moe_ep=1` 二者取一，且 `tp*dp == moe_tp*moe_ep`）。

#### 3.3.2 batch_size × ctx_tokens 二维 sweep

`VLLMBackend.find_best_agg_result_under_constraints`（`sdk/backends/vllm_backend.py:390-495`）：

```python
b_list = [1..15] + [16,20,...,28] + [32,40,...,56] + [64,80,...,240]
         + [256,288,...,480] + [512,768] + [1024]        # 非均匀，小 batch 密
ctx_tokens_list = _get_ctx_tokens_list_for_agg_sweep(isl, ctx_stride=512,
                                                     enable_chunked_prefill)
for b in b_list:
    for ctx_tokens in ctx_tokens_list:
        if b - ceil(ctx_tokens/isl) < 1:  break           # 至少 1 个 gen 请求
        balance_score = isl * b / ctx_tokens / osl         # prefill/gen 配比
        if balance_score > 1: ...去重/校正...
        summary = run_agg(b, ctx_tokens)                   # 步分解 → 指标
        if summary.check_oom(): break                      # 内层单调，OOM 提前停
        if tpot <= tpot_sla and ttft <= ttft_sla: keep     # SLA 硬过滤
sort by seq/s desc; head(top_k=10)
```

- `ctx_tokens`：一步 context 里混合处理的 token 数（chunked prefill 粒度）。
- `balance_score = isl * b / ctx_tokens / osl`：>1 表示 prefill 相对 decode 过重，需配比校正。
- **单调性剪枝**：`b`、`ctx_tokens` 增大 → 内存单调增，故内层 OOM 即 `break`。

#### 3.3.3 top-n 与 target-concurrency 的搜索语义

| 概念 | 位置 | 语义 |
|---|---|---|
| `top_k=10` | pareto 内部 sweep | 每个并行点内部保留的最优 batch/ctx 组合数 |
| `top_n`（`--top-n 10`） | `pick_default` / `pick_load_match` | **最终表格/CSV 保留的配置条数**（agg/disagg 各 top_n） |
| `target_concurrency` | `pick_load_match` | **load-match 选型**：在 SLA 下服务 N 个并发请求，**最少要多少 GPU / 多少 replica**。结果列 `replicas_needed`, `total_gpus_needed`, `load_served_pct`。比较指标默认 `1/total_gpus_needed`，GPU 封顶时改 `tokens/s/gpu_cluster` |

---

### 3.4 估算层：延迟 / 吞吐 / 内存 / 功耗公式

#### 3.4.1 静态单点（`run_static`）

`BaseBackend.run_static`（`sdk/backends/base_backend.py:185-346`）— 对算子图累加：

```
ttft            = sum(context_latency_dict)
tpot            = generation_latency / max(osl-1, 1)
request_latency = ttft + tpot * (osl-1)
seq_s           = global_bs / request_latency * 1000 * pp_size
seq_s_gpu       = seq_s / (tp * pp * dp)
tokens_s        = seq_s * osl
tokens_s_gpu    = tokens_s / (tp * pp * dp)
tokens_s_user   = 1000 / tpot
num_total_gpus  = tp * pp * dp
parallel        = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}"
```

#### 3.4.2 agg 混合步（vLLM/TRTLLM 主路径）

`VLLMBackend.run_agg`（`sdk/backends/vllm_backend.py:249-369`）把请求生命周期切成 **mix step**（同时含 ctx/gen token）与 **genonly step**：

```
tpot = (mix_step_latency * num_mix_steps_for_tpot_calc
        + genonly_step_latency * num_genonly_steps)
       / (num_mix_steps_for_tpot_calc + num_genonly_steps)
# trtllm: num_mix_steps_for_tpot_calc = max(1, num_mix_steps-3)   # 3 步流水线经验修正

output_throughput = 1000 / (num_mix_steps * mix + num_genonly_steps * genonly)
                    * b * (osl - 1)
scale_factor      = pp_size * attention_dp_size
output_throughput *= scale_factor
concurrency       = b * scale_factor
request_rate      = output_throughput / (osl - 1)
tokens_s_user     = 1000 / tpot
request_latency   = ttft + tpot * max(osl - 1, 0)
num_total_gpus    = tp * pp * dp
balance_score     = isl * b / ctx_tokens / osl
```

#### 3.4.3 显存模型

`TRTLLMBackend._get_memory_usage`（`sdk/backends/trtllm_backend.py:497-597`，vLLM 复用）：

```
weights = Σ op.get_weights() / pp_size          # 已含 tp/moe_tp/moe_ep 分片

num_tokens = (isl - prefix) * b                  # gen 时 b*beam

# 激活（经验 c_dict，随 family / tp 变）
#   GPT:   {1:10, 2:6,  4:5,  8:5}
#   LLAMA: {1:11, 2:6.5,4:5,  8:5}
#   MOE / DEEPSEEK: {1:22, 2:13, 4:10, 8:10}
activations = 2 * num_tokens * (num_heads * head_size) * c_dict[min(tp, 8)]
activations = max(activations, 70MB)
if nextn > 0: activations *= (nextn + 1)         # MTP draft 额外激活

# KV cache（不除 pp，各 stage 都要持有）
kv_heads_per_gpu    = ceil(num_kv_heads / tp)
kvcache_per_token   = kv_heads_per_gpu * head_size * num_layers * 2
# DEEPSEEK/MLA: kvcache_per_token = num_layers * 576
kvcache = (b * isl + b * beam * osl) * kvcache_quant_bytes * kvcache_per_token

total_GB = (weights + activations + kvcache
            + nccl_mem[min(tp,8)] + other_mem) / 2^30
```

OOM 判定：`summary.set_memory_and_check_oom(memory, system_spec["gpu"]["mem_capacity"])`（`base_backend.py:343`）。

#### 3.4.4 Disagg 速率匹配

`_build_disagg_summary_dict`（`sdk/picking.py:46-151`）：

```
seq_s = min( prefill_seq_s * prefill_workers * 0.90,   # _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR
             decode_seq_s * decode_workers  * 0.92 )   # _RATE_MATCHING_DECODE_DEGRADATION_FACTOR
num_total_gpus = prefill_gpus * p_workers + decode_gpus * d_workers
tokens_s       = seq_s * osl
request_latency = prefill_ttft + decode_tpot * max(osl-1, 0)
disagg_power_avg = (p_power * ttft + d_power * decode_time) / (ttft + decode_time)
# pick_autoscale 模式: ttft_corrected = ttft * 1.8  (_AUTOSCALE_TTFT_CORRECTION_FACTOR)
```

#### 3.4.5 功耗

```
power_avg = Σ(op_energy_wms) / Σ(op_latency_ms)      # 加权平均功率（瓦）
agg/disagg 场景按 mix/genonly 或 prefill/decode 时间加权
```

---

### 3.5 选型层：SLA 过滤 + Pareto + top-n / load-match

#### 3.5.1 结果处理分发

`process_experiment_result`（`cli/utils.py:52-76`）：

```python
load_match = target_request_rate is not None or target_concurrency is not None
use_request_latency = target_request_latency > 0
x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
if load_match: pick_load_match(...)   # picking.py:279
else:          pick_default(...)      # picking.py:183
```

#### 3.5.2 `pick_default`（default 模式实际路径）

`sdk/picking.py:183-271`：

1. **集群折算**：
   ```
   tokens/s/gpu_cluster = tokens/s/gpu
                          * floor(total_gpus / num_total_gpus)   # 整机能塞下的副本数
                          * num_total_gpus / total_gpus
   ```
2. **Pareto 前沿**：`get_pareto_front(df, x_axis_col, "tokens/s/gpu_cluster", maximize_x=not use_request_latency, maximize_y=True)`
   - 默认 x=`tokens/s/user`（用户体感速度），y=`tokens/s/gpu_cluster`（集群效率）→ 右上角双赢。
3. **组内选优**：`get_best_configs_under_tpot_constraint`（或 `..._under_request_latency_constraint`），`group_by = "(d)parallel" if disagg else "parallel"`，取 `top_n`。
4. 返回 `best_config_df, best_throughput, best_latencies, pareto_frontier_df`。

> 对本例 `--total-gpus 8`：若某配置 `num_total_gpus=4`，则可跑 2 个 replica；`tokens/s/gpu_cluster = tokens/s/gpu * 2 * 4 / 8 = tokens/s/gpu`。若 `num_total_gpus=3`（理论），`floor(8/3)=2`，利用率折算为 `2*3/8=0.75`，惩罚装不满的并行切分。

#### 3.5.3 `pick_load_match`（`target-concurrency` 路径）

`sdk/picking.py:279-362` + `get_best_configs_for_target_load`（`sdk/pareto_analysis.py:630+`）：

```
replicas_needed   = ceil(target_concurrency / concurrency_per_replica)
total_gpus_needed = replicas_needed * num_total_gpus
load_served_pct   = min(1.0, 实际可服务并发 / target_concurrency)
```

- 目标：**满足 `target_concurrency=4`（或 target_request_rate）时最小化 GPU 数**。
- 排序指标：默认 `1/total_gpus_needed`；若 GPU 数封顶（`max_total_gpus`）则改 `tokens/s/gpu_cluster`。
- 输出列额外含 `replicas_needed / total_gpus_needed / load_served_pct`，报告打印 `Target Concurrency`。

#### 3.5.4 `pick_autoscale`（disagg 专用）

`sdk/picking.py:370-478`：prefill 按 `TTFT×1.8`、decode 按 TPOT **独立选型**，再笛卡尔积出 disagg 配置（`workers=1`），按 `tokens/s/gpu` 排序取 top_n。

#### 3.5.5 最终输出

`log_final_summary`（`cli/report_and_save.py:274+`）：

1. **summary box**：输入配置 + SLA、chosen experiment、各模式 best throughput。
2. **prettytable 表格**（top-n 行）：`tokens/s/gpu_cluster, tokens/s/user, cluster_request_rate, ttft, request_latency, concurrency(=c×replicas), total_gpus, replicas, parallel, bs, power_w` 等。
3. **Pareto 前沿图**：终端 plotext + 可选 matplotlib。
4. `--save-dir` 时 `save_results` 落盘 CSV / 图 / 生成部署配置。

---

## 4. 端到端数据流

```
[CLI argv]
  aic-npu default --model-path Qwen/Qwen3-30B-A3B --total-gpus 8
    --system ascend_910b --backend vllm_ascend --backend-version 0.18.0
    --isl 128 --osl 512 --ttft 1000 --tpot 30 --top-n 10
        │
        ▼
[cli/main.py:1411]  main() → argparse → Namespace
        │  set_systems_paths(args.systems_paths)
        ▼
[cli/main.py:602]   build_default_task_configs()
  ├─ get_supported_databases / _ensure_backend_version_available
  │    systems/*.yaml × data_dir/{backend}/{version}
  └─ TaskConfig("agg")  +  TaskConfig("disagg")          # total_gpus ≥ 2
        │  TaskContext → TaskConfigFactory.create
        │  get_model_family → MOE  (Qwen3MoeForCausalLM)
        │  validate() vs PerfDatabase.supported_quant_mode
        ▼
[cli/main.py:916]   _execute_task_configs() → TaskRunner.run
        │
        ├─ RuntimeConfig(isl=128, osl=512, ttft=1000,
        │                tpot=网格[1..20,20..300 step5], prefix=0)
        ├─ get_database(ascend_910b, vllm_ascend, 0.18.0) → PerfDatabase
        │    system_spec(mem_capacity=64GiB, mem_bw=2TB/s, BF16=256TFLOPS,
        │                nccl_mem, other_mem)
        │    + gemm / attn / moe / nccl 插值表
        ├─ ModelConfig(quant, nextn, ...)
        ├─ enumerate_parallel_config → [[tp,pp,dp,moe_tp,moe_ep], ...]
        │    约束: dp*tp*pp==8, dp*tp==moe_tp*moe_ep,
        │          vllm 禁 (moe_tp>1 且 moe_ep>1)
        ▼
[sdk/pareto_analysis.py:26]  agg_pareto / disagg_pareto
  for each parallel_config:
    get_model(...) → MOEModel 算子图
      (Attention + Router + MoEDispatch + MoE + logits)
    InferenceSession / DisaggInferenceSession
      find_best_agg_result_under_constraints
        for b in b_list:                       # 1..1024 非均匀
          for ctx_tokens in ctx_tokens_list:    # chunked prefill 粒度
            backend.run_agg → mix/genonly 步
              → ttft / tpot / throughput / memory / power
            OOM? → break 内层
            SLA: ttft≤1000 and tpot≤30        → 候选
        sort seq/s desc → head(top_k=10)
  concat → ColumnsAgg / ColumnsDisagg DataFrame
        │
        ▼
[cli/utils.py:52]   process_experiment_result
  pick_default(pareto_df, total_gpus=8, serving_mode,
               target_tpot=30, target_request_latency=None, top_n=10)
    tokens/s/gpu_cluster = tokens/s/gpu
                           * floor(8 / num_total_gpus) * num_total_gpus / 8
    get_pareto_front(x=tokens/s/user, y=tokens/s/gpu_cluster)
    get_best_configs_under_tpot_constraint(group_by=parallel, top_n=10)
  # 若接线 target_concurrency=4 → pick_load_match
  #   → replicas_needed / total_gpus_needed / load_served_pct
        │
        ▼
[cli/report_and_save.py:274]   log_final_summary + 可选 save_results
  summary box + prettytable(top-n) + Pareto 曲线(plotext/matplotlib)
  列: tokens/s/gpu_cluster, tokens/s/user, ttft, request_latency,
      concurrency, total_gpus, replicas, parallel, bs, power_w, ...
```

---

## 5. 关键符号索引（file:line）

| 符号 | 位置 |
|---|---|
| `main` / `configure_parser` | `src/aiconfigurator_npu/cli/main.py:1411` / `:460` |
| `_add_default_mode_arguments` | `src/aiconfigurator_npu/cli/main.py:105` |
| `build_default_task_configs` | `src/aiconfigurator_npu/cli/main.py:602` |
| `_execute_task_configs` | `src/aiconfigurator_npu/cli/main.py:916` |
| `_ensure_backend_version_available` | `src/aiconfigurator_npu/cli/main.py:543` |
| `process_experiment_result` | `src/aiconfigurator_npu/cli/utils.py:52` |
| `merge_experiment_results_by_mode` | `src/aiconfigurator_npu/cli/utils.py:146` |
| `log_final_summary` / `save_results` | `src/aiconfigurator_npu/cli/report_and_save.py:274` |
| `TaskConfig` / `TaskConfigFactory` / `TaskRunner` | `src/aiconfigurator_npu/sdk/task.py:616` / `:249` / `:1053` |
| `TaskRunner.run` / `run_agg` / `run_disagg` | `src/aiconfigurator_npu/sdk/task.py:1351` / `:1068` / `:1153` |
| `ModelConfig` / `RuntimeConfig` | `src/aiconfigurator_npu/sdk/config.py` |
| `get_model` / `get_model_family` / `check_is_moe` | `src/aiconfigurator_npu/sdk/models.py:160` / `:449` / `:458` |
| `calc_expectation` / `BaseModel` / `MOEModel` | `src/aiconfigurator_npu/sdk/models.py:483` / `:499` / `:884` |
| `MoE.get_weights` / `MoEDispatch` | `src/aiconfigurator_npu/sdk/operations.py:269` / `:322` |
| `enumerate_parallel_config` | `src/aiconfigurator_npu/sdk/utils.py:101` |
| `enumerate_ttft_tpot_constraints` | `src/aiconfigurator_npu/sdk/utils.py:188` |
| `_parse_hf_config_json` | `src/aiconfigurator_npu/sdk/utils.py:490` |
| `_load_model_config_from_model_path` / `get_model_config_from_model_path` | `src/aiconfigurator_npu/sdk/utils.py:909` / `:964` |
| `get_database` / `get_supported_databases` | `src/aiconfigurator_npu/sdk/perf_database.py:236` / `:93` |
| `PerfDatabase` / `query_moe` | `src/aiconfigurator_npu/sdk/perf_database.py:2058` / `:4748` |
| `BaseBackend.run_static` | `src/aiconfigurator_npu/sdk/backends/base_backend.py:185` |
| `TRTLLMBackend._get_memory_usage` | `src/aiconfigurator_npu/sdk/backends/trtllm_backend.py:497` |
| `VLLMBackend.run_agg` / `find_best_agg_result_under_constraints` | `src/aiconfigurator_npu/sdk/backends/vllm_backend.py:249` / `:390` |
| `InferenceSession` / `DisaggInferenceSession` | `src/aiconfigurator_npu/sdk/inference_session.py:26` / `:141` |
| `agg_pareto` / `disagg_pareto` / `get_pareto_front` | `src/aiconfigurator_npu/sdk/pareto_analysis.py:26` / `:176` |
| `get_best_configs_for_target_load` | `src/aiconfigurator_npu/sdk/pareto_analysis.py:630` |
| `pick_default` / `pick_load_match` / `pick_autoscale` | `src/aiconfigurator_npu/sdk/picking.py:183` / `:279` / `:370` |
| `_build_disagg_summary_dict` | `src/aiconfigurator_npu/sdk/picking.py:46` |
| `DefaultHFModels` / `SupportedSystems` / `ARCHITECTURE_TO_MODEL_FAMILY` | `src/aiconfigurator_npu/sdk/common.py:277` / `:328` / `:343` |
| 系统规格 | `src/aiconfigurator_npu/systems/ascend_910b.yaml` |
| 性能数据 | `src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/` |
| 离线模型配置 | `model_configs/*.json` |

---

## 6. 术语与输出列说明

### 6.1 关键术语

| 术语 | 含义 |
|---|---|
| **ISL / OSL** | Input / Output Sequence Length，请求输入/输出 token 数 |
| **TTFT** | Time To First Token，首 token 延迟（≈ prefill 完成时间） |
| **TPOT** | Time Per Output Token，每输出 token 延迟（decode 步平均） |
| **request_latency** | `ttft + tpot × (osl-1)`，端到端请求延迟 |
| **agg / disagg** | 聚合服务（同一实例做 prefill+decode） / 分离式服务 |
| **ctx_tokens** | 一步 context（mix step）中混合处理的 token 数 |
| **mix / genonly step** | 同时含 prefill+decode 的步 / 纯 decode 步 |
| **balance_score** | `isl×b / ctx_tokens / osl`，prefill 与 decode 配比指标 |
| **tokens/s/user** | `1000/tpot`，单用户体感生成速度 |
| **tokens/s/gpu_cluster** | 整机折算后的集群每卡吞吐（含副本数与装箱利用率） |
| **parallel** | `tp{T}pp{P}dp{D}etp{ET}ep{EP}` 并行策略缩写 |
| **SILICON/HYBRID/EMPIRICAL/SOL** | 性能数据模式：实测 / 实测+SOL 兜底 / 经验 / 理论 |

### 6.2 输出 DataFrame 主要列

**ColumnsAgg（聚合模式）**：`model, isl, osl, prefix, concurrency, request_rate, bs, global_bs, ttft, tpot, seq/s, seq/s/gpu, tokens/s, tokens/s/gpu, tokens/s/user, request_latency, num_total_gpus, tp, pp, dp, moe_tp, moe_ep, parallel, gemm, kvcache, fmha, moe, comm, memory, backend, version, system, power_w, balance_score, num_ctx_reqs, num_gen_reqs, num_tokens, ctx_tokens, gen_tokens`

**ColumnsDisagg（分离模式）**：在 Agg 基础上拆成 `(p)*` / `(d)*` 前缀（prefill / decode 各自的并行、量化、内存、backend），外加 `(p)workers` / `(d)workers`、`(p)seq/s/worker` / `(d)seq/s/worker`。

**选型附加列**（load-match）：`replicas_needed, total_gpus_needed, load_served_pct`。

**选型附加列**（pick_default）：`tokens/s/gpu_cluster`。

---

## 附：计算框架的三句话总结

1. **不跑推理，只查表**：把「模型结构 × 并行策略 × batch × 序列长」翻译成算子形状，去离线微基准数据库（`systems/data/...`）插值得到延迟/能量，累加成 TTFT/TPOT。
2. **枚举 + 硬约束剪枝**：三维（并行 × batch × ctx_tokens）枚举，OOM 单调剪枝 + TTFT/TPOT SLA 硬过滤，得到可行域。
3. **Pareto 选型**：在可行域上取 `(tokens/s/user, tokens/s/gpu_cluster)` Pareto 前沿；default 模式按固定 GPU 预算最大化吞吐取 top-n；load-match 模式按目标并发最小化 GPU（`target-concurrency`）。
