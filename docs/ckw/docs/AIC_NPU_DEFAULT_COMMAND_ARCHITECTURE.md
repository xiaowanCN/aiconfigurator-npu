# aic-npu default 命令执行流程与框架架构详解

## 目录

- [1. 命令概述](#1-命令概述)
- [2. 整体架构图](#2-整体架构图)
- [3. 详细执行流程](#3-详细执行流程)
  - [阶段 1: CLI 入口与参数解析](#阶段-1-cli-入口与参数解析)
  - [阶段 2: 构建 TaskConfig](#阶段-2-构建-taskconfig)
  - [阶段 3: TaskConfig 初始化与配置工厂](#阶段-3-taskconfig-初始化与配置工厂)
  - [阶段 4: 执行任务](#阶段-4-执行任务)
  - [阶段 5: Pareto 分析核心](#阶段-5-pareto-分析核心)
  - [阶段 6: 性能数据库 (HYBRID 模式)](#阶段-6-性能数据库-hybrid-模式)
  - [阶段 7: 结果处理与输出](#阶段-7-结果处理与输出)
- [4. 关键数据流](#4-关键数据流)
- [5. 核心算法: Pareto 前沿搜索](#5-核心算法-pareto-前沿搜索)
- [6. 输出目录结构](#6-输出目录结构)

---

## 1. 命令概述

### 命令参数解析

```bash
aic-npu default \
  --model-path /workspace/model_configs/qwen_3_8_config.json \  # 模型配置文件
  --total-gpus 8 \                                               # 总GPU数量
  --system ascend_910b \                                         # 系统类型(Ascend 910B)
  --backend vllm-ascend \                                        # 后端(vllm-ascend)
  --database-mode HYBRID \                                       # 数据库模式(混合模式)
  --isl 4000 \                                                   # 输入序列长度
  --osl 500 \                                                    # 输出序列长度
  --ttft 1200 \                                                  # TTFT目标(ms)
  --tpot 50 \                                                    # TPOT目标(ms)
  --top-n 10 \                                                   # 输出前10个配置
  --save-dir results                                             # 保存目录
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model-path` | `/workspace/model_configs/qwen_3_8_config.json` | Qwen3-8B 模型配置文件路径 |
| `--total-gpus` | `8` | 部署使用的总 GPU 数量 |
| `--system` | `ascend_910b` | 目标硬件系统 (Ascend 910B NPU) |
| `--backend` | `vllm-ascend` | 推理后端 (vllm-ascend) |
| `--database-mode` | `HYBRID` | 性能数据库模式 (实测 + SOL 估算) |
| `--isl` | `4000` | 输入序列长度 (Input Sequence Length) |
| `--osl` | `500` | 输出序列长度 (Output Sequence Length) |
| `--ttft` | `1200` | 首 Token 延迟目标 (Time To First Token, ms) |
| `--tpot` | `50` | 每 Token 延迟目标 (Time Per Output Token, ms) |
| `--top-n` | `10` | 输出前 N 个最优配置 |
| `--save-dir` | `results` | 结果保存目录 |

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           aic-npu CLI 入口层                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  main.py: main() → parse_args() → _run_default_mode()                      │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TaskConfig 构建层                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  build_default_task_configs()                                               │  │
│  │    ├── 创建 agg TaskConfig (聚合模式)                                        │  │
│  │    └── 创建 disagg TaskConfig (分离模式)                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TaskConfigFactory 配置工厂                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  TaskConfigFactory.create(ctx)                                              │  │
│  │    ├── Layer 1: _agg_defaults_layer / _disagg_defaults_layer (默认配置)      │  │
│  │    ├── Layer 2: YAML patch 层 (用户自定义覆盖)                                │  │
│  │    ├── Layer 3: Profile 层 (量化配置等)                                       │  │
│  │    └── Layer 4: _finalize_agg / _finalize_disagg (最终调整)                  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         _execute_task_configs 执行层                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  for exp_name, task_config in task_configs.items():                         │  │
│  │      TaskRunner.run(task_config)                                            │  │
│  │        ├── run_agg()     → agg_pareto()                                     │  │
│  │        └── run_disagg()  → disagg_pareto()                                  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Pareto 分析层                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  agg_pareto() / disagg_pareto()                                             │  │
│  │    ├── 枚举并行配置 (TP/PP/DP/MoE-TP/MoE-EP)                                │  │
│  │    ├── 遍历每种并行配置                                                       │  │
│  │    │   ├── get_model() → 获取模型定义                                         │  │
│  │    │   ├── get_backend() → 获取后端                                           │  │
│  │    │   ├── InferenceSession → 创建推理会话                                    │  │
│  │    │   └── find_best_agg_result_under_constraints() → 约束搜索               │  │
│  │    └── 合并结果，返回 Pareto DataFrame                                        │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         推理会话与模型层                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  InferenceSession / DisaggInferenceSession                                  │  │
│  │    ├── 模型权重估算 (Model.get_weights())                                    │  │
│  │    ├── 模型分析 (Model.analyze_model())                                      │  │
│  │    └── 性能估算 (Backend.query())                                            │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            性能数据库层                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  PerfDatabase                                                               │  │
│  │    ├── SILICON 模式: 使用实测数据                                             │  │
│  │    ├── HYBRID 模式: 实测 + SOL估算 (本命令使用)                               │  │
│  │    ├── EMPIRICAL 模式: SOL + 经验因子                                         │  │
│  │    └── SOL 模式: 理论峰值计算                                                 │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         结果处理与输出层                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  process_experiment_result()                                                │  │
│  │    ├── 选择最优配置 (top-n)                                                  │  │
│  │    ├── 生成 Pareto 前沿                                                      │  │
│  │    └── 保存结果 (CSV/YAML/PNG)                                               │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 详细执行流程

### 阶段 1: CLI 入口与参数解析

**入口文件**: `src/aiconfigurator_npu/cli/main.py`

```python
# 执行流程
main()
  → parse_args()           # 解析命令行参数
  → setup_logging()        # 配置日志系统
  → _run_default_mode(args)  # 进入 default 模式
```

**关键源码位置**:
- CLI 入口: `main.py:main()` (约第 1200 行)
- 参数定义: `main.py:_add_default_mode_arguments()` (约第 105 行)
- 模式验证: `main.py:_validate_model_path()` (约第 69 行)

---

### 阶段 2: 构建 TaskConfig

**函数**: `build_default_task_configs()` (main.py:602)

```python
def build_default_task_configs(
    model_path: str,           # "/workspace/model_configs/qwen_3_8_config.json"
    total_gpus: int,           # 8
    system: str,               # "ascend_910b"
    backend: str,              # "vllm-ascend"
    database_mode: str,        # "HYBRID"
    isl: int,                  # 4000
    osl: int,                  # 500
    ttft: float,               # 1200
    tpot: float,               # 50
    ...
) -> dict[str, TaskConfig]:
```

**执行流程**:

```
build_default_task_configs()
  │
  ├── 1. 验证后端可用性
  │      └── _ensure_backend_version_available("ascend_910b", "vllm-ascend")
  │          ├── 查询 support_matrix.csv
  │          └── 确认后端版本存在
  │
  ├── 2. 创建 agg TaskConfig
  │      └── TaskConfig(
  │            serving_mode="agg",
  │            model_path="/workspace/model_configs/qwen_3_8_config.json",
  │            system_name="ascend_910b",
  │            backend_name="vllm-ascend",
  │            total_gpus=8,
  │            isl=4000,
  │            osl=500,
  │            ttft=1200,
  │            tpot=50,
  │            database_mode="HYBRID"
  │          )
  │
  └── 3. 创建 disagg TaskConfig (如果 total_gpus >= 2)
         └── TaskConfig(
               serving_mode="disagg",
               ... # 同上参数
             )
```

**输出**: `{"agg": TaskConfig, "disagg": TaskConfig}`

---

### 阶段 3: TaskConfig 初始化与配置工厂

**类**: `TaskConfig` (task.py:616)

#### 3.1 创建 TaskContext

```python
ctx = TaskContext(
    serving_mode="agg",
    model_path="/workspace/model_configs/qwen_3_8_config.json",
    model_family="QWEN",           # 从模型配置推断
    system_name="ascend_910b",
    backend_name="vllm-ascend",
    isl=4000,
    osl=500,
    ttft=1200,
    tpot=50,
    total_gpus=8
)
```

#### 3.2 TaskConfigFactory.create(ctx)

配置工厂使用多层配置叠加机制:

```
TaskConfigFactory.create(ctx)
  │
  ├── Layer 1: 默认配置层
  │   │
  │   ├── agg 模式: _agg_defaults_layer()
  │   │   └── worker_config = {
  │   │         num_gpu_per_worker: [1, 2, 4, 8],
  │   │         tp_list: [1, 2, 4, 8],
  │   │         pp_list: [1],           # 默认禁用流水线并行
  │   │         dp_list: [1],           # 默认禁用数据并行
  │   │         moe_tp_list: [1],       # 非MoE模型
  │   │         moe_ep_list: [1]        # 非MoE模型
  │   │       }
  │   │
  │   └── disagg 模式: _disagg_defaults_layer()
  │       ├── prefill_worker_config = {
  │       │     num_gpu_per_worker: [1, 2, 4, 8],
  │       │     tp_list: [1, 2, 4, 8],
  │       │     ...
  │       │   }
  │       ├── decode_worker_config = {
  │       │     num_gpu_per_worker: [1, 2, 4, 8],
  │       │     tp_list: [1, 2, 4, 8],
  │       │     ...
  │       │   }
  │       ├── replica_config = {
  │       │     num_gpu_per_replica: [1, 2, 4, 8, 16, ...],
  │       │     max_gpu_per_replica: 128,
  │       │     max_prefill_worker: 32,
  │       │     max_decode_worker: 32
  │       │   }
  │       └── advanced_tuning_config = {
  │             prefill_latency_correction_scale: 1.1,
  │             decode_latency_correction_scale: 1.08,
  │             prefill_max_batch_size: 1,
  │             decode_max_batch_size: 512
  │           }
  │
  ├── Layer 2: YAML patch 层 (如果有用户自定义配置)
  │   └── 深度合并用户配置到默认配置
  │
  ├── Layer 3: Profile 层 (量化配置)
  │   └── 可选 profile: "fp8", "fp8_static", "float16", "nvfp4", "mxfp4"
  │       默认使用 float16 (无显式 profile)
  │
  └── Layer 4: Finalize 层
      └── _finalize_agg() / _finalize_disagg()
          └── 根据 total_gpus 限制 num_gpu_per_worker
              例如: total_gpus=8 时, num_gpu_per_worker 限制为 [1, 2, 4, 8]
```

#### 3.3 转换量化模式为枚举

```python
_convert_worker_config_to_enum()
  ├── gemm_quant_mode: str → GEMMQuantMode
  ├── moe_quant_mode: str → MoEQuantMode
  ├── kvcache_quant_mode: str → KVCacheQuantMode
  ├── fmha_quant_mode: str → FMHAQuantMode
  └── comm_quant_mode: str → CommQuantMode
```

#### 3.4 验证配置

```python
validate()
  ├── 检查后端支持 (vllm-ascend 是否支持 ascend_910b)
  ├── 验证量化模式与性能数据兼容性
  └── 检查模型是否适合当前后端
```

---

### 阶段 4: 执行任务

**函数**: `_execute_task_configs()` (main.py:916)

```python
def _execute_task_configs(
    task_configs: dict[str, TaskConfig],  # {"agg": ..., "disagg": ...}
    mode: str,                            # "default"
    top_n: int,                           # 10
) -> tuple:
```

**执行流程**:

```
_execute_task_configs(task_configs)
  │
  ├── 创建 TaskRunner 实例
  │
  ├── 遍历每个任务配置
  │   │
  │   ├── [1] 执行 agg 任务
  │   │   │
  │   │   ├── TaskRunner.run(task_config)
  │   │   │   │
  │   │   │   └── run_agg(task_config.config)
  │   │   │       │
  │   │   │       ├── Step 1: 创建 RuntimeConfig
  │   │   │       │   └── RuntimeConfig(
  │   │   │       │         isl=4000,
  │   │   │       │         osl=500,
  │   │   │       │         ttft=1200,
  │   │   │       │         tpot=[1,2,...,19,20,25,...,295],  # 搜索范围
  │   │   │       │         prefix=0
  │   │   │       │       )
  │   │   │       │
  │   │   │       ├── Step 2: 获取性能数据库
  │   │   │       │   └── PerfDatabase(
  │   │   │       │         system="ascend_910b",
  │   │   │       │         backend="vllm-ascend",
  │   │   │       │         mode="HYBRID"
  │   │   │       │       )
  │   │   │       │
  │   │   │       ├── Step 3: 创建 ModelConfig
  │   │   │       │   └── ModelConfig(
  │   │   │       │         gemm_quant_mode=float16,
  │   │   │       │         kvcache_quant_mode=float16,
  │   │   │       │         fmha_quant_mode=float16,
  │   │   │       │         moe_quant_mode=float16,
  │   │   │       │         comm_quant_mode=half
  │   │   │       │       )
  │   │   │       │
  │   │   │       ├── Step 4: 枚举并行配置
  │   │   │       │   └── enumerate_parallel_config(
  │   │   │       │         num_gpu_list=[1,2,4,8],
  │   │   │       │         tp_list=[1,2,4,8],
  │   │   │       │         pp_list=[1],
  │   │   │       │         dp_list=[1],
  │   │   │       │         moe_tp_list=[1],
  │   │   │       │         moe_ep_list=[1]
  │   │   │       │       )
  │   │   │       │       生成: [(1,1,1,1,1), (2,1,1,1,1), (4,1,1,1,1), (8,1,1,1,1)]
  │   │   │       │
  │   │   │       └── Step 5: 调用 agg_pareto()
  │   │   │
  │   │   └── 返回 {"pareto_df": DataFrame}
  │   │
  │   └── [2] 执行 disagg 任务
  │       │
  │       ├── TaskRunner.run(task_config)
  │       │   │
  │       │   └── run_disagg(task_config.config)
  │       │       │
  │       │       ├── Step 1: 创建 RuntimeConfig
  │       │       │
  │       │       ├── Step 2: 获取 prefill/decode 性能数据库
  │       │       │   ├── prefill_database = PerfDatabase("ascend_910b", "vllm-ascend")
  │       │       │   └── decode_database = PerfDatabase("ascend_910b", "vllm-ascend")
  │       │       │
  │       │       ├── Step 3: 创建 prefill/decode ModelConfig
  │       │       │
  │       │       ├── Step 4: 枚举 prefill/decode 并行配置
  │       │       │   ├── prefill_parallel_config_list
  │       │       │   └── decode_parallel_config_list
  │       │       │
  │       │       └── Step 5: 调用 disagg_pareto()
  │       │
  │       └── 返回 {"pareto_df": DataFrame}
  │
  └── 返回 (chosen_exp, best_configs, pareto_fronts, best_throughputs, best_latencies)
```

---

### 阶段 5: Pareto 分析核心

**文件**: `src/aiconfigurator_npu/sdk/pareto_analysis.py`

#### 5.1 agg_pareto() 执行流程

```python
def agg_pareto(
    model_path: str,
    runtime_config: RuntimeConfig,
    database: PerfDatabase,
    backend_name: str,
    model_config: ModelConfig,
    parallel_config_list: list[list[int]],
    enable_chunked_prefill: bool = False
) -> pd.DataFrame:
```

**详细流程**:

```
agg_pareto()
  │
  ├── 初始化结果 DataFrame (columns=ColumnsAgg)
  │
  ├── 遍历每种并行配置 (tp, pp, dp, moe_tp, moe_ep)
  │   │
  │   ├── 1. 获取模型定义
  │   │   └── get_model(model_path, model_config, backend_name)
  │   │       └── 返回 Qwen3_8B 模型实例
  │   │           ├── 模型架构: QwenForCausalLM
  │   │           ├── 参数量: ~8B
  │   │           └── 层数: 36
  │   │
  │   ├── 2. 获取后端实现
  │   │   └── get_backend("vllm-ascend")
  │   │       └── 返回 VllmBackend 实例
  │   │           └── 实现 query_gemm(), query_attention() 等接口
  │   │
  │   ├── 3. 创建推理会话
  │   │   └── InferenceSession(model, database, backend)
  │   │       ├── 加载模型权重信息
  │   │       └── 初始化性能查询接口
  │   │
  │   ├── 4. 搜索最优配置
  │   │   └── sess.find_best_agg_result_under_constraints(
  │   │         runtime_config=runtime_config,
  │   │         top_k=10,
  │   │         max_batch_size=512,
  │   │         ctx_stride=512
  │   │       )
  │   │       │
  │   │       ├── 遍历不同的 batch_size (1, 2, 4, 8, ..., 512)
  │   │       │
  │   │       ├── 对每个 batch_size:
  │   │       │   │
  │   │       │   ├── 计算 prefill 延迟
  │   │       │   │   └── model.analyze_model("prefill", bs, ctx_tokens)
  │   │       │   │       ├── GEMM 延迟
  │   │       │   │       ├── Attention 延迟
  │   │       │   │       ├── MoE 延迟 (如果是 MoE 模型)
  │   │       │   │       └── 通信延迟
  │   │       │   │
  │   │       │   ├── 计算 decode 延迟
  │   │       │   │   └── model.analyze_model("decode", bs, 1)
  │   │       │   │       ├── GEMM 延迟
  │   │       │   │       ├── Attention 延迟
  │   │       │   │       └── 通信延迟
  │   │       │   │
  │   │       │   └── 检查是否满足约束
  │   │       │       ├── TTFT = prefill_latency ≤ 1200ms?
  │   │       │       └── TPOT = decode_latency ≤ 50ms?
  │   │       │
  │   │       └── 返回满足约束的最优配置
  │   │
  │   └── 5. 收集结果到 DataFrame
  │       └── 包含字段:
  │           ├── tp, pp, dp, moe_tp, moe_ep
  │           ├── bs (batch_size)
  │           ├── ctx_tokens (context tokens)
  │           ├── ttft (ms)
  │           ├── tpot (ms)
  │           ├── tokens/s/user (用户吞吐)
  │           ├── tokens/s/gpu (GPU 吞吐)
  │           └── total_gpus
  │
  ├── 合并所有并行配置的结果
  │   └── pd.concat([results_df, result_df])
  │
  ├── 去重并排序
  │   └── results_df.sort_values(by="tokens/s/gpu", ascending=False)
  │
  └── 返回 Pareto DataFrame
```

#### 5.2 disagg_pareto() 执行流程

```python
def disagg_pareto(
    model_path: str,
    runtime_config: RuntimeConfig,
    prefill_database: PerfDatabase,
    prefill_backend_name: str,
    prefill_model_config: ModelConfig,
    prefill_parallel_config_list: list[list[int]],
    prefill_latency_correction_scale: float,
    decode_database: PerfDatabase,
    decode_backend_name: str,
    decode_model_config: ModelConfig,
    decode_parallel_config_list: list[list[int]],
    decode_latency_correction_scale: float,
    **kwargs
) -> pd.DataFrame:
```

**详细流程**:

```
disagg_pareto()
  │
  ├── 1. 创建 DisaggInferenceSession
  │   └── DisaggInferenceSession(
  │         model_path=model_path,
  │         prefill_database=prefill_database,
  │         decode_database=decode_database,
  │         prefill_backend=get_backend("vllm-ascend"),
  │         decode_backend=get_backend("vllm-ascend")
  │       )
  │
  ├── 2. 枚举 prefill/decode worker 组合
  │   └── 遍历所有 (prefill_config, decode_config) 组合
  │       ├── prefill_config: (tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1)
  │       ├── prefill_config: (tp=2, pp=1, dp=1, moe_tp=1, moe_ep=1)
  │       ├── ...
  │       └── decode_config: (tp=8, pp=1, dp=1, moe_tp=1, moe_ep=1)
  │
  ├── 3. 对每种组合:
  │   │
  │   ├── 计算 prefill 延迟
  │   │   └── prefill_latency = prefill_session.analyze_model("prefill", bs, isl)
  │   │       └── 应用 prefill_latency_correction_scale (默认 1.1)
  │   │
  │   ├── 计算 decode 延迟
  │   │   └── decode_latency = decode_session.analyze_model("decode", bs, 1)
  │   │       └── 应用 decode_latency_correction_scale (默认 1.08)
  │   │
  │   ├── 计算通信开销 (KV transfer)
  │   │   └── kv_transfer_latency = calculate_kv_transfer_latency(...)
  │   │
  │   └── 检查 TTFT/TPOT 约束
  │       ├── TTFT = prefill_latency + kv_transfer_latency ≤ 1200ms?
  │       └── TPOT = decode_latency ≤ 50ms?
  │
  ├── 4. 计算 replica 数量
  │   └── 根据 total_gpus 和每个 replica 的 GPU 数量
  │       ├── num_gpu_per_replica = prefill_gpus + decode_gpus
  │       └── num_replicas = total_gpus // num_gpu_per_replica
  │
  └── 返回 Pareto DataFrame
      └── 包含字段:
          ├── (p)tp, (p)pp, (p)dp, (p)moe_tp, (p)moe_ep  # prefill 并行配置
          ├── (d)tp, (d)pp, (d)dp, (d)moe_tp, (d)moe_ep  # decode 并行配置
          ├── (p)bs, (d)bs                                 # prefill/decode batch size
          ├── ttft, tpot
          ├── tokens/s/user, tokens/s/gpu
          ├── num_prefill_workers, num_decode_workers
          └── num_total_gpus
```

---

### 阶段 6: 性能数据库 (HYBRID 模式)

**文件**: `src/aiconfigurator_npu/sdk/perf_database.py`

#### 数据库模式说明

| 模式 | 说明 | 数据来源 |
|------|------|----------|
| `SILICON` | 实测数据模式 | 仅使用硬件实测数据 |
| `HYBRID` | 混合模式 (本命令使用) | 实测数据 + SOL 估算 |
| `EMPIRICAL` | 经验模式 | SOL + 经验因子 |
| `SOL` | 理论峰值模式 | 纯理论计算 |

#### HYBRID 模式数据来源

```
PerfDatabase (HYBRID 模式)
  │
  ├── 数据来源优先级:
  │   │
  │   ├── 1. SILICON 数据 (实测)
  │   │   └── 从 CSV 文件加载:
  │   │       ├── gemm_perf.txt          → GEMM 性能数据
  │   │       ├── context_attention_perf.txt → Context Attention 性能
  │   │       ├── generation_attention_perf.txt → Generation Attention 性能
  │   │       ├── moe_perf.txt           → MoE 性能数据
  │   │       ├── MatMulV2.csv           → 矩阵乘法 V2 性能
  │   │       ├── QuantBatchMatmulV3.csv → 量化矩阵乘法性能
  │   │       ├── FusedInferAttentionScore.csv → 融合注意力性能
  │   │       └── GroupedMatmul_MoE_*.csv → MoE 矩阵乘法性能
  │   │
  │   ├── 2. SOL 估算 (Speed of Light)
  │   │   └── 基于硬件理论峰值计算:
  │   │       ├── Ascend 910B 算力: ~320 TFLOPS (FP16)
  │   │       ├── 内存带宽: ~1.6 TB/s
  │   │       └── 通信带宽: HCCL 性能数据
  │   │
  │   └── 3. 经验因子修正
  │       └── 基于历史数据的修正系数
  │
  ├── 性能数据类型:
  │   │
  │   ├── GEMM 性能
  │   │   ├── MatMulV2: 通用矩阵乘法
  │   │   ├── QuantBatchMatmulV3: 量化矩阵乘法
  │   │   └── GroupedMatmul_MoE: MoE 专家矩阵乘法
  │   │
  │   ├── Attention 性能
  │   │   ├── FusedInferAttentionScore (Context)
  │   │   ├── FusedInferAttentionScore_Decode (Generation)
  │   │   └── 支持不同序列长度和 head 数量
  │   │
  │   ├── MoE 性能
  │   │   ├── GroupedMatmul_MoE_BF16
  │   │   └── GroupedMatmul_MoE_W8A8
  │   │
  │   └── 通信性能
  │       ├── AllReduce: 集合通信
  │       └── P2P: 点对点通信
  │
  └── 查询接口:
      ├── query_gemm(M, N, K, quant_mode) → 延迟 (ms)
      ├── query_attention(seq_len, head_dim, quant_mode) → 延迟 (ms)
      ├── query_moe(hidden_dim, expert_num, quant_mode) → 延迟 (ms)
      └── query_communication(size, comm_type) → 延迟 (ms)
```

#### 性能数据文件位置

```
src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/
  ├── context_attention_perf.txt
  ├── FusedInferAttentionScore.csv
  ├── FusedInferAttentionScore_Decode.csv
  ├── gemm_perf.txt
  ├── generation_attention_perf.txt
  ├── GroupedMatmul_MoE_BF16.csv
  ├── GroupedMatmul_MoE_W8A8.csv
  ├── MatMulV2.csv
  ├── moe_perf.txt
  └── QuantBatchMatmulV3.csv
```

---

### 阶段 7: 结果处理与输出

**函数**: `process_experiment_result()` (cli/utils.py)

```python
def process_experiment_result(
    task_config: TaskConfig,
    result: dict[str, pd.DataFrame],
    top_n: int = 10
) -> tuple:
```

**执行流程**:

```
process_experiment_result()
  │
  ├── 1. 提取 Pareto 前沿
  │   └── get_pareto_front(df, x="tokens/s/user", y="tokens/s/gpu")
  │       ├── 使用 Pareto 支配关系筛选
  │       └── 返回非支配解集合
  │
  ├── 2. 选择 top-n 配置
  │   └── pick_default(pareto_df, top_n=10)
  │       ├── 按 tokens/s/gpu 降序排序
  │       └── 取前 10 个配置
  │
  ├── 3. 生成输出文件
  │   │
  │   ├── best_config_topn.csv
  │   │   └── 包含前 10 个最优配置的详细信息
  │   │
  │   ├── pareto.csv
  │   │   └── 完整的 Pareto 前沿数据
  │   │
  │   ├── exp_config.yaml
  │   │   └── 实验配置快照
  │   │
  │   ├── generator_config.yaml
  │   │   └── 可直接用于部署的配置文件
  │   │
  │   └── pareto_frontier.png
  │       └── Pareto 前沿可视化图
  │
  └── 4. 打印摘要
      └── log_final_summary()
          ├── 显示最优配置
          ├── 显示 TTFT/TPOT 预测值
          └── 显示吞吐量预估
```

---

## 4. 关键数据流

```
                              输入参数
                                 │
                                 ▼
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  ModelConfig  │      │ RuntimeConfig │      │  PerfDatabase │
│  (模型配置)    │      │ (运行时配置)   │      │  (性能数据)    │
│               │      │               │      │               │
│ - gemm_quant  │      │ - isl: 4000   │      │ - SILICON     │
│ - kvcache     │      │ - osl: 500    │      │ - HYBRID      │
│ - fmha        │      │ - ttft: 1200  │      │ - EMPIRICAL   │
│ - moe         │      │ - tpot: 50    │      │ - SOL         │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │      Model      │
                      │    (模型定义)     │
                      │                 │
                      │ - Qwen3_8B      │
                      │ - 36 layers     │
                      │ - 8B params     │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │     Backend     │
                      │    (后端实现)     │
                      │                 │
                      │ - VllmBackend   │
                      │ - query_gemm()  │
                      │ - query_attn()  │
                      └────────┬────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  InferenceSession     │
                    │    (推理会话)          │
                    │                       │
                    │ - find_best_result()  │
                    │ - analyze_model()     │
                    └───────────┬───────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   Pareto Analysis     │
                    │    (Pareto 分析)       │
                    │                       │
                    │ - agg_pareto()        │
                    │ - disagg_pareto()     │
                    │ - get_pareto_front()  │
                    └───────────┬───────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │    Results      │
                      │    (结果输出)     │
                      │                 │
                      │ - CSV files     │
                      │ - YAML files    │
                      │ - PNG plots     │
                      └─────────────────┘
```

---

## 5. 核心算法: Pareto 前沿搜索

### 5.1 搜索空间

```
搜索空间 = TP × PP × DP × MoE-TP × MoE-EP × BatchSize × ContextTokens

对于 Qwen3-8B (非MoE模型) + total_gpus=8:
  - TP: [1, 2, 4, 8]
  - PP: [1] (默认禁用)
  - DP: [1] (默认禁用)
  - MoE-TP: [1] (非MoE)
  - MoE-EP: [1] (非MoE)
  - BatchSize: [1, 2, 4, ..., 512]
  - ContextTokens: [512, 1024, ..., 4000] (ISL)
```

### 5.2 约束条件

```
约束:
  - TTFT ≤ 1200ms (首 Token 延迟)
  - TPOT ≤ 50ms (每 Token 延迟)
  - GPU 内存 ≤ 可用显存 (OOM 检查)
```

### 5.3 优化目标

```
优化目标: 最大化 tokens/s/gpu (吞吐量效率)

计算公式:
  - tokens/s/user = 1000 / TPOT
  - tokens/s/gpu = (tokens/s/user × batch_size × num_replicas) / total_gpus
```

### 5.4 Pareto 支配关系

```
解 A 支配解 B 当且仅当:
  - A.ttft ≤ B.ttft 且 A.tpot ≤ B.tpot (至少一个严格小于)
  - A.tokens/s/gpu ≥ B.tokens/s/gpu

Pareto 前沿: 所有非支配解的集合
```

---

## 6. 输出目录结构

```
results/
└── qwen_3_8_config.json_ascend_910b_vllm-ascend_isl4000_osl500_ttft1200_tpot50_[timestamp]/
    │
    ├── agg/                              # 聚合模式结果
    │   ├── exp_config.yaml               # 实验配置快照
    │   ├── best_config_topn.csv          # 前 10 个最优配置
    │   ├── pareto.csv                    # 完整 Pareto 前沿数据
    │   ├── pareto_frontier.png           # Pareto 前沿可视化
    │   │
    │   ├── top1/                         # 最优配置
    │   │   └── generator_config.yaml     # 可直接用于部署的配置
    │   ├── top2/                         # 第 2 优配置
    │   │   └── generator_config.yaml
    │   ├── ...
    │   └── top10/                        # 第 10 优配置
    │       └── generator_config.yaml
    │
    ├── disagg/                           # 分离模式结果
    │   ├── exp_config.yaml
    │   ├── best_config_topn.csv
    │   ├── pareto.csv
    │   ├── pareto_frontier.png
    │   │
    │   ├── top1/
    │   │   └── generator_config.yaml
    │   ├── ...
    │   └── top10/
    │       └── generator_config.yaml
    │
    └── pareto_frontier.png               # agg vs disagg 对比图
```

### 输出文件说明

| 文件 | 说明 |
|------|------|
| `exp_config.yaml` | 实验配置快照，包含所有输入参数 |
| `best_config_topn.csv` | 前 N 个最优配置的详细信息 |
| `pareto.csv` | 完整的 Pareto 前沿数据 |
| `pareto_frontier.png` | Pareto 前沿可视化图 |
| `generator_config.yaml` | 可直接用于部署的生成器配置 |

### best_config_topn.csv 字段说明

| 字段 | 说明 |
|------|------|
| `tp` | Tensor Parallelism 大小 |
| `pp` | Pipeline Parallelism 大小 |
| `dp` | Data Parallelism 大小 |
| `moe_tp` | MoE Tensor Parallelism 大小 |
| `moe_ep` | MoE Expert Parallelism 大小 |
| `bs` | Batch Size |
| `ctx_tokens` | Context Tokens 数量 |
| `ttft` | 预测的 TTFT (ms) |
| `tpot` | 预测的 TPOT (ms) |
| `tokens/s/user` | 用户吞吐量 |
| `tokens/s/gpu` | GPU 吞吐量 |
| `num_total_gpus` | 总 GPU 数量 |

---

## 附录: 关键源码文件索引

| 文件 | 说明 |
|------|------|
| `src/aiconfigurator_npu/cli/main.py` | CLI 入口，模式分发 |
| `src/aiconfigurator_npu/cli/utils.py` | 结果处理工具 |
| `src/aiconfigurator_npu/cli/report_and_save.py` | 报告生成与保存 |
| `src/aiconfigurator_npu/sdk/task.py` | TaskConfig, TaskRunner |
| `src/aiconfigurator_npu/sdk/pareto_analysis.py` | Pareto 分析核心 |
| `src/aiconfigurator_npu/sdk/inference_session.py` | 推理会话 |
| `src/aiconfigurator_npu/sdk/models.py` | 模型定义 |
| `src/aiconfigurator_npu/sdk/perf_database.py` | 性能数据库 |
| `src/aiconfigurator_npu/sdk/operations.py` | 算子定义 |
| `src/aiconfigurator_npu/sdk/common.py` | 通用常量和枚举 |
| `src/aiconfigurator_npu/sdk/config.py` | 配置类定义 |
| `src/aiconfigurator_npu/sdk/backends/vllm_backend.py` | vLLM 后端实现 |
