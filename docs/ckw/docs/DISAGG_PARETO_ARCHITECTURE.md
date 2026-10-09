# Disagg Pareto 架构与流程文档

## 1. 概述

`disagg_pareto` 是 AIConfigurator NPU 的核心函数，用于在**分离式（Disaggregated）推理架构**下寻找最优配置的 Pareto 前沿。它通过枚举 prefill 和 decode worker 的各种组合，在满足 SLA 约束（TTFT/TPOT）的前提下，找到吞吐量/成本最优的配置方案。

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TaskRunner.run_disagg()                          │
│                                (task.py)                                    │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            disagg_pareto()                                  │
│                          (pareto_analysis.py)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  输入参数:                                                           │   │
│  │  • model_path              • prefill_database / decode_database      │   │
│  │  • runtime_config          • prefill_model_config / decode_model_config│  │
│  │  • prefill_parallel_config_list / decode_parallel_config_list        │   │
│  │  • latency_correction_scale (prefill/decode)                         │   │
│  │  • kwargs: num_gpu_list, max_num_gpu, max_prefill_worker, etc.       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DisaggInferenceSession                                 │
│                       (inference_session.py)                                │
│                                                                             │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │  prefill_database   │              │   decode_database   │               │
│  │  prefill_backend    │              │   decode_backend    │               │
│  └─────────────────────┘              └─────────────────────┘               │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                find_best_disagg_result_under_constraints()                  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: 获取 Prefill Worker 候选集                                   │   │
│  │  └─► get_worker_candidates(mode="static_ctx")                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: 获取 Decode Worker 候选集                                    │   │
│  │  └─► get_worker_candidates(mode="static_gen")                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: 遍历约束对 (ttft, tpot)                                      │   │
│  │  └─► _find_best_result_under_constraints()                           │   │
│  │      │                                                               │   │
│  │      ├── 3.1 过滤 Prefill 候选 (ttft < constraint)                   │   │
│  │      ├── 3.2 过滤 Decode 候选 (tpot 范围匹配)                         │   │
│  │      ├── 3.3 按 parallel 分组 Decode 候选                             │   │
│  │      ├── 3.4 执行 Rate Matching (_match_workers)                     │   │
│  │      └── 3.5 构建 Disagg 结果 (_build_disagg_summary_dict)           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: 返回合并后的 Pareto 结果 DataFrame                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. 详细调用流程图

```
                         ┌─────────────────────┐
                         │   disagg_pareto()   │
                         └──────────┬──────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────────┐
│  get_backend  │         │  get_backend  │         │  创建 Session     │
│  (prefill)    │         │  (decode)     │         │                   │
└───────┬───────┘         └───────┬───────┘         └─────────┬─────────┘
        │                         │                           │
        └─────────────────────────┼───────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   DisaggInferenceSession    │
                    │  (prefill_db, decode_db)    │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │ set_latency_correction_scales│
                    │ set_rate_matching_factors    │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              find_best_disagg_result_under_constraints()                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ get_worker_     │    │ get_worker_     │    │ 枚举约束对          │
│ candidates      │    │ candidates      │    │ (ttft, tpot)        │
│ (Prefill)       │    │ (Decode)        │    │                     │
│ mode=static_ctx │    │ mode=static_gen │    │ • request_latency   │
└────────┬────────┘    └────────┬────────┘    │   -> ttft/tpot pairs │
         │                      │             │ • 直接使用 ttft/tpot │
         ▼                      ▼             └──────────┬──────────┘
┌─────────────────┐    ┌─────────────────┐               │
│ 枚举所有        │    │ 枚举所有        │               │
│ parallel_config │    │ parallel_config │               │
│ x batch_size    │    │ x batch_size    │               │
└────────┬────────┘    └────────┬────────┘               │
         │                      │                        │
         ▼                      ▼                        │
┌─────────────────┐    ┌─────────────────┐               │
│ InferenceSession│    │ InferenceSession│               │
│ .run_static()   │    │ .run_static()   │               │
│ (ctx mode)      │    │ (gen mode)      │               │
└────────┬────────┘    └────────┬────────┘               │
         │                      │                        │
         ▼                      ▼                        │
┌─────────────────┐    ┌─────────────────┐               │
│ prefill_        │    │ decode_         │               │
│ summary_df      │    │ summary_df      │               │
└────────┬────────┘    └────────┬────────┘               │
         │                      │                        │
         └──────────────────────┼────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────┐
         │     _find_best_result_under_constraints()        │
         │                                                  │
         │  for each (ttft, tpot) constraint:               │
         │  ┌────────────────────────────────────────────┐  │
         │  │ 1. Filter prefill: ttft < constraint       │  │
         │  │ 2. Filter decode: tpot in valid range      │  │
         │  │ 3. Group decode by parallel config         │  │
         │  │                                            │  │
         │  │ for each decode_parallel_group:            │  │
         │  │ ┌────────────────────────────────────────┐ │  │
         │  │ │ for each (prefill, decode) pair:       │ │  │
         │  │ │   ┌──────────────────────────────────┐ │ │  │
         │  │ │   │ _match_workers()                 │ │ │  │
         │  │ │   │  • 搜索最优 worker 数量组合       │ │ │  │
         │  │ │   │  • 最大化 min(P,D) / total_gpus  │ │ │  │
         │  │ │   └──────────────────────────────────┘ │ │  │
         │  │ │                                        │ │  │
         │  │ │ _build_disagg_summary_dict()           │ │  │
         │  │ │  • 计算合并后的性能指标                 │ │  │
         │  │ └────────────────────────────────────────┘ │  │
         │  │                                            │  │
         │  │ 4. 选择每个 category 最优结果              │  │
         │  │ 5. 返回 top_k 结果                        │  │
         │  └────────────────────────────────────────────┘  │
         └──────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   返回 DataFrame      │
                    │   (disagg_summary_df) │
                    └───────────────────────┘
```

## 4. 核心组件说明

### 4.1 `disagg_pareto()` - 入口函数

**文件**: `src/aiconfigurator_npu/sdk/pareto_analysis.py:176-307`

**职责**: 作为 Pareto 分析的入口点，负责:
1. 解析输入参数
2. 创建 `DisaggInferenceSession`
3. 设置延迟修正因子和速率匹配退化因子
4. 调用核心优化函数

**关键参数**:

| 参数 | 说明 |
|------|------|
| `prefill_database` | Prefill 阶段使用的性能数据库 |
| `decode_database` | Decode 阶段使用的性能数据库 |
| `prefill_parallel_config_list` | Prefill 的并行配置搜索空间 `[[tp,pp,dp,moe_tp,moe_ep], ...]` |
| `decode_parallel_config_list` | Decode 的并行配置搜索空间 |
| `prefill_latency_correction_scale` | Prefill 延迟修正系数（默认 1.1） |
| `decode_latency_correction_scale` | Decode 延迟修正系数（默认 1.08） |

### 4.2 `DisaggInferenceSession` - 分离式推理会话

**文件**: `src/aiconfigurator_npu/sdk/inference_session.py:141-807`

**职责**: 管理分离式推理的核心逻辑，包括:
- 维护 prefill/decode 两套独立的数据库和后端
- 执行 worker 候选枚举
- 实现速率匹配（Rate Matching）算法

**核心属性**:
```python
self._prefill_database: PerfDatabase    # Prefill 性能数据库
self._prefill_backend: BaseBackend      # Prefill 推理后端
self._decode_database: PerfDatabase     # Decode 性能数据库
self._decode_backend: BaseBackend       # Decode 推理后端
self._prefill_latency_correction_scale  # Prefill 延迟修正
self._decode_latency_correction_scale   # Decode 延迟修正
self._rate_matching_prefill_degradation_factor   # Prefill 速率匹配退化因子（默认 0.9）
self._rate_matching_decode_degradation_factor    # Decode 速率匹配退化因子（默认 0.92）
```

### 4.3 `get_worker_candidates()` - Worker 候选枚举

**文件**: `src/aiconfigurator_npu/sdk/inference_session.py:327-447`

**职责**: 枚举所有可能的 worker 配置，返回性能摘要 DataFrame

**流程**:
```
for each parallel_config in parallel_config_list:
    1. 创建 Model（设置 tp/pp/dp/moe_tp/moe_ep）
    2. 创建 InferenceSession
    3. for each batch_size in b_list:
        a. 运行 run_static() 静态推理
        b. 检查是否 OOM
        c. 收集结果到 summary_df
        d. 如果 OOM，跳过更大 batch_size
```

**返回 DataFrame 列**（`ColumnsStatic`）:
- `tp`, `pp`, `dp`, `moe_tp`, `moe_ep` - 并行配置
- `global_bs` - 全局 batch size
- `ttft` / `tpot` - 延迟指标
- `tokens/s`, `tokens/s/gpu`, `seq/s`, `seq/s/gpu` - 吞吐量指标
- `num_total_gpus` - GPU 总数
- `parallel` - 并行配置标识符

### 4.4 `_find_best_result_under_constraints()` - 约束优化

**文件**: `src/aiconfigurator_npu/sdk/inference_session.py:586-702`

**职责**: 在给定 TTFT/TPOT 约束下，找到最优的 P/D 组合

**算法流程**:

```python
# 1. 过滤 Prefill 候选
prefill_candidates = prefill_df[prefill_df['ttft'] < ttft_constraint]
prefill_candidates = sort_by(['seq/s/gpu', 'global_bs'], ascending=[False, True])
prefill_candidates = head(MAX_PREFILL_WORKERS)  # 默认 32

# 2. 过滤 Decode 候选
decode_candidates = decode_df[
    (decode_df['tpot'] < tpot * 1.0) &  # DECODE_FILTER_RATIO_MAX
    (decode_df['tpot'] > tpot * 0.0)    # DECODE_FILTER_RATIO_MIN
]

# 3. 按 parallel 分组处理
for parallel_value, parallel_group in decode_candidates.groupby('parallel'):
    # 4. 遍历所有 (prefill, decode) 组合
    for decode_worker in parallel_group:
        for prefill_worker in prefill_candidates:
            # 5. 执行速率匹配
            prefill_num_worker, decode_num_worker = _match_workers(...)
            # 6. 构建结果字典
            result = _build_disagg_summary_dict(...)

    # 7. 选择当前 category 最优结果
    best = max(results, key=lambda x: (x['tokens/s/gpu'], -x['num_total_gpus']))
```

### 4.5 `_match_workers()` - 速率匹配算法

**文件**: `src/aiconfigurator_npu/sdk/inference_session.py:539-584`

**职责**: 找到最优的 prefill/decode worker 数量组合

**算法**:
```python
def _match_workers(prefill_throughput, prefill_gpus, decode_throughput, decode_gpus, ...):
    best_prefill_num, best_decode_num = -1, -1
    max_throughput_per_gpu = 0

    for decode_num_worker in decode_num_worker_list:
        for prefill_num_worker in prefill_num_worker_list:
            # 计算总 GPU 数
            total_gpus = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker

            # 检查是否满足 GPU 数量约束
            if total_gpus not in num_gpu_set:
                continue

            # 应用速率匹配退化因子
            prefill_corrected = prefill_throughput * prefill_num_worker * prefill_degradation
            decode_corrected = decode_throughput * decode_num_worker * decode_degradation

            # 计算每 GPU 吞吐量（取 P/D 中的较小值）
            throughput_per_gpu = min(prefill_corrected, decode_corrected) / total_gpus

            # 更新最优解
            if throughput_per_gpu > max_throughput_per_gpu:
                max_throughput_per_gpu = throughput_per_gpu
                best_prefill_num, best_decode_num = prefill_num_worker, decode_num_worker

    return best_prefill_num, best_decode_num
```

**核心思想**:
- Prefill 和 Decode 的吞吐量需要匹配
- 取 `min(prefill_throughput, decode_throughput)` 作为系统有效吞吐量
- 目标是最大化 `有效吞吐量 / 总GPU数`

## 5. 数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输入数据                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │  prefill_db     │     │  decode_db      │     │  runtime_config     │   │
│  │  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────────┐  │   │
│  │  │ gemm_data │  │     │  │ gemm_data │  │     │  │ isl: 4000     │  │   │
│  │  │ attn_data │  │     │  │ attn_data │  │     │  │ osl: 1000     │  │   │
│  │  │ moe_data  │  │     │  │ moe_data  │  │     │  │ ttft: 3000    │  │   │
│  │  │ ...       │  │     │  │ ...       │  │     │  │ tpot: 50      │  │   │
│  │  └───────────┘  │     │  └───────────┘  │     │  └───────────────┘  │   │
│  └─────────────────┘     └─────────────────┘     └─────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  parallel_config_lists                                              │   │
│  │  prefill: [[1,1,1,1,1], [2,1,1,1,1], [4,1,1,1,1], [8,1,1,1,1]]    │   │
│  │  decode:  [[1,1,1,1,1], [2,1,1,1,1], [4,1,1,1,1], [8,1,1,1,1]]    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           处理过程                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 1: Worker 候选枚举                                            │   │
│  │                                                                     │   │
│  │  Prefill Workers:                        Decode Workers:            │   │
│  │  ┌────────────────────────────────┐      ┌────────────────────────┐│   │
│  │  │ parallel=(1,1,1,1,1)           │      │ parallel=(1,1,1,1,1)   ││   │
│  │  │   bs=1: ttft=100, seq/s=10     │      │   bs=1: tpot=20, seq/s=5│   │
│  │  │   bs=2: ttft=180, seq/s=18     │      │   bs=2: tpot=22, seq/s=9│   │
│  │  │   ...                          │      │   ...                  ││   │
│  │  ├────────────────────────────────┤      ├────────────────────────┤│   │
│  │  │ parallel=(2,1,1,1,1)           │      │ parallel=(2,1,1,1,1)   ││   │
│  │  │   bs=1: ttft=60, seq/s=12      │      │   bs=1: tpot=15, seq/s=6│   │
│  │  │   ...                          │      │   ...                  ││   │
│  │  └────────────────────────────────┘      └────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 2: 约束过滤                                                   │   │
│  │                                                                     │   │
│  │  TTFT 约束: ttft < 3000ms                                          │   │
│  │  TPOT 约束: 0 < tpot < 50ms                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 3: 速率匹配 (Rate Matching)                                   │   │
│  │                                                                     │   │
│  │  for each (prefill_candidate, decode_candidate):                    │   │
│  │    ┌────────────────────────────────────────────────────────────┐   │   │
│  │    │ 遍历 (prefill_num_worker, decode_num_worker) 组合          │   │   │
│  │    │                                                            │   │   │
│  │    │  P_throughput = P_seq/s × P_num_worker × 0.9              │   │   │
│  │    │  D_throughput = D_seq/s × D_num_worker × 0.92             │   │   │
│  │    │                                                            │   │   │
│  │    │  effective_throughput = min(P_throughput, D_throughput)    │   │   │
│  │    │  throughput_per_gpu = effective / total_gpus               │   │   │
│  │    │                                                            │   │   │
│  │    │  目标: 最大化 throughput_per_gpu                           │   │   │
│  │    └────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 4: 结果聚合                                                   │   │
│  │                                                                     │   │
│  │  每个 parallel category 选择最优结果                                 │   │
│  │  合并所有约束对的结果                                                 │   │
│  │  按 tokens/s/gpu 降序排序                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输出数据                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  disagg_summary_df (ColumnsDisagg):                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  tokens/s/gpu │ tokens/s │ num_total_gpus │ ttft │ tpot │ ...     │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │     120.5     │   964    │       8        │ 2800 │  45  │ ...     │   │
│  │     115.2     │   922    │       8        │ 2600 │  48  │ ...     │   │
│  │      98.7     │  1184    │      12        │ 2200 │  42  │ ...     │   │
│  │     ...       │   ...    │      ...       │ ...  │ ...  │ ...     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6. 关键概念说明

### 6.1 速率匹配 (Rate Matching)

在分离式架构中，Prefill 和 Decode 是独立运行的，需要通过速率匹配来平衡它们的吞吐量：

```
                    ┌─────────────┐
                    │   用户请求   │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     Prefill Workers    │
              │  (高吞吐量, 计算密集)   │
              │  throughput: 100 seq/s │
              └────────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Decode  │       │ Decode  │       │ Decode  │
   │ Worker 1│       │ Worker 2│       │ Worker 3│
   │ 30 seq/s│       │ 30 seq/s│       │ 30 seq/s│
   └─────────┘       └─────────┘       └─────────┘

   Prefill: 100 seq/s × 1 worker × 0.9 = 90 seq/s
   Decode:   30 seq/s × 3 workers × 0.92 = 82.8 seq/s

   系统有效吞吐量 = min(90, 82.8) = 82.8 seq/s
```

### 6.2 退化因子 (Degradation Factor)

| 因子 | 默认值 | 说明 |
|------|--------|------|
| `prefill_degradation_factor` | 0.9 | Prefill 流水线气泡导致的性能损失 |
| `decode_degradation_factor` | 0.92 | Decode batch 未充分填充导致的性能损失 |

### 6.3 延迟修正 (Latency Correction)

| 修正系数 | 默认值 | 说明 |
|----------|--------|------|
| `prefill_latency_correction_scale` | 1.1 | Prefill 实际延迟 = 测量延迟 × 1.1 |
| `decode_latency_correction_scale` | 1.08 | Decode 实际延迟 = 测量延迟 × 1.08 |

### 6.4 并行配置标识符 (Parallel Key)

用于快速分组和比较不同并行配置的 worker：

```python
parallel = f"{tp}_{pp}_{dp}_{moe_tp}_{moe_ep}"
# 例如: "8_1_1_1_1" 表示 TP=8, PP=1, DP=1, MoE_TP=1, MoE_EP=1
```

## 7. 性能约束枚举

当指定 `request_latency` 时，系统会自动枚举满足约束的 (TTFT, TPOT) 对：

```python
# request_latency = ttft + osl * tpot
# 例如: request_latency=5000ms, osl=1000, ttft=3000ms
# 则: tpot = (5000 - 3000) / 1000 = 2ms

# 枚举结果:
constraint_pairs = [
    (2000, 3.0),  # ttft=2000ms, tpot=3.0ms
    (2500, 2.5),  # ttft=2500ms, tpot=2.5ms
    (3000, 2.0),  # ttft=3000ms, tpot=2.0ms
    ...
]
```

## 8. 配置示例

### 8.1 同构部署

```yaml
# prefill 和 decode 使用相同硬件
prefill_worker_config:
  system_name: "ascend_910b"
  backend_name: "vllm-ascend"
  backend_version: "0.18.0"
  tp_list: [1, 2, 4, 8]
  dp_list: [1, 2, 4, 8]

decode_worker_config:
  system_name: "ascend_910b"
  backend_name: "vllm-ascend"
  backend_version: "0.18.0"
  tp_list: [1, 2, 4, 8]
  dp_list: [1, 2, 4, 8]
```

### 8.2 异构部署

```yaml
# prefill 使用高端 GPU，decode 使用低端 GPU
prefill_worker_config:
  system_name: "h200_sxm"
  backend_name: "trtllm"
  tp_list: [8, 16]

decode_worker_config:
  system_name: "l40s_sxm"
  backend_name: "trtllm"
  tp_list: [4, 8]
```

## 9. 输出结果说明

### 9.1 DataFrame 列定义

| 列名 | 类型 | 说明 |
|------|------|------|
| `tokens/s/gpu` | float | 每 GPU 吞吐量（核心优化指标） |
| `tokens/s` | float | 总吞吐量 |
| `seq/s/gpu` | float | 每 GPU 序列吞吐量 |
| `seq/s` | float | 总序列吞吐量 |
| `num_total_gpus` | int | GPU 总数 |
| `ttft` | float | Time To First Token (ms) |
| `tpot` | float | Time Per Output Token (ms) |
| `prefill_tp` | int | Prefill Tensor Parallelism |
| `prefill_pp` | int | Prefill Pipeline Parallelism |
| `decode_tp` | int | Decode Tensor Parallelism |
| `decode_pp` | int | Decode Pipeline Parallelism |
| `prefill_num_worker` | int | Prefill Worker 数量 |
| `decode_num_worker` | int | Decode Worker 数量 |

### 9.2 结果解读

```python
# 获取 Pareto 前沿
result_df = disagg_pareto(...)
pareto_df = get_pareto_front(result_df, "tokens/s/user", "tokens/s/gpu")

# 最优配置示例:
# tokens/s/gpu=120.5, ttft=2800ms, tpot=45ms
# prefill: TP=8, 1 worker
# decode:  TP=4, 2 workers
# 总 GPU: 8×1 + 4×2 = 16
```

## 10. 架构优势

1. **解耦设计**: Prefill 和 Decode 可以独立配置和优化
2. **异构支持**: 支持不同硬件、不同后端版本的混合部署
3. **灵活搜索**: 可自定义并行配置搜索空间
4. **精确建模**: 使用独立的性能数据库，准确预测实际性能
5. **约束驱动**: 支持多种 SLA 约束（TTFT、TPOT、request_latency）
