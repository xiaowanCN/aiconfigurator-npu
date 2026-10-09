# SILICON 模式数据需求与缺口分析

> 分析对象命令：
>
> ```bash
> aic-npu default \
>   --model-path /workspace/model_configs/qwen_3_8_config.json \
>   --total-gpus 8 \
>   --system ascend_910b \
>   --backend vllm-ascend \
>   --database-mode HYBRID \
>   --isl <> --osl <> \
>   --ttft 1600 --tpot 60 \
>   --top-n 10 \
>   --save-dir results
> ```
>
> 问题：若将 `--database-mode` 切换为 `SILICON`，需要哪些数据？目前还缺少哪些？

---

## 1. 结论概览

`--database-mode SILICON` 表示**仅使用真机实测数据表**（`*_perf.txt`）进行查表 + 插值估算，缺任何一张被用到的表都会直接报错 `PerfDataNotAvailableError`（并提示可改用 HYBRID）。

而 `HYBRID` 是「实测优先、缺失则回退 SOL + 经验系数」，所以当前缺通信表也能跑通。

| 模式 | 数据要求 |
|------|----------|
| **HYBRID** | 实测表有就用，没有就 `SOL / 0.8` 经验兜底 → 不因缺表中断 |
| **SILICON** | 模型路径上**每一个查表算子**都必须有对应 `*_perf.txt`，且查询维度落在插值网格内 → 缺表直接 raise |

---

## 2. 关键代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| `DatabaseMode` 枚举（SILICON/HYBRID/EMPIRICAL/SOL/SOL_FULL） | `src/aiconfigurator_npu/sdk/common.py` | 528–537 |
| `PerfDataFilename` 枚举（全部表文件名） | `src/aiconfigurator_npu/sdk/common.py` | 551–583 |
| `--database-mode` CLI 参数（default 子命令，choices 排除 SOL_FULL，默认 SILICON） | `src/aiconfigurator_npu/cli/main.py` | 141–152 |
| `PerfDatabase.__init__`：读 `systems/{system}.yaml`，拼 `data_dir` / `nccl_data_dir` | `src/aiconfigurator_npu/sdk/perf_database.py` | 2102–2123 |
| `_load_op_data`：按 `PerfDataFilename` 加载各 `*_perf.txt` | `src/aiconfigurator_npu/sdk/perf_database.py` | 2125–2166 |
| nccl 特殊路径：走 `data_dir/nccl/{misc.nccl_version}/` | `src/aiconfigurator_npu/sdk/perf_database.py` | 2151–2155 |
| `LoadedOpData.raise_if_not_loaded`：SILICON 下文件缺失 → raise | `src/aiconfigurator_npu/sdk/perf_database.py` | 2027–2043 |
| `_query_silicon_or_hybrid`：SILICON 失败直接 raise；HYBRID 回退 empirical | `src/aiconfigurator_npu/sdk/perf_database.py` | 3474–3513 |
| 系统规格（`data_dir: data/ascend_910b`、`misc.nccl_version: '2.26.0'`） | `src/aiconfigurator_npu/systems/ascend_910b.yaml` | 5, 31 |

---

## 3. SILICON 需要的数据（路径约定）

数据根路径由 `systems/ascend_910b.yaml` 的 `data_dir` 和 backend/version 决定，对应命令参数（`--system ascend_910b --backend vllm-ascend`）：

```text
src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/<op>_perf.txt   # 算子/计算类
src/aiconfigurator_npu/systems/data/ascend_910b/nccl/2.26.0/nccl_perf.txt           # HCCL 集合通信
```

> 注：仓库根目录另有平行目录 `systems/data/ascend_910b/...`（可通过 `--systems-paths` 加载），结构相同。

### 3.1 计算/算子类表（跑 Qwen3 系列通常必需）

| 文件 | 用途 | 关键字段 |
|------|------|----------|
| `gemm_perf.txt` | GEMM 延迟 | `framework, version, device, op_name, kernel_source, gemm_dtype, m, n, k, latency` |
| `context_attention_perf.txt` | Prefill attention | `..., batch_size, isl, num_heads, num_key_value_heads, head_dim, beam_width, attn_dtype, kv_cache_dtype, step, latency` |
| `generation_attention_perf.txt` | Decode attention | 同上（decode 维度） |
| `moe_perf.txt` | MoE（若 MoE 模型） | `..., moe_dtype, num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, distribution, kernel_source, latency` |

### 3.2 通信类表（TP > 1 时必需）

| 文件 | 用途 | 关键字段 |
|------|------|----------|
| `custom_allreduce_perf.txt` | TP AllReduce（custom 内核） | `allreduce_dtype, num_gpus, message_size, latency, power` |
| `nccl/2.26.0/nccl_perf.txt` | HCCL 集合通信 | `nccl_dtype, op_name, num_gpus, message_size, latency, power`；`op_name` ∈ `all_reduce` / `all_gather` / `reduce_scatter` / `alltoall` |

### 3.3 按模型可选表（用到才强制）

以下表仅在模型算子路径走到时才强制要求，缺则 SILICON raise：

- MLA 相关：`context_mla` / `generation_mla` / `mla_bmm` → `*_mla*_perf.txt`
- DSA 模块：`dsa_context_module` / `dsa_generation_module` → `dsa_*_module_perf.txt`
- 静态量化 scale：`compute_scale` / `scale_matrix` → `computescale_perf.txt` / `scale_matrix_perf.txt`
- WideEP / MoE dispatch：`wideep_*` / `trtllm_alltoall`
- Mamba 系列：`mamba2` / `gdn` → `mamba2_perf.txt` / `gdn_perf.txt`

---

## 4. 当前已有 vs 缺少

### 4.1 ✅ 已有（正式路径下）

根目录：`src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/`

| 文件 | 角色 |
|------|------|
| `gemm_perf.txt` | GEMM 实测（已有表头与数据） |
| `moe_perf.txt` | MoE 实测 |
| `context_attention_perf.txt` | Prefill attention 实测 |
| `generation_attention_perf.txt` | Decode attention 实测 |
| `MatMulV2.csv` / `QuantBatchMatmulV3.csv` | GEMM 原始 CSV（转换源） |
| `GroupedMatmul_MoE_BF16.csv` / `_W8A8.csv` | MoE 原始 CSV |
| `FusedInferAttentionScore.csv` / `_Decode.csv` | Attention 原始 CSV |

### 4.2 ❌ 缺少（SILICON 会挂的点）

| 缺失项 | 影响 | 说明 |
|--------|------|------|
| **`custom_allreduce_perf.txt`** | TP > 1 时查通信表 raise | 正式目录 `systems/data/...` 下不存在 |
| **`nccl/2.26.0/nccl_perf.txt`** | 集合通信查表 raise | 正式目录下不存在 |
| MLA / DSA / scale / WideEP 等 | 视模型而定 | 按模型算子路径缺哪张报哪张 |

### 4.3 ⚠️ 特别注意：backup 里有通信数据，但没接入

路径：`backup/aic_npu_v0920_workspace_backup/workspace/data/hccl_data/`

| 文件 | 状态 |
|------|------|
| `custom_allreduce_perf.txt` | 18 行，half/8 卡 |
| `nccl_perf.txt` | 37 行，**且缺 `reduce_scatter`、`alltoall` 算子** |

按 `docs/HCCL_COMM_PERF_DATA_COLLECTION_GUIDE.md` 说明：

1. backup 路径**不会被框架读取**，必须放到 `systems/data/ascend_910b/...` 才生效；
2. 即使放进去，这份 `nccl_perf.txt` 算子不全（缺 `reduce_scatter` / `alltoall`），TP/EP 常用算子查到时仍会失败或无法插值。

### 4.4 其他目录说明

- `results/`：CLI 输出（`best_config_topn.csv`、`pareto.csv`、`exp_config.yaml` 等），**不是**性能数据库。
- 本仓库**没有** `.db` / `.sqlite` 性能数据库；「库」= CSV 风格 `*_perf.txt` + `PerfDatabase` 内存索引 + 插值。

---

## 5. SILICON 失败行为（报错样例）

```text
raise_if_not_loaded()
  → File does not exist at {filepath}
  → "not supported by AIC in SILICON mode"

_query_silicon_or_hybrid (SILICON)
  → 失败附加 "Consider using HYBRID mode." 并 re-raise
```

---

## 6. 切换到 SILICON 前的行动清单

1. **补通信两表**到正式路径：
   - `systems/data/ascend_910b/vllm-ascend/0.18.0/custom_allreduce_perf.txt`
   - `systems/data/ascend_910b/nccl/2.26.0/nccl_perf.txt`（需含 `all_reduce` / `all_gather` / `reduce_scatter` / `alltoall`）
2. **补全 nccl 算子覆盖**（backup 那份缺 `reduce_scatter`、`alltoall`，不够用）。
3. **按实际模型**（`qwen_3_8_config.json` 对应结构）确认是否还用到 MLA / DSA / scale / WideEP 等，缺则按 `PerfDataFilename` 枚举补对应表。
4. 参考采集/放置文档：
   - `docs/HCCL_COMM_PERF_DATA_COLLECTION_GUIDE.md`（放置路径 §8.1、字段、验证脚本）
   - `docs/HCCL_ALLREDUCE_PERF_DATA_GUIDE.md`
   - `docs/V1_0_0/4_数据库表设计.md`（表结构设计）

---

## 7. 相关文档索引

| 文档 | 要点 |
|------|------|
| `docs/HCCL_COMM_PERF_DATA_COLLECTION_GUIDE.md` | 最完整：两套通信文件、路径、字段、SILICON raise 条件、放置路径、验证脚本；L48 明确当前正式目录无此二文件 |
| `docs/HCCL_ALLREDUCE_PERF_DATA_GUIDE.md` | custom_allreduce / nccl 字段与样例；SILICON 查询伪代码 518–547 |
| `docs/ARCHITECTURE_AND_CALCULATION_FRAMEWORK.md` | path 公式、本仓库数据布局、四模式表（L203, L300–334） |
| `docs/AIC_NPU_DEFAULT_COMMAND_ARCHITECTURE.md` | SILICON=仅实测；HYBRID 数据源优先级与文件列表（L596–656） |
| `docs/V1_0_0/2_系统架构与功能模块设计.md` | 四种 mode；SILICON「缺失直接报错」（L61, L114–121） |
| `docs/V1_0_0/4_数据库表设计.md` | 「数据库」= CSV `.txt`；`gemm_perf` 字段样例 |
| `docs/V1_0_0/6_CLI接口文档.md` | 参数取值、非法取值报错、SILICON 失败提示改 HYBRID（L75, L260, L268, L407） |
| `docs/数据采集.txt` | backup HCCL 数据位置与正确放置路径分析 |

---

*生成日期：2026-09-24*
