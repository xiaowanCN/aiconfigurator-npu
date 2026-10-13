# 三模型适配度审计报告（对标 NVIDIA aiconfigurator）

- 审计分支：`check/model-adaptation-audit`（自 `feature/deepseek-v4-pro-support` 切出）
- 审计对象：`Qwen3.8-27B`、`DeepSeek-R1`、`DeepSeek-V4-Pro`
- 参考基准：NVIDIA 侧 `aic-npu-cn-ascend/aiconfigurator`（`aic-core` main）
- 审计时间：2026-10-13

---

## 0. 结论速览

**三个模型的适配均未全量完成。** 只有 DeepSeek-V4-Pro 能在 SILICON 模式跑出结果，
DeepSeek-R1 与 Qwen3.8-27B 存在硬性数据缺口。你日常使用的 `--database-mode HYBRID`
会把这些缺口**静默降级为 SOL 估算**，因此表现上"能出数"，但数字在缺数据处并不可信。

| 模型 | 走的计算图 | SILICON 模式实测结果 | 判定 |
|---|---|---|---|
| `deepseek-ai--DeepSeek-V4-Pro` | `DeepSeekV4Model` → CSA/HCA + mHC + MoE | agg ✅（64 卡：312.70 tokens/s）；**disagg 无结果** | **半可用** |
| `deepseek-ai--DeepSeek-R1` | `DeepSeekModel` → MLA + `MLABmm` + MoE | ❌ agg / disagg 均无结果 | **未可用**（缺 MoE ep 网格 + MLA heads 网格） |
| `qwen3_8_27B` | `Qwen35Model` → GDN + dense attention | ❌ agg / disagg 均无结果 | **未适配**（GDN 表完全缺失 + head_dim=256 未采） |

额外提示：V4-Pro 的 disagg 同样不出结果（日志 `Experiment disagg returned no results`），
且开启跨 ep 迁移后仍是如此，说明它不完全是 MoE ep 缺口造成的（更像是 PD 拆分在 64 卡下
装不下 / 达不到 SLA），列为待深入项 T14。

补充：仓库里另外三个同样声明支持的模型也不成立 —— `zai-org/GLM-5`（缺 DSA 表）、
`Qwen/Qwen3-235B-A22B`（MoE 形状未采）、`deepseek-ai/DeepSeek-V3`（同 R1）。
`src/aiconfigurator_npu/systems/support_matrix.csv` 原来对所有行一律标记 `PASS`，
**与实测不符**，已在本次修正（见 §4-D）。顺带发现 `Qwen/Qwen3-32B` 实际可用
（13,181 tokens/s）却未登记在矩阵里。

一键复现判定：

```bash
python tools/check_model_readiness.py
```

> **补全计划见 `docs/DATA_COLLECTION_PLAN.md`**：按 P0/P1/P2 分级给出每张表的
> 目标维度、采集脚本、可直接执行的命令与验收方式。本文只负责现状体检与根因分析。

---

## 1. 直接回答：DeepSeek-R1 需要 DSA 数据吗？

**不需要。这是本次审计澄清的一个认知误区。**

英伟达侧 `b300_sxm/sparse_attention/vllm/0.24.0/dsa_context_module_perf.parquet`
（以及 `dsa_generation_module_perf.parquet`）服务于 **DeepSeek-V3.2 的稀疏注意力（DSA）**，
对应的算子是 `ContextDSAModule` / `GenerationDSAModule`，只在 `DeepSeekV32Model` 里被构造：

- `aiconfigurator/aic-core/src/aiconfigurator_core/sdk/models/deepseek_v32.py:297,329,433,547`
  调用 `ops.ContextDSAModule(...)` / `ops.GenerationDSAModule(...)`
- `aiconfigurator/aic-core/src/aiconfigurator_core/sdk/models/deepseek.py:322`
  DeepSeek-V3/R1 走的是 `ops.ContextMLA(...)` + `ops.GenerationMLA(...)`，**完全不碰 DSA**

NPU 侧路由完全一致（`src/aiconfigurator_npu/sdk/models.py`）：

| 模型 / 架构 | NPU 模型类 | 需要的表 |
|---|---|---|
| `DeepseekV3ForCausalLM`（R1、V3） | `DeepSeekModel` | `context_mla` / `generation_mla` / `mla_bmm` / `moe` / `gemm` |
| `DeepseekV32ForCausalLM`、`GlmMoeDsaForCausalLM`（GLM-5） | `DeepSeekV32Model` | **`dsa_context_module` / `dsa_generation_module`** ← 这才是 DSA 的使用方 |
| `DeepseekV4ForCausalLM`（V4-Pro） | `DeepSeekV4Model` | `dsv4_csa/hca_*_module` / `mhc_module` / `moe` / `gemm` |
| `Qwen3_5ForConditionalGeneration`（Qwen3.8-27B） | `Qwen35Model` | **`gdn`** / `attention` / `gemm` |

所以：**R1 缺的是 MLA 的 heads 网格和 MoE 的 ep 网格，不是 DSA。**
真正需要 DSA 数据的是 `zai-org/GLM-5`（见 §4-C）。

旁证：`model_configs/deepseek-ai--DeepSeek-R1_config.json` 里没有任何
`index_n_heads` / `index_head_dim` / `index_topk` 之类的稀疏注意力字段；
而 `deepseek-ai--DeepSeek-V4-Pro_config.json` 有 `index_n_heads=64`、`index_topk=1024`。

---

## 2. 采集粒度对标（NVIDIA b300_sxm / vllm vs NPU ascend_910b / vllm-ascend）

NVIDIA 侧按算子分子目录维护（`data/<gpu>/<op>/<backend>/<version>/*.parquet`），
NPU 侧是扁平目录 `data/ascend_910b/vllm-ascend/0.23.0/<op>_perf.txt`。下表为实测统计。

| 算子 | NVIDIA b300_sxm/vllm/0.24.0 | NPU ascend_910b/vllm-ascend/0.23.0 | 差距 |
|---|---|---|---|
| gemm | 844 KB | 2430 行，dtype 仅 `float16`、`sq` | 无 w8a8 / fp8 / int8 行（代码已把 fp8 重映射到 float16 兜底） |
| MoE | 582 KB | 279 行 | **结构性缺口**，见 §3-A |
| MoE all-to-all 通信 | `comm/moe_a2a_perf.parquet`（16 KB） | **整表缺失** | EP 场景的 all-to-all 通信未被单独采集 |
| context/generation MLA | 59/36 KB（module 版） | 167/324 行，heads ∈ {16,32,64,128} | **缺 heads 4/8** |
| mla_bmm | 5 KB | 96 行，heads ∈ {16,32,64,128} | **缺 heads 4/8** |
| dense attention | 261/190 KB | 253/1275 行，`head_dim` 仅 **128** | **缺 head_dim=256**（Qwen3.5 需要） |
| linear_attention / GDN | 63 KB | **整表缺失** | Qwen3.5 不可用 |
| mamba2 | trtllm 有 15 KB | 缺失 | Nemotron-H 不可用 |
| mhc | 5 KB | 32 行（仅 `hc_mult=4`） | 覆盖极薄 |
| comm（nccl / allreduce） | 多种 gpu 数 | `num_gpus` 仅 {2,4,8} | **16/32/64 卡集群通信未采** |
| sparse_attention(DSA) | 103/41 KB + dsv4 系列 5 张 | DSA 缺失；dsv4 有 4 张 | 见 §4-C |

**规律**：DSv4 这条线（最新工作）的粒度做到了 heads 下探到 4；
而 MLA / dense attention / MoE 这些"更早"的表仍按旧规划采集，没有跟上
`TaskConfig._widen_gpu_sweep_lists` 把 tp/ep 扩展到 16/32/64 之后的搜索空间。

---

## 3. 根因分析（为什么 SILICON 模式跑不出结果）

### 3-A. MoE 表：(experts, ep, distribution) 网格是"对角线"而非笛卡尔积

`moe_perf.txt` 里每个专家形状**只在单一 ep、单一 distribution 上采过**：

```
hidden=7168 inter=2048 topk=8 experts=256  (DeepSeek-R1/V3)  → ep=[8]        dist=power_law_1.2
hidden=7168 inter=3072 topk=6 experts=384  (DeepSeek-V4-Pro) → ep=[16,32,64] dist=power_law_1.01
hidden=6144 inter=2048 topk=8 experts=256  (GLM-5)           → 无行
hidden=4096 inter=1536 topk=8 experts=128  (Qwen3-235B)      → 无行
```

而搜索空间强制 `dp * tp == moe_tp * moe_ep` 且 vLLM 系列把 `moe_tp` 钉死为 1
（`src/aiconfigurator_npu/sdk/task.py` vllm/vllm-ascend 分支），
**于是 `ep` 就等于 worker 宽度**。结果：

- R1 只有 8 卡 worker 可被 silicon 定价 —— 而 R1 权重约 671 GB，8 张 910B（64 GB/卡）根本装不下；
- 更大的 worker（16/32/64 卡）缺 ep 数据 → 全部配置被跳过 → "No results found"。

这正是你看到 R1 在 SILICON 模式下整轮失败的原因。

### 3-B. `_resolve_moe_distribution` 只在"键缺失"时回退，不管"形状缺失"

这是 §3-A 之外的**独立代码缺陷**，也是 R1 失败的第一现场。

DeepSeek 家族的 `_power_law_alpha = 1.01`（`models.py:1324,1685,1958`），
而 ascend 的 `moe_perf.txt` 里 R1 形状的行挂在 `power_law_1.2` 下。
原 `_resolve_moe_distribution` 的解析顺序是"精确匹配 → 同 family → uniform"，
`power_law_1.01` **精确命中**，于是返回一个该形状下为空的 dict，
再往下走 `token_points[-1]` 就抛出难以理解的：

```
IndexError: list index out of range
Failed to query moe data for ... distribution='power_law_1.01'
```

### 3-C. MLA heads 网格未随 tp 扩展下探

`context_mla_perf.txt` / `generation_mla_perf.txt` / `mla_bmm_perf.txt` 的
`num_heads` 只有 `{16,32,64,128}`。以 128 头模型按 tp 切分：

| tp | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| rank-local heads | 128 | 64 | 32 | 16 | **8** | **4** | **2** |
| MLA 表 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| DSv4 表 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

对比可见：DSv4 的表做到了 heads=4，MLA 的表停在 16。
tp≥16 的配置会在 `_nearest_1d_point_helper` 抛
`x is less than the smallest value in the list`。

注意这里还有一个**硬件约束**（已在 DSv4 分支发现，commit `b0dd50f`）：
Ascend 910B 的 `SparseAttnSharedkv` 要求 `q_head_num` 为 4 的倍数，
`num_heads=2`（tp=64）在 kernel 层根本跑不起来。所以 heads 的下限是 4、不是 2。

### 3-D. Qwen3.8-27B：算子就位但两张表都没有

`qwen3_8_27B_config.json` 的 `layer_types` 里 `linear_attention : full_attention = 3 : 1`，
因此走 `Qwen35Model` → `ops.GDNKernel(...)`（`models.py:4890`），需要 `gdn_perf.txt`；
同时 `full_attention` 层需要 `head_dim=256`（该配置 `head_dim: 256`），
而 `context_attention_perf.txt` 只采了 `head_dim=128`。两处都不通。

---

## 4. 缺陷清单

### 已完成修复（本次提交）

| # | 缺陷 | 位置 | 处置 |
|---|---|---|---|
| F-A | MoE 精确命中 distribution 但该形状为空 → `IndexError` | `perf_database.py::_resolve_moe_distribution` | 改为**形状感知**：候选 distribution 逐个验证该专家形状是否存在；均无则返回精确请求，交上层报精准缺失维度 |
| F-B | 缺失数据报错不可读 | `perf_database.py::query_moe` | 空 slice 抛 `PerfDataNotAvailableError`，明确指出缺的是哪个 `(experts, topk, hidden, inter, ep, quant, distribution)` 以及该去重采还是切 HYBRID |
| F-C | tp 与 num_heads 不可除时整轮搜索被 `assert` 打断 | `task.py::prune_parallel_configs_by_silicon_coverage` | 在枚举后、Pareto 前剪掉 `num_heads % tp != 0` 及 MLA heads 网格外的配置，并打印剪枝原因 |
| F-D | MoE ep 缺口导致 worker 宽度集体失效 | `perf_database.py::moe_transfer_slice` | 新增**默认关闭**的 `AIC_NPU_MOE_TRANSFER=1` 逃生舱：用最近 ep 的测量值按 SOL 比迁移，并 WARNING 标注"这是估算不是实测" |
| F-E | 无法在跑之前判断模型是否真的可用 | `tools/check_model_readiness.py` | 新增静态检查工具 |
| F-F | `support_matrix.csv` 全量标 PASS，且 Backend 写成 `vllm`（数据目录名是 `vllm-ascend`，导致 `support --backend vllm-ascend` 匹配不上） | `src/aiconfigurator_npu/systems/support_matrix.csv` | 按实测回填：仅 V4-Pro agg 为 PASS，Backend 修正为 `vllm-ascend` |

### 待办（需要 910B 硬件重采，不能在开发环境完成）

| # | 任务 | 涉及命令/文件 |
|---|---|---|
| **T1** | **MoE 补 ep 网格**：把 DeepSeek-R1/V3 的 `experts=256/topk=8/inter=2048` 采到 `ep ∈ {4,8,16,32,64}`；V4 的 `experts=384` 补 `ep ∈ {1,2,4,8}`；并统一或显式冗余两种 `distribution` | `collector/npu/collect_moe.py` |
| **T2** | **MLA / mla_bmm 补 num_heads ∈ {4, 8}**（tp=16/32）；注意 heads=2 在 Ascend 上不可实现 | `collector/npu/collect_mla.py`、`collect_mla_bmm.py` |
| **T3** | **dense attention 补 `head_dim=256`**（Qwen3.5 需要），并按其 24 头补齐 `num_heads ∈ {24,12,6,3}` | `collector/npu/collect_attn.py` |
| **T4** | **采集 GDN 表**（`gdn_perf.txt`），否则 Qwen3.8-27B 无解 | 需新增对应 factory/collector |
| **T5** | **采集 DSA 表**（`dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt`），GLM-5 与 V3.2 依赖 | 需新增 |
| **T6** | **通信表补 16/32/64 卡**：`nccl_perf.txt` 与 `custom_allreduce_perf.txt` 现在只到 8 卡，64 卡集群的 TP/EP 通信全靠外推 | `collector/npu/collect_hccl.py` |
| **T7** | **补 MoE all-to-all 通信**（NVIDIA 侧有 `comm/moe_a2a_perf.parquet`），否则大 EP 部署的通信开销没有被建模 | 需新增 |

### 待办（纯代码，可继续做）

| # | 任务 | 说明 |
|---|---|---|
| **T8** | ~~修正 `support_matrix.csv`~~ **已完成**（见 F-F） | 剩余：把实测可用的 `Qwen/Qwen3-32B` 补登记进矩阵 |
| **T9** | 澄清仓库根的 `systems/` 残留副本（**已判定：不应删除**） | 结构与包内不一致（YAML 藏在 `ascend_910b_aiconfigurator/` 子目录，而加载器要求 `<root>/<system>.yaml`），因此**永远不会被加载**，里面的 0.18.0 数据不影响运行时。但它保存了唯一留存的 TensorCast 原始 CSV（`GroupedMatmul_MoE_BF16.csv` 等），是有价值的溯源资产；且 `DSA_COLLECTION_COMMANDS.md`、`GLM5_ADAPTATION_DESIGN.md` 仍引用该路径。**已验证它与 0.23.0 表不同源**（91 个重合单元格全部不一致，最大偏差 461%），因此只能当存档，不能拿来补 0.23.0（见 `DATA_COLLECTION_PLAN.md` P0-2）。建议在目录内放一个 README 说明其性质，而非删除 |
| **T10** | 对齐英伟达侧搜索维度 | NPU 的 `enumerate_parallel_config` 缺少 NVIDIA 已有的 `cp_list`（context parallel）、`moe_backend`（MegaMoE）过滤，且仍在用已被上游废弃的 `enable_wideep`；NVIDIA 已改为由数据表覆盖率（`moe_a2a_coverage` / `moe_expert_compute_coverage`）驱动 large-EP 决策 |
> **是否需要计划 —— 结论：需要立项，但不要插队，且不阻塞三模型出结果。**
>
> 影响面：三个目标模型都经 `vllm-ascend` → `VLLMBackend`，所以**都会受影响**。
> 证据链（已核到行）：
> - NPU 侧 `inference_summary.py:81`：`self._is_oom = self._memory["total"] >= (mem_capacity / (1 << 30))`
>   —— 直接把显存塞到 100%，没有任何框架级上限；
> - 上游 `backends/vllm_backend.py:15-90`：新增 `gpu_memory_utilization` 上限
>   （`0.14/0.19→0.9`，`0.22/0.24→0.92`）+ 2% KV 粒度容差，并配 5 个钩子
>   （`get_default_free_gpu_memory_fraction` / `get_kv_cache_memory_check_params` /
>   `memory_fraction_of_free` / `_oom_check_kwargs` / `_mix_step_efficiency`），
>   这 5 个钩子 NPU 侧**一个都没有**（全 sdk 搜不到 `free_gpu_memory_fraction`）；
> - 上游动机：不加这个上限会选出**真实部署 admit 不了**的 decode batch
>   （引用 ai-dynamo/aiconfigurator#1396：Kimi-K2.5-NVFP4 预测 bs=96/rank，实机只 29/rank）。
>
> 但它**不是**三模型出结果的瓶颈：R1/Qwen3.8 卡在数据缺失；V4-Pro agg 已经能出数；
> 而 V4-Pro 的 disagg 无结果是"bf16 ≈2894 GiB 装不下"的真实容量问题 ——
> 加上限只会让可见内存更少，**不会**把 disagg 变可行。
> 因此它的定位是"**数字可信度**"问题：现在推荐的 batch size 可能偏乐观。
>
> 落地前提（缺一不可）：① 先在 NPU 上标定 vllm-ascend 的 `gpu_memory_utilization` 默认值，
> 不能直接套 NVIDIA 的版本表；② 需连带改 `BaseBackend` + `InferenceSummary` 的 OOM 判定
> + TRTLLM/SGLang 的 `fraction_of_free`，属跨层改动。**建议排在 `DATA_COLLECTION_PLAN.md`
> 第 1–3 项之后。**
| **T12** | README / `docs/DeepSeek-V4-Pro_ADAPTATION.md` 与数据现状同步 | 文档仍写 MLA/dsv4/mhc 表 "pending collection"，实际 0.23.0 里这些文件已存在 |
| **T13** | `mhc_module_perf.txt` 仅 32 行 | V4-Pro 的 mHC 分支插值可靠性存疑，建议加密 `num_tokens` 采样 |
| **T14** | V4-Pro 的 disagg 模式恒无结果 | 日志 `main.py:978 Experiment disagg returned no results`；开启跨 ep 迁移后仍无结果，说明原因不全在数据缺失，需在 910B 上确认是 PD 拆分容量不足还是约束过紧 |
| **T15** | `Qwen/Qwen3-32B` 的 agg 模式报"显存放不下" | 8 卡与 64 卡均报 `does not fit in GPU memory for any parallel configuration`，而 disagg 在 64 卡可跑到 13,181 tokens/s。已确认**不是**本次新增的 tp 剪枝造成（该模型 4→4 / 7→7 未被剪）。疑似显存估算偏保守，需单独立项排查 |

---

## 5. 本次改动的行为影响与验证

回归验证（同一条命令、同一份数据，改动前后逐一比对）：

| 场景 | 改动前 | 改动后 | 结论 |
|---|---|---|---|
| DeepSeek-V4-Pro 64 卡 SILICON | `312.70 tokens/s`，`4.89 tokens/s/gpu`，`tp8pp1dp8etp1ep64`，disagg 无结果 | **完全一致**（含 disagg 仍无结果） | 默认路径零行为变化 |
| DeepSeek-V4-Pro disagg | 无结果 | 开启 `AIC_NPU_MOE_TRANSFER=1` 后**仍无结果** → 确认为独立问题 T14 | 一致 |
| DeepSeek-R1 64 卡 SILICON | `No successful experiment runs`，报 `IndexError` | 默认仍是：报**可读的** `PerfDataNotAvailableError`（指出缺 `moe_ep_size=64`）；启用 `AIC_NPU_MOE_TRANSFER=1` 后出结果（agg `22.81 tokens/s/gpu`） | 失败可诊断；逃生舱可用 |
| Qwen3.8-27B 8 卡 SILICON | `num_heads 24 should be divisible by tp_size 64` 断言崩溃 | 该非法 tp 在枚举阶段被剪掉，日志给出剪枝原因；最终报缺 `gdn_perf.txt` 与 `head_dim=256` | 崩溃 → 明确诊断 |
| GLM-5 | `support_matrix` 标 PASS，实际抛 DSA 表缺失 | 同上，报缺 `dsa_context_module_perf.txt` | 一致，且工具可提前发现 |
| Qwen3-235B-A22B | `support_matrix` 标 PASS，实际无结果 | 同上，报 MoE 无 `(experts=128, topk=8, hidden=4096, inter=1536)` 形状 | 一致 |
| Qwen3-32B | 未登记在矩阵 | 实测可用，`13,181.57 tokens/s`（agg） | 待补登记（T8） |
| `aic-npu support` 子命令 | Backend 列写 `vllm`，传 `--backend vllm-ascend` 匹配失败 | 修正为 `vllm-ascend`，V4-Pro 现正确报 agg=YES / disagg=NO | 已修复 |

**重要**：`AIC_NPU_MOE_TRANSFER=1` 出来的 R1 数字是**估算值**（跨 ep 迁移），
只能用于趋势对比，不能当作实测选型依据。T1 完成前请勿据此做交付决策。

---

## 6. 复现命令

```bash
# 1) 就绪度体检（不需要 NPU，秒级）
python tools/check_model_readiness.py

# 2) R1 默认行为（应报可读的缺失诊断）
aic-npu default --model-path ./model_configs/deepseek-ai--DeepSeek-R1_config.json \
  --total-gpus 64 --system ascend_910b --backend vllm-ascend --backend-version 0.23.0 \
  --database-mode SILICON --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 --top-n 10

# 3) R1 逃生舱（跨 ep 迁移，数字为估算）
AIC_NPU_MOE_TRANSFER=1 aic-npu default ... --database-mode SILICON ...

# 4) V4-Pro 基线（应保持不变）
aic-npu default --model-path ./model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json \
  --total-gpus 64 --system ascend_910b --backend vllm-ascend --backend-version 0.23.0 \
  --database-mode SILICON --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 --top-n 10
```

---

## 7. 建议的推进顺序

1. **先 runway T9 + T8**（清理误导性残留副本、修正 support matrix），成本极低但能避免持续误判；
2. **T1 + T2** 是让 R1 真正可用的关键路径，建议在下次 910B 机时窗口一次性补齐两个表的完整网格；
3. **T3 + T4** 是 Qwen3.8-27B 能否上线的前提，二者缺一不可；
4. **T10 + T11** 属于与上游对齐的架构债，建议在 T1–T4 完成后统一处理。
