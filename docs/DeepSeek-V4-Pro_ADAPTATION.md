# DeepSeek-V4-Pro 适配与数据采集文档（aic-npu / Ascend 910B + vllm-ascend 0.23.0）

目标：让 `aic-npu` 能完成 `deepseek-ai/DeepSeek-V4-Pro` 在 `ascend_910b` + `vllm-ascend 0.23.0` 上的配置搜索，
并且**结果全部落在硅数据上**（`--database-mode SILICON` 不回落 SOL）。

上游参照：`aiconfigurator`（NVIDIA 版）已支持 `deepseek-ai/DeepSeek-V4-Pro`，架构名 `DeepseekV4ForCausalLM`，
模型族 `DEEPSEEKV4`，对标 vLLM 0.24.0。

| | |
|---|---|
| 创建日期 | 2026-10-10 |
| 本次修订 | 2026-10-12（重写为「适配 + 采集」手册，补齐 HCA / MoE 缺口与落位校验） |
| 分支 | `feature/deepseek-v4-pro-support` |
| 目标命令 | `aic-npu default --model-path /workspace/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json --total-gpus 64 --system ascend_910b --backend vllm-ascend --backend-version 0.23.0 --database-mode HYBRID --isl 4096 --osl 1024 --ttft 1800 --tpot 100 --top-n 10 --save-dir results` |

---

## 0. TL;DR：当前状态

| 类别 | 状态 |
|---|---|
| A/B/C/E（架构登记、算子建模、支持矩阵、本地冒烟） | ✅ 全部完成，无需硬件 |
| D3 `moe_perf.txt`（V4 维度） | ⚠️ 已采 `moe_v4`，但**命中不了**（F2），且 EP 覆盖不足（F3） |
| D4 `mhc_module_perf.txt` | ✅ 已采 `dsv4_mhc`，待落位 |
| D5/D6 `dsv4_csa_{context,generation}` | ✅ 已采 `dsv4_csa_ctx_v2` / `dsv4_csa_gen_v2`，待落位 |
| **D7/D8 `dsv4_hca_{context,generation}`** | ❌ **未采集 —— 当前最大阻塞项** |
| 落位到 `0.23.0/` 数据目录 | ❌ 8 张表里一张 dsv4 / mhc 都还没有 |

**一句话**：命令现在用 `HYBRID` 能跑完，但 31/61 层注意力（HCA）走 `SOL/0.5`、MoE 走 `SOL/0.4`，
口径混用，结论不可用；`SILICON` 会直接抛 `PerfDataNotAvailableError`。
补完 D7/D8 并修掉 F2/F3 后，`SILICON` 才能全链路命中硅数据。

**两个容易踩的坑（本次修订发现）**：

1. **并行维度会被 `--total-gpus` 扩展**：`tp/dp/moe_ep` 不止 `[1,2,4,8]`，
   64 卡下会到 **64**。所以 `num_heads` 要采到 **2**、`moe_ep` 要采 **64**（见 §3.2 / §3.6 / F8）。
2. **64 卡只是"刚好放得下"**：bf16 权重 ≈2894 GiB，只有 `ep=64` 可行，剩余 7~19 GB 给 KV，
   `--ttft 1800 --tpot 100` 很可能无解。建议先按 **128 卡**评估（见 §8.0 / F9）。

---

## 1. 架构拆解：V4-Pro 关键参数

来源：`aiconfigurator/aic-core/src/aiconfigurator_core/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json`
已同步到 aic-npu 的 `model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json`。

| 参数 | 值 |
|------|-----|
| `hidden_size` | 7168 |
| `num_hidden_layers` | 61 |
| `num_attention_heads` / `num_key_value_heads` | 128 / **1**（压缩 KV 为 MQA） |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | 1536 / 1024 / 16 |
| `head_dim` | 512（压缩 KV latent 维度） |
| `qk_rope_head_dim` | 64 |
| `index_head_dim` / `index_n_heads` / `index_topk` | 128 / 64 / 1024 |
| `sliding_window` | 128 |
| `n_routed_experts` / `num_experts_per_tok` | 384 / 6 |
| `moe_intermediate_size` | 3072 |
| `n_shared_experts` | 1 |
| `num_nextn_predict_layers` | 1（MTP） |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 20 |
| `vocab_size` | 129280 |
| `max_position_embeddings` | 1048576 |
| `compress_ratios` | 61 项：`128, (4,128)×30, 0` |

V4-Pro 与 V3/R1、V3.2/GLM-5 都不同：

| 维度 | V3 / R1 | V3.2 / GLM-5 | **V4-Pro** |
|------|---------|--------------|------------|
| 架构名 | `DeepseekV3ForCausalLM` | `DeepseekV32ForCausalLM` / `GlmMoeDsaForCausalLM` | `DeepseekV4ForCausalLM` |
| 模型族 | `DEEPSEEK` | `DEEPSEEKV32` | **`DEEPSEEKV4`**（新增） |
| 注意力表 | 内核级 MLA 表 | 模块级 DSA 表 | **模块级 DSv4 表（CSA/HCA 各一对）+ mHC 表** |
| KV latent | `kv_lora_rank=512` + `qk_nope 128` + `v_head 128` | 同 V3 | **`head_dim=512` 单一 latent，`num_key_value_heads=1`（MQA）** |
| 输出投影 | `o_proj: [h*v_dim → h]` | 同 V3 | **分组低秩 `o_groups=16` / `o_lora_rank=1024` 两段投影** |
| 稀疏选择 | 无 | indexer topk | **CSA（compress=4）带 indexer；HCA（compress=128）无 indexer；SWA（0）并入 HCA** |
| 残差流 | 普通 residual | 普通 residual | **mHC，`hc_mult=4`，每层 pre/post 两处** |
| Dense 层 | `first_k_dense_replace=3` | — | **无 dense 层，61 层全 MoE** |
| MoE | 256 experts / topk 8 / inter 2048 | 同 V3 | **384 experts / topk 6 / inter 3072** |

### 1.1 `compress_ratios` 语义 → 注意力算子聚合

| ratio | 含义 | 层数（Pro） | 是否有 indexer |
|-------|------|------------|----------------|
| `0` | SWA（纯滑窗） | 1 | 否 |
| `4` | CSA（Compressed Sparse Attention，4× 压缩 + indexer topk） | 30 | 是 |
| `128` | HCA（Highly-Compressed Attention，128× 压缩） | 30 | 否 |

建模时 **SWA(0) 折进 HCA(128)**（上游 `deepseek_v4.py:128`：`ratio_counts[128] += ratio_counts.pop(0, 0)`），
最终注意力算子聚合为 **CSA 一组（30 层）+ HCA 一组（31 层）**。

> **这条直接决定采集清单**：CSA 和 HCA 是**两张独立的表、两个独立缓存**
> （`perf_database.py` 里 `attr = f"_context_dsv4_{attn_kind}_module_data"`），
> **HCA 不是可选项**——61 层里有 31 层读它。只采 CSA = 一半注意力走 SOL。

### 1.2 注意力算子拆解（每层）

对齐上游 `operators/dsv4.rs` 的权重公式反推出的计算图：

| # | 子算子 | 形状 | 量化 |
|---|--------|------|------|
| 1 | `q_a_proj` | `[T, h] × [h, q_lora]` | gemm |
| 2 | `q_b_proj` | `[T, q_lora] × [q_lora, heads × head_dim]` | gemm |
| 3 | `kv_a_proj` | `[T, h] × [h, head_dim]`（MQA，1 个 KV head） | gemm |
| 4 | compressor（ratio≠0） | `2 × ratio_mult × [h, head_dim]`，`ratio_mult = 2 if ratio==4 else 1` | gemm |
| 5 | indexer（仅 ratio==4） | `wq_b: [q_lora → index_n_heads × index_head_dim]`、`weights_proj: [h → index_n_heads]`、`2 × [h → index_head_dim]` | gemm / bf16 |
| 6 | indexer FP8 MQA logits + topk | `[T, index_n_heads, index_head_dim] × [KV_compressed, index_head_dim]` | fp8 |
| 7 | 稀疏注意力（滑窗 + 压缩 KV） | `QK: 2·heads·pairs·(head_dim+qk_rope)`、`PV: 2·heads·pairs·head_dim` | fmha |
| 8 | `o_proj` 一段（bf16） | `[T, heads × head_dim] × [heads × head_dim, o_lora]` | float16 |
| 9 | `o_proj` 二段（分组） | `[T, o_groups × o_lora] × [o_groups × o_lora, h]` | gemm |

> `head_dim + qk_rope_head_dim = 576`，与 V3 的 `kv_lora_rank + qk_rope = 576` 一致；
> V 维度 = `head_dim = 512`，与 V3 的 `kv_lora_rank = 512` 一致。
> 即 **V4 的注意力核心就是 V3 的 MLA，只是 KV 被压缩、o_proj 换成两段分组低秩**。

### 1.3 mHC 算子拆解（每层 pre / post 各一处）

对齐上游 `operators/mhc.rs`：

```
hc_dim  = hc_mult × h
mix_hc  = (2 + hc_mult) × hc_mult
sites   = 2                      # attn mHC + FFN mHC
pre_ops  = sites × (2·T·hc_dim·mix_hc + T·hc_dim·3 + T·(hc²+2hc)·sinkhorn + 2·T·hc·h)
post_ops = sites × (2·T·hc·hc·h + 2·T·hc·h)
weights  = 2 × (mix_hc × hc_dim + mix_hc + 3) × bytes_per_element
```

mHC 与 router / logits 一样，量化**固定 bf16**（上游 `deepseek_v4.py` 多处 `GEMMQuantMode.bfloat16`）；
aic-npu 侧 `models.py` 构造 mHC op 时硬编码 `common.GEMMQuantMode.float16`，**且 mHC 表没有量化列**。

### 1.4 KV cache 拆解（非线性，分段）

对齐上游 `DeepSeekV4Model.get_kvcache_bytes_per_sequence()`。每层每序列：

```
sliding 部分：min(seq_len, sliding_window) × head_dim          # 所有层都有
压缩部分  ：ratio != 0 时  (seq_len // ratio) × head_dim
CSA 额外  ：2 × ratio × coff × head_dim 个 FP32（coff = 2 if ratio==4 else 1）
CSA       ：+ 压缩项 × indexer 条目（FP4 = index_head_dim × 0.5 字节/元素）
CSA       ：+ 2 × ratio × 2 × index_head_dim 个 FP32
```

aic-npu 的内存模型是线性的，因此拆成两项供 backend 使用：

- `get_kvcache_elements_per_token()` = `Σ_layers head_dim / ratio`（ratio=0 记 0）→ Pro = `30×128 + 30×4 = 3960`
- `get_kvcache_sliding_window_elements()` = `layers × sliding_window × head_dim` → Pro = `61 × 128 × 512`

---

## 2. 目标命令会读哪些表（采集清单的唯一依据）

`aic-npu default ... --isl 4096 --osl 1024 --total-gpus 64` 的查询路径：

```
config.json architectures[0] = "DeepseekV4ForCausalLM"
    ↓ common.ARCHITECTURE_TO_MODEL_FAMILY              → "DEEPSEEKV4"
    ↓ utils._parse_hf_config_json()                    → extra_params = DeepSeekV4Config(...)
    ↓ sdk/models.py get_model()                        → DeepSeekV4Model
    ↓ context_ops / generation_ops
    │     ├─ DeepSeekV4MHCModule (pre/post)            → mhc_module_perf.txt
    │     ├─ ContextDeepSeekV4AttentionModule   ×2     → dsv4_csa_context_module_perf.txt
    │     │                                            → dsv4_hca_context_module_perf.txt
    │     ├─ GenerationDeepSeekV4AttentionModule ×2    → dsv4_csa_generation_module_perf.txt
    │     │                                            → dsv4_hca_generation_module_perf.txt
    │     ├─ GEMM（shared gate/up/down、router、logits）→ gemm_perf.txt
    │     ├─ MoE / MoEDispatch                         → moe_perf.txt / custom_allreduce_perf.txt
    │     └─ Embedding / ElementWise / P2P
    ↓ PerfDatabase.query_*()
```

| # | 表文件 | 采集脚本 | 状态 |
|---|--------|----------|------|
| D1 | `gemm_perf.txt` | `collector/npu/collect_gemm.py` | ✅ 已有（通用表，覆盖 7168 / 3072） |
| D2 | `custom_allreduce_perf.txt` | `collector/npu/collect_hccl.py` | ✅ 已有（通用表，`BIZ_HIDDENS` 含 7168） |
| D3 | `moe_perf.txt` | `collector/npu/collect_moe.py` | ⚠️ `moe_v4` 已采，待修 F2/F3 后重转 |
| D4 | `mhc_module_perf.txt` | `collector/npu/collect_mhc.py` | ✅ 已采 `dsv4_mhc` |
| D5 | `dsv4_csa_context_module_perf.txt` | `collector/npu/collect_dsv4_attn.py` | ✅ 已采 `dsv4_csa_ctx_v2` |
| D6 | `dsv4_csa_generation_module_perf.txt` | 同上 | ✅ 已采 `dsv4_csa_gen_v2` |
| D7 | `dsv4_hca_context_module_perf.txt` | 同上 | ❌ **待采** |
| D8 | `dsv4_hca_generation_module_perf.txt` | 同上 | ❌ **待采** |

### 2.1 上游多出来的 `dsv4_*` 表：不需要采

上游 `b300_sxm/sparse_attention/vllm/0.24.0` 下比 aic-npu 多出
`dsv4_hca_attn_module_perf.parquet`、`dsv4_paged_mqa_logits_module_perf.parquet`
（enum 里还有 `dsv4_csa_attn_module`、`dsv4_csa_topk_calib`、`dsv4_megamoe_module`）。

**aic-npu 一律不读它们**——`PerfDataFilename`（`sdk/common.py:662-667`）只登记了 5 张 DSv4 相关表，
没有对应的 loader / query。上游自己也把它们标记为废弃或占位：

| 上游表 | 上游结论 | aic-npu 是否需要 |
|--------|---------|------------------|
| `dsv4_csa_attn` / `dsv4_hca_attn` | `operations/dsv4.py:178-184`：「design placeholders with no query path and were retired（csa_attn 从未产出数据）」 | ❌ |
| `dsv4_paged_mqa_logits` | 唯一消费者是 Rust `operators/dsv4.rs:458 query_context_cp`，仅当 `cp_size > 1`；Python opspec 目前**不下发 `cp_size`**（`_reject_cp` 仍 guard） | ❌（aic-npu 不建模 CP，见 §5.3） |
| `dsv4_csa_topk_calib` | 同上，CP 路径的 topK Δ 修正 | ❌ |
| `dsv4_megamoe` | SGLang + Blackwell（SM≥100）专有 kernel | ❌（见 §5.2） |

**结论：不要去补采这几张表。缺的只有 HCA 的 context / generation。**

---

## 3. 采集矩阵：每张表必须覆盖的 key

### 3.1 量化 key（**精确匹配，不插值**）

`--model-path .../deepseek-ai--DeepSeek-V4-Pro_config.json` 的 `quantization_config.quant_method = "fp8"`
会被 `_infer_quant_modes_from_raw_config()` 推成 `fp8_block`，再由 C6 的重映射
（`sdk/models.py:190-206`，`backend_name == "vllm-ascend"` 分支）落到：

| 目标 | 推理值 | 重映射后（**采集必须写这个值**） |
|------|--------|------------------------------|
| GEMM | `fp8_block` | **`float16`** |
| MoE | `fp8_block` | **`float16`** |
| FMHA | `fp8` | **`float16`** |
| KV cache | `fp8` | **`float16`** |

> 注意：C6 最终选的是 **float16 而不是 `sq`**（`models.py:176-189` 有完整理由：
> `sq` 在 910B decode 区间实测比 bf16 慢，`default` 入口又不暴露 `--gemm-quant-mode`）。
> 因此采集器的 `--mla-dtype/--kv-cache-dtype/--gemm-type` 三个默认值（都是 `float16`）**正好对齐，不用改**。

### 3.2 `num_heads`（**精确 key，不插值**）—— 注意 tp 不止到 8

`num_heads = num_attention_heads(128) // tp`。
基础列表是 `tp_list = [1,2,4,8]`（`sdk/task.py:255-272`），**但 `TaskConfig._widen_gpu_sweep_lists`
（`sdk/task.py:31-53`）会把 `tp_list` / `dp_list` / `moe_ep_list` / `num_gpu_per_worker`
再补上 16 / 32 / 64 / 128 / 256 中所有 `<= --total-gpus` 的档位**——因为
bf16 权重约 **2894 GiB（≈3.1 TB）**，任何 ≤8 卡的 worker 都放不下。

因此 `--total-gpus 64` 时：

```
tp ∈ {1,2,4,8,16,32,64}   →   num_heads = 128 // tp ∈ {128,64,32,16,8,4,2}
```

**七个值都要采集**（缺一个，对应 tp 的配置就回落 SOL）。
上游/旧文档写的 `128/64/32/16` 是**未考虑维度扩展**的错误结论。

### 3.3 context 表（`data[fmha][kv][gemm][num_heads][s][b]`）

| 轴 | 目标命令取值 | 要求网格 |
|---|-------------|---------|
| `s`（= `isl + prefix`，`prefix=0`） | **4096** | 必须含 4096；建议 `128…16384` 全序列 |
| `b` | 受 OOM 限制，实测很小（≤32） | `1 2 4 8 16 32` 足够 |

> ⚠️ **网格不能退化（这是最容易翻车的一条）**
>
> `query_context_dsv4_module` 走 `_interp_3d(..., "cubic")` → `_interp_2d_1d`，
> 会在**每个 `num_heads` 下**对 `(s, b)` 建一个二维 Delaunay 三角剖分再做三次插值。
> 只要某个轴只剩**一个取值**，所有点共线，scipy 直接抛
> `QhullError: Initial simplex is flat` —— **整张表废掉，连 `b=1` 都查不出来**
> （不是"缺哪个点回落哪个点"）。
>
> 所以每个 `num_heads` 必须满足：`b` 至少 2 个不同取值、`s` 至少 2 个不同取值，
> 且 `(heads, b)` 单元内在目标 `s` 附近有 ≥2 个点。
> 只采 `--batch-list 1` 是最典型的踩法。

### 3.4 generation 表（`data[kv][gemm][num_heads][b][s]`）

| 轴 | 目标命令取值 | 要求网格 |
|---|-------------|---------|
| `s`（= `isl + step`） | **4096 ~ 5120** | 必须含 4096 与 ≥5120 的上界（8192） |
| `b` | agg `max_batch_size=512`、disagg `decode_max_num_tokens=512`（`pareto_analysis.py:115`、`inference_session.py:709`） | **必须到 512**，仅到 256 会被边界外推 |

### 3.5 mHC 表（`data[op_name][hc_mult][hidden_size][num_tokens]`）

| 轴 | 值 |
|---|-----|
| `op_name` | `pre` 和 `post` 两个都要（`query_mhc_module` 里写死） |
| `hc_mult` | `4`（精确 key） |
| `hidden_size` | `7168`（精确 key） |
| `num_tokens` | 插值轴；context 侧 = `b×isl`，上限 `prefill_max_num_tokens=16384` → 网格至少到 16384 |

### 3.6 MoE 表（**全部精确 key**）

```
data[quant][distribution][topk][num_experts][hidden][inter][moe_tp][moe_ep][num_tokens]
```

| key | 值 |
|-----|-----|
| `quant` | `float16` |
| `distribution` | **`power_law_1.01`**（V4 的 `_power_law_alpha`，见 F2） |
| `topk` / `num_experts` | `6` / `384` |
| `hidden` / `inter` | `7168` / `3072` |
| `moe_tp` | `1`（vllm-ascend 固定，`moe_tp_list=[1]`） |
| `moe_ep` | **64（唯一可行档）**，16 / 32 作为保险，见下方说明 |
| `num_tokens` | 插值轴；需到 16384 |

`moe_ep` 同样受 `_widen_gpu_sweep_lists` 扩展 → `{1,2,4,8,16,32,64}`。
但按每卡常驻权重估算（`oth/tp + moe/ep`，`oth≈11.6 GiB`、`moe≈2882 GiB`）：

| ep | 每卡权重（tp=8） | 64 GB 卡是否放得下 |
|----|-----------------|-------------------|
| 64 | **≈46 GiB** | ✅ 唯一可行 |
| 32 | ≈92 GiB | ❌ |
| 16 | ≈182 GiB | ❌ |
| ≤8 | ≥360 GiB | ❌ |

所以 **`ep=64` 是必须采的**，16 / 32 采集成本极低（本地专家数 = 384/ep，越大越省显存），
顺手一起采即可。

---

## 4. 采集命令（可直接复制执行）

### 4.0 公共前提（NPU 环境，2026-10-12 实测）

CSA/HCA 采集走 `--module-source framework`（vllm-ascend 自己的 DSv4 attention 类），
跑通真实 metadata 路径需要以下前提，均已在 `collector/npu/dsv4_factory.py` 内处理，
**但第 1 条同时需要启动环境配合**：

1. **vendor 自定义算子包必须在加载路径上**。DSA 元数据算子
   （`torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata`）的 aclnn 实现在
   vllm-ascend 自带的 custom op 包 `libcust_opapi.so` 里，torch_npu 的
   FunctionLoader 只在 `libopapi.so` / `libopapi_serv.so` 里找符号，**永远找不到**。

   ```bash
   export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
   export LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
   ```

   `_build_dsv4_real_metadata()` 里也会调 `bootstrap_custom_op_env(include_vendor_lib=True)` 做进程内兜底。
   **缺失症状**：每个 spec 报 `aclnnSparseAttnSharedKvMetadata ... not in libopapi.so`，随后全部静默回退
   到 `dsa_forward` profiling 兜底路径——采集"成功"但数据不真实。

2. **CSA/HCA 的 compress_ratio 必须写进 HF config**。`DeepseekV4Attention` 只从
   `hf_config.compress_ratios[layer_idx]` 读压缩率，不认任何构造参数。
   factory 在构造模块前把 `spec.compress_ratio`（CSA=4 / HCA=128）写入 `compress_ratios[0]`。
   **缺失症状**：所有 "CSA" 行实际测的是 HCA（ratio 128）。

3. **attn_metadata dict 的 key 必须是各缓存模块自己的构造 prefix**。
   `ops/dsa.py::dsa_forward` 的 `filter_metadata()` 按 key 排序后把 values 交给
   `AscendDSAImpl._forward_prefill()` 按 ratio **位置解包**（ratio 128 期望 3 个、ratio 4 期望 5 个）。
   **缺失症状**：`ValueError: too many values to unpack (expected 3)`。

4. **rope 运行时 buffer 必须按 spec 重建**。`rope_dsv4._ROPE_STATE.runtime_buffer` 是进程级全局缓存，
   按**首个** spec 的 `max_num_batched_tokens` 分配后永不扩容；generation 扫描 batch=1 在前会让后续大 batch
   在 `get_cos_and_sin_dsa` 里广播失败。factory 每个 spec 构建模块前清空该缓存。
   **缺失症状**：`shape of broadcast result should be [N,1,1,64], but self is [1,1,1,64]`。

5. **wo_a 权重需整形为 3-D 分组布局**。单卡路径 `_forward_o_proj` 的 `else` 分支用
   `npu_transpose_batchmatmul` 要求权重是 3-D；vLLM 的 `ColumnParallelLinear` 建出来的是 2-D。
   生产环境走 finegrained o_lora TP 从不触发该分支，采集器 TP=1 会触发，故在模块构建后整形。
   **缺失症状**：`IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)`。

6. **图模式（NPU graph）是默认口径，上游 NVIDIA 版也一样用图模式**。

   `benchmark_npu` 默认先 `torch.npu.NPUGraph()` 捕获再回放（消除 Python 派发开销），
   失败才回落 eager。上游 `collector/vllm/collect_dsv4_attn.py` 对同一个 DSv4 模块
   用的就是 `benchmark_with_power(..., use_cuda_graph=True)`，而且**没有开**
   `allow_graph_fail`（默认 `False`）—— 即**图捕获失败会直接抛错**，上游预期它必定成功。

   上游能成功的原因，正是下面这个结构（`collect_dsv4_attn.py:486-541`）：

   ```python
   common   = _make_common_metadata(...)                       # metadata 预构建
   metadata = _build_metadata_and_bind_caches(attn_module, vllm_config, common, ...)

   with set_current_vllm_config(vllm_config), set_forward_context(metadata, vllm_config), \
        torch.inference_mode():
       attn_module(positions, hidden_states, None)             # 预跑一次
       torch.cuda.synchronize()

       def kernel_func(...):
           attn_module(positions, hidden_states, None)          # ← 计时区里只有这一行

       with benchmark_with_power(..., use_cuda_graph=True) as result:
           pass
   ```

   即 **metadata 与 vLLM forward context 都在计时闭包之外**，`kernel_func` 是纯设备前向，
   所以可以捕获。aic-npu 的 `dsv4_factory` 已经是同样的结构（`_create_dsv4_kv_cache_and_metadata`
   在外、`forward_fn` 只调 `module.forward`）。

   **所以：`--no-graph` 不是默认项，只是逃生开关。**
   只有当 `module.forward` 内部（vllm-ascend 的 `compressor_metadata` / indexer 等自定义算子）
   仍然做主机→设备拷贝、导致捕获必失败时才加：

   ```
   NPUGraph: ERROR — capture failed: RuntimeError: copy_between_host_and_device_opapi
   ... aclrtMemcpy, error code is 107030
   Not_Supported(EE1016): Synchronizing a stream failed. Stream (stream_id=7)
   during the capture stage is not supported.
   ```

   判定方法：看每个点的日志尾部是 `graph` 还是 `eager (...)`。
   若显示 `graph` 就说明图模式成功，不需要 `--no-graph`。

7. **eager 口径下必须把"每轮不变的工作"挪出计时区**。
   早期实现把 `set_current_vllm_config` / `set_weight_prefetch_method` /
   `torch.tensor([num_tokens], device=...)` 都写在 `forward_fn` 内部，
   于是每一次计时迭代都要付一遍 vLLM forward context 的构造成本：
   实测 `b=1 s=128` 测出 **≈3981 µs**，且 heads=128/64/32/16/8/4 的值几乎一样
   （3981 / 3961 / 4068 / 3967 / 4002 / 3896 µs）——
   说明测到的是**固定的主机开销，不是 kernel 时间**，数据不可用。

   现已把这些常量工作提到 `forward_fn` 之外（只构造一次），`forward_fn` 只剩
   `module.forward(positions, hidden, None)`；forward context 保持打开并暴露
   `forward_fn.close`，由采集器在 spec 结束时释放。

   **自检方法**：冒烟时看日志里的延迟是否随 `num_heads` / `s` 单调变化。
   如果不同 `num_heads` 的值几乎相同（差异 < 5%），说明仍然在测主机开销。

> **旧数据作废**：2026-10-12 修复前的 `dsv4_*` 输出（profiling 兜底数据、ratio 128 冒充 CSA 的数据）
> **一律不可用**；重新采集务必使用**新的 `--output-dir`**——`--resume` 会跳过旧 checkpoint 里已完成的 key。

### 4.1 D7 / D8：HCA（**当前最缺，先跑这两个**）

四个组合（csa/hca × context/generation）**必须各自一个 `--output-dir`**：
checkpoint 文件名共享，同一目录跑两批会互相覆盖已完成记录。

```bash
# ---------- D7: HCA context ----------
cd /workspace && \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds hca \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 2 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_hca_ctx_v2 --resume

# ---------- D8: HCA generation ----------
cd /workspace && \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_generation --attn-kinds hca \
  --batch-list 1 2 4 8 16 32 64 128 256 384 512 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 2 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_hca_gen_v2 --resume
```

> generation 的 `--batch-list` 到 **512**（不是 256）：agg 的 `max_batch_size=512`、
> disagg 的 `decode_max_num_tokens=512`，超过采集网格上界会被 `_nearest_1d_point_helper` 夹到边界外推。

### 4.2 D5 / D6：CSA（已采，若需重采/补点用同样参数）

```bash
cd /workspace && \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds csa \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 2 --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_csa_ctx_v2 --resume

# generation：把 --op-types 换成 dsv4_generation，--batch-list 加到 512
cd /workspace && \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_generation --attn-kinds csa \
  --batch-list 1 2 4 8 16 32 64 128 256 384 512 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 2 --module-source framework \
  --output-dir ./data/dsv4_csa_gen_v2 --resume
```

**容量限制（实测）**：`n_local_heads` 恒为 `128`（TP=1），context 路径的 `o_proj_input`
形状为 `[batch×seq, 128, 512]`：`b=16 × s=16384`（262144 tokens）时**单这一项就要 32 GiB**，
910B（64 GB）放不下。这类尾部规格采集器记 FAILED 后继续跑（`--resume` 可续采），属单卡容量极限。

`--module-source`：

| 取值 | 说明 |
|------|------|
| `framework`（默认） | 实例化 vllm-ascend 自己的 DSv4 attention 类。类名靠候选模块探测，可用 `--attention-cls module.path:ClassName` 显式指定 |
| `reference` | 内置 torch 参考实现（与 SDK 的 SOL 拆解同构），不依赖 vllm-ascend 内部布局，可先跑通拿量级 |

换环境首次跑建议先冒烟（2 个点）再全量：

```bash
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds hca \
  --batch-list 1 --seq-len-list 128 --num-heads-list 128 \
  --module-source framework --warmup-iters 1 --bench-iters 3 \
  --output-dir ./data/dsv4_hca_smoke
```

### 4.3 D4：mHC（已采 `dsv4_mhc`）

```bash
python collector/npu/collect_mhc.py \
  --op-types pre post \
  --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  --hc-mult 4 --hidden-size 7168 --sinkhorn-iters 20 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_mhc --resume
```

2 op × 16 tokens = 32 个点，纯 GEMM + elementwise，几分钟可完成。

### 4.4 D3：MoE（**需按 F2/F3 重采重转**）

先补齐 EP 覆盖（F3 已在 `collector/npu/collect_moe.py` 加了 `deepseek-v4-pro-ep64` / `-ep32` /
`-ep16` / `-ep4` / `-ep2`；**ep=64 是唯一可行部署档，必须采**）：

```bash
# 1) 冒烟（2 个点 × 3 个 EP 档）
python collector/npu/collect_moe.py \
  --models deepseek-v4-pro-ep64 deepseek-v4-pro-ep32 deepseek-v4-pro-ep16 \
  --quant-types bf16 --token-list 1 128 --warmup-iters 1 --bench-iters 3 \
  --output-dir ./data/moe_v4_smoke

# 2) 全量（token 档必须到 16384：context 侧 b×isl 可达该量级）
python collector/npu/collect_moe.py \
  --models deepseek-v4-pro-ep64 deepseek-v4-pro-ep32 deepseek-v4-pro-ep16 \
  --quant-types bf16 \
  --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  --output-dir ./data/moe_v4 --resume
```

`ep_size` 只影响**单卡本地专家数**（`local = 384 / ep`），是显存与实测延迟的折中：

| ep | local experts | 单卡采集显存 | 对应部署每卡权重 | 优先级 |
|----|---------------|--------------|------------------|--------|
| **64** | 6 | ~0.8 GB | **≈46 GiB ✅** | **必采** |
| 32 | 12 | ~1.6 GB | ≈92 GiB ❌ | 顺手采（保险） |
| 16 | 24 | ~3.2 GB | ≈182 GiB ❌ | 顺手采（保险） |
| 8 | 48 | ~6.3 GB | ≈360 GiB ❌ | 低 |
| 4 | 96 | ~12.7 GB | ≈721 GiB ❌ | 低 |
| 2 | 192 | ~25.3 GB | ≈1441 GiB ❌ | 低 |
| 1 | 384 | ~50.7 GB | ≈2882 GiB ❌ | 单卡还会 OOM，跳过 |

> 采集侧 ep 越大 → 本地专家越少 → **越省显存**；所以"必采的 ep=64"恰恰是最容易采的。

**转换 + 并入现有 `moe_perf.txt`（一次搞定，推荐 `--append`）**：

`moe_perf.txt` 是**所有 MoE 模型共用**的一张表，现有 0.23.0 里已经有 234 行
（DeepSeek-V3/R1、V2-Lite、MiniMax、Mixtral、Qwen …）。而 `convert_moe` 默认以 `"w"`
覆盖写，且只能看到 `--input-dir` 里放的 CSV（这里只有 V4），
**所以直接 `cp` 过去会把其它模型的行全部抹掉**。

加 `--append` 后转换器改为「以 `a` 模式写、跳过表头」，直接把 V4 行并进现有表，
并且会自动检查/补一个尾换行，不会把两行粘在一起：

```bash
python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/moe_v4 \
  --output-dir src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0 \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --moe-distribution power_law_1.01 \
  --append
```

> `--append` 只在目标文件**已存在**时追加；不存在时退化成正常写入（带表头），
> 所以首次建表也能用同一条命令。
> 注意它**不幂等**：对同一批 CSV 跑两次会得到两份重复行
> （`load_moe_data` 对重复 key 只保留先到的那条并打 debug 日志，不会报错，但数据是脏的）。
> 重跑前请先 `git checkout` 或备份 `moe_perf.txt`。

**等价的手工做法**（不想用 `--append` 时）：先转到隔离目录，再把数据行（去掉表头）追加：

```bash
# 1) 转到隔离目录，产出只含 V4 行的 moe_perf.txt
python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/moe_v4 \
  --output-dir ./data/moe_v4_out \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --moe-distribution power_law_1.01

# 2) 追加数据行（tail -n +2 去掉表头）
SRC=./data/moe_v4_out/moe_perf.txt
DST=src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0/moe_perf.txt

# 保险：确保目标文件以换行结尾，否则追加会把最后一行和第一行粘在一起
if [ -s "$DST" ] && [ -n "$(tail -c1 "$DST")" ]; then printf '\n' >> "$DST"; fi

tail -n +2 "$SRC" >> "$DST"
```

<details>
<summary>Windows / PowerShell 备选（本项目主要在 BC-Linux 上操作，仅备查）</summary>

```powershell
$src = "./data/moe_v4_out/moe_perf.txt"
$dst = "src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0/moe_perf.txt"
Get-Content -Encoding utf8 $src | Select-Object -Skip 1 | Add-Content -Encoding utf8 $dst
```

</details>

### 4.5 落位

落位目标目录统一记作：

```bash
DATA_DIR=src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
```

**为什么是「5 张表 `cp` + MoE 追加」**：

| 表 | 目标目录现状 | 落位方式 | 原因 |
|---|-------------|---------|------|
| `dsv4_{csa,hca}_{context,generation}_module_perf.txt`（4 张） | **原本不存在** | 直接 `cp` | 新表，覆盖没有副作用 |
| `mhc_module_perf.txt`（1 张） | **原本不存在** | 直接 `cp` | 同上 |
| `moe_perf.txt` | **已存在 234 行** | **追加（不能 cp）** | 全模型共用表，`cp` 会抹掉 V3/R1 等 |

一次跑完（Linux）：

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
mkdir -p "$DATA_DIR"

# ---------- 1) 5 张新表：直接 cp ----------
cp ./data/dsv4_mhc/mhc_module_perf.txt                        "$DATA_DIR/"
cp ./data/dsv4_csa_ctx_v2/dsv4_csa_context_module_perf.txt    "$DATA_DIR/"
cp ./data/dsv4_hca_ctx_v2/dsv4_hca_context_module_perf.txt    "$DATA_DIR/"
cp ./data/dsv4_csa_gen_v2/dsv4_csa_generation_module_perf.txt "$DATA_DIR/"
cp ./data/dsv4_hca_gen_v2/dsv4_hca_generation_module_perf.txt "$DATA_DIR/"

# ---------- 2) MoE：转换 + 追加（不要 cp！）----------
python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/moe_v4 \
  --output-dir "$DATA_DIR" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --moe-distribution power_law_1.01 \
  --append

# ---------- 3) 自检 ----------
echo "moe rows      : $(($(wc -l < "$DATA_DIR/moe_perf.txt") - 1))"   # 应 = 234 + V4 行数
echo "distributions : $(cut -d, -f14 "$DATA_DIR/moe_perf.txt" | sort -u | tr '\n' ' ')"
head -1 "$DATA_DIR/dsv4_hca_context_module_perf.txt"                  # 表头列名应为 11 列

# ---------- 4) 完整性校验 ----------
python tools/check_dsv4_data.py --isl 4096 --osl 1024
```

> 想先用转换工具改写 `framework/version/device` 三列再落位也可以（等价于上面的 `cp`）：
>
> ```bash
> python tools/convert_to_aiconfigurator.py \
>   --input-dir ./data/dsv4_all \
>   --output-dir "$DATA_DIR" \
>   --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
>   --mla-output none --dsv4-output --mhc-output
> ```
> 注意这条**不要**带 `--append`——它是给 4+1 张新表用的；`moe_perf.txt` 仍按第 2 步单独处理
> （否则若 `./data/dsv4_all` 里恰好放了 MoE CSV，会被追加一份进去）。

### 4.6 落位后校验（**必做**）

```bash
python tools/check_dsv4_data.py --isl 4096 --osl 1024
```

该脚本按 §3 的 key 矩阵逐项做 SILICON 探针查询，输出每张表的命中情况与网格边界，
有硬缺失时以非零退出码结束。详见 §7。

---

## 5. 已发现的缺陷与修复计划

| # | 缺陷 | 影响 | 修复 | 状态 |
|---|------|------|------|------|
| **F1** | 未采集 `dsv4_hca_{context,generation}` | 31/61 层注意力走 `SOL/0.5`；`SILICON` 直接抛错 | 按 §4.1 补采 | ⬜ 待执行（需 NPU） |
| **F2** | MoE `distribution` key 不匹配：转换脚本硬编码 `power_law_1.2`，而 `DeepSeekV4Model._power_law_alpha = 1.01`（`models.py:1958`，与上游 `deepseek_v4.py:83` 一致）→ 查询 key 是 `power_law_1.01`，表里只有 `power_law_1.2` → 回落 `uniform` → `IndexError` → HYBRID 兜底 `SOL/0.4` | **MoE 硅数据 100% 用不上**（V3/R1/V3.2 同样受影响） | ① `convert_to_aiconfigurator.py` 新增 `--moe-distribution`（`power_law_1.2` / `power_law_1.01` / `both`）；② `query_moe` 的 distribution 回落改为「精确 → 同族 `power_law_*` → `uniform`」 | ✅ 已修复 |
| **F3** | `collect_moe.py` 只登记 `ep_size=8`，而搜索空间 `moe_ep_list=[1,2,4,8]` | ep=1/2/4 的配置全部回落 SOL | 新增 `deepseek-v4-pro-ep2` / `-ep4` 两条 `MODEL_CONFIGS` | ✅ 已修复 |
| **F4** | `collect_dsv4_attn.py` 的 `DEFAULT_GENERATION_BATCH_LIST` 只到 256，运行时 b 可达 512 | b>256 被边界外推 | 默认列表补 `384 512` | ✅ 已修复 |
| **F5** | 无落位后校验手段，缺表/缺 key 只能等 `default` 跑到一半才发现 | 排查成本高 | 新增 `tools/check_dsv4_data.py`（§7） | ✅ 已修复 |
| **F6** | 文档 §D1 称「`distribution` 固定 `power_law_1.2`，与现有 V3 行一致」——错误 | 误导落位 | 本文档 §3.6 / §4.4 已更正 | ✅ 已修复 |
| **F7** | `0.23.0` 目录当前**一张** dsv4 / mhc 表都没有，采集结果尚未落位 | 全链路仍走 SOL | 按 §4.5 落位 + §4.6 校验 | ⬜ 待执行 |
| **F8** | 采集矩阵按「tp/ep ≤ 8」制定，但 `TaskConfig._widen_gpu_sweep_lists`（`sdk/task.py:31-53`）会把 tp/dp/moe_ep 扩展到 16/32/64…≤ `--total-gpus`。64 卡下 tp 到 64、ep 到 64 | `num_heads` 缺 8/4/2、`moe_ep` 缺 16/32/64 → 全部可行配置都回落 SOL | `collect_dsv4_attn.py` 默认 heads 补 `8 4 2`；`collect_moe.py` 补 `ep64/ep32/ep16`；文档 §3.2/§3.6 更正 | ✅ 已修复 |
| **F9** | 未评估规模可行性：bf16 权重 ≈2894 GiB，64 GB 卡只有 `ep=64` 放得下，剩余 7~19 GB | 64 卡 + 严 SLA 可能无解，容易被误判为"数据缺失" | 新增 §8.0 规模分析；建议按 128 卡评估 | ✅ 已澄清（待确认业务规模） |
| **F10** | 落位只有 PowerShell 片段，且 MoE 与另外 5 张表的处理方式混在一起说明，容易把 `moe_perf.txt` 整体 `cp` 覆盖掉现有 234 行 | 丢失 V3/R1 等模型的 MoE 硅数据 | §4.4/§4.5 改为 bash 为主；转换器新增 `--append`（自动补尾换行）；明确「5 张新表 cp + MoE 追加」 | ✅ 已修复 |
| **F11** | **网格退化**：每个 `num_heads` 下 `(s, b)` 轴只有 1 个取值时，`_interp_2d_1d` 的二维三次插值抛 `QhullError`，**整张表不可用**（连存在的那个 b 也查不出来），而不是"缺哪个点回落哪个点" | 已采集的表 100% 白采 | §3.3 加了醒目说明；`check_dsv4_data.py` 新增网格退化诊断（逐 heads 报 b/s 取值数、单 s 单元数、够不到目标 s 的单元数） | ✅ 已修复（数据侧待重采） |
| **F12** | `--append` 不幂等，落位脚本被重复执行 → `moe_perf.txt` 里同一 key 出现多份（`load_moe_data` 只保留先到的那条且仅 debug 日志） | 表膨胀、难以人工核对 | `check_dsv4_data.py` 新增重复行检测（warning）；§4.4 明确"重跑前先 `git checkout`" | ✅ 已修复（数据侧待清理） |

### F2 详细说明（实测证据）

本地对 0.23.0 库做探针（V3 的 `7168/2048/8/256/tp1/ep8`）：

```
dist keys: ['power_law_1.2']

power_law_1.01 SILICON  → IndexError（list index out of range）
power_law_1.01 HYBRID   → 3.53 ms    （SOL / 0.4 兜底）
power_law_1.2  SILICON  → 2.03988 ms （真硅数据）
```

即：**表里有什么 key 决定能不能命中，而 V3/R1/V3.2/V4 查的都是 `power_law_1.01`。**
F2 的第 ② 项修复（回落同族 `power_law_*`）让现有 `power_law_1.2` 行立即生效，无需重新转换历史数据；
第 ① 项让后续采集可以写出语义正确的 key。

> ⚠️ **行为变更**：修复后 DeepSeek-V3/R1/V3.2/V4 的 MoE 会从 `SOL/0.4` 切到真实硅数据，
> 这些模型的历史 `default` 结果数值会变化（更准，且通常更快）。这是预期的。

---

## 6. 任务清单（A/B/C/E 已完成，存档）

### A 类：架构登记与配置解析（无需硬件）

| # | 任务 | 位置 | 状态 |
|---|------|------|------|
| A1 | 新增 V4 离线 HF config | `model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json` | ✅ |
| A2 | `ModelFamily` 加 `DEEPSEEKV4`；`ARCHITECTURE_TO_MODEL_FAMILY` 加 `DeepseekV4ForCausalLM` | `sdk/common.py` | ✅ |
| A3 | 新增 `DeepSeekV4Config` dataclass + `DEEPSEEK_V4_HF_MODELS` + `DefaultHFModels` 登记 + `deepseek_v4_indexer_cache_entry_bytes()` | `sdk/common.py` | ✅ |
| A4 | `_parse_hf_config_json` 新增 `DeepseekV4ForCausalLM` 分支 | `sdk/utils.py` | ✅ |

### B 类：算子与建模（无需硬件）

| # | 任务 | 位置 | 状态 |
|---|------|------|------|
| B1 | `PerfDataFilename` 新增 5 张表 | `sdk/common.py` | ✅ |
| B2 | `load_context_dsv4_module_data` / `load_generation_dsv4_module_data` / `load_mhc_module_data` | `sdk/perf_database.py` | ✅ |
| B3 | `query_context_dsv4_module` / `query_generation_dsv4_module` / `query_mhc_module`（含 SOL 解析估算） | `sdk/perf_database.py` | ✅ |
| B4 | `DeepSeekV4MHCModule` / `ContextDeepSeekV4AttentionModule` / `GenerationDeepSeekV4AttentionModule` | `sdk/operations.py` | ✅ |
| B5 | `DeepSeekV4Model`（compress_ratios 聚合、mHC pre/post、共享专家、MTP、KV cache 建模） | `sdk/models.py` | ✅ |
| B6 | `get_model()` 分派、`check_is_moe`、`_apply_model_quant_defaults` 适配 | `sdk/models.py` | ✅ |

### C 类：`aic-npu default` / 支持矩阵适配（无需硬件）

| # | 任务 | 位置 | 状态 |
|---|------|------|------|
| C1 | 支持矩阵把 `vllm` 版本提升到 `0.23.0`，并登记 V4-Pro 的 agg / disagg | `src/aiconfigurator_npu/systems/support_matrix.csv` | ✅ |
| C2 | MTP（nextn=1）白名单加 `DEEPSEEKV4` | `sdk/task.py:_base_common_layer` | ✅ |
| C3 | `validate()` 的注意力表 key 选择加 V4 分支 | `sdk/task.py:validate` | ✅ |
| C4 | backend 激活/KV cache 内存建模加 `DEEPSEEKV4` 分支 | `sdk/backends/trtllm_backend.py`、`sglang_backend.py` | ✅ |
| C5 | `supported_quant_mode` 注册 V4 表 | `sdk/perf_database.py` | ✅ |
| C6 | vllm-ascend 上把 HF FP8/FP4 推导出的 NVIDIA 量化模式重映射到 Ascend 可用表项 | `sdk/models.py:_apply_model_quant_defaults` | ✅ |

C6 的重映射表（**采集时的量化 key 以此为准**）：

| 目标 | NVIDIA FP8/FP4 模式 | Ascend 表项 |
|------|--------------------|-------------|
| GEMM | `fp8` / `fp8_static` / `fp8_block` / `fp8_ootb` / `nvfp4` | `float16` |
| MoE | `fp8` / `fp8_block` / `nvfp4` / `w4afp8` / `w4a16_mxfp4` / `w4a8_mxfp4_mxfp8` | `float16` |
| FMHA | `fp8` | `float16` |
| KV cache | `fp8` | `float16` |

### E 类：本地冒烟验证（无卡）

| # | 命令 | 结果 |
|---|------|------|
| E1 | `aic-npu support --model deepseek-ai/DeepSeek-V4-Pro --system ascend_910b --backend vllm` | ✅ `Version 0.23.0`，agg / disagg 均 `YES` |
| E2 | `aic-npu default ... --database-mode HYBRID --total-gpus 128 --ttft 100000 --tpot 1000` | ✅ 跑完 agg（518 条）+ disagg（121 条），输出帕累托表 |
| E3 | `python tools/smoke_dsv4_model.py`（tp=1/2/4/8 构造模型） | ✅ 每相 2 个 `*_attention` op（CSA=30 / HCA=31）、mHC pre/post、共享专家/MoE 齐全 |
| E4 | `python tools/check_dsv4_data.py --isl 4096 --osl 1024` | ✅ 在表未落位时正确报出 8 项缺失 |

> Windows 控制台（`gbk` 编码）下打印最终汇总表时会报
> `UnicodeEncodeError: ... '\u2022'`，这是 CLI 汇总框里的项目符号在 GBK 终端下无法编码，
> 与本次改动无关（设置 `PYTHONIOENCODING=utf-8` 即可消除）。

---

## 7. `tools/check_dsv4_data.py`：采集完整性校验

无需 NPU、无需跑完整个 `default`，直接对落位后的数据库做 SILICON 探针查询。

```bash
# 默认 ascend_910b / vllm-ascend / 0.23.0，按目标命令的 isl/osl 探测
python tools/check_dsv4_data.py --isl 4096 --osl 1024

# 指定其它版本 / 只查某一类
python tools/check_dsv4_data.py --version 0.23.0 --only attn
python tools/check_dsv4_data.py --only moe --moe-ep-list 64

# 落位到仓库之前，先在 staging 目录里验证刚采的表
python tools/check_dsv4_data.py --systems-path /workspace/staging_systems --isl 4096 --osl 1024
```

`--systems-path` 指向一个含 `<system>.yaml` + `data/` 的目录副本，用于**在把表拷进仓库之前**验证，
避免脏数据进主干。`--moe-ep-optional`（默认 `16 32`）里的 EP 缺失只报 warning、不算失败——
这两档采集很便宜（本地专家 24 / 12 个），但对应部署每卡需 90~182 GiB，64 GB 卡上必然 OOM，
采不到也不会影响最终结论。`--moe-ep-list` 默认 `16 32 64`，其中只有 **64** 是硬要求；
只关心可行性时可以直接 `--moe-ep-list 64`。

检查项：

1. **表是否加载**：5 张表（csa/hca × ctx/gen + mhc）是否存在且非空；
2. **CSA/HCA context**：`(fmha=float16, kv=float16, gemm=float16)` ×
   `num_heads ∈ {128,64,32,16,8,4,2}` × `b ∈ {1,2,4,8,16,32}` × `s = isl` 的探针；
3. **CSA/HCA generation**：同上 key × `b ∈ {1,8,64,128,256,512}` × `s = isl + osl`；
4. **mHC**：`op ∈ {pre,post}` × `hc_mult=4` × `hidden=7168`，报告 `num_tokens` 网格上界；
5. **MoE**：`(float16, power_law_1.01, topk=6, experts=384, hidden=7168, inter=3072, tp=1)`
   × `ep ∈ {16,32,64}`（默认 16/32 为 tolerant），报告实际可用的 `distribution` key
   与 `num_tokens` 上界；
6. 打印每张表的**网格边界**（min/max `s`、`b`、`num_tokens`），便于发现"有表但网格不够"。

退出码：`0` 全部命中；`1` 有硬缺失（缺表或缺 key）；`2` 参数/环境问题。
CI 或采集流程可以直接用它卡住回归。

---

## 8. 运行

### 8.0 规模可行性：64 卡够不够

按 `DeepSeekV4Model` 的实际权重（bf16）拆账：

| 组成 | 大小 |
|------|------|
| MoE 专家（384 × 3 × 7168 × 3072 × 2 B × 61 层） | **≈ 2882 GiB** |
| 其余（MLA / mHC / 共享专家 / router / 词表） | ≈ 11.6 GiB |
| 合计 | **≈ 2894 GiB（≈ 3.1 TB）** |

每卡常驻 ≈ `11.6/tp + 2882/ep`（GiB）。910B 单卡 64 GB：

| ep | tp=1 | tp=8 | tp=64 | 结论 |
|----|------|------|-------|------|
| **64** | 57 | **46** | 45 | ✅ 唯一放得下（剩余 7~19 GB 给 KV + 激活） |
| 32 | 102 | 92 | 90 | ❌ |
| 16 | 192 | 182 | 180 | ❌ |

推论：

1. **可行配置只有 `ep=64`**，即 `num_gpu_per_worker = tp × dp = 64` —— 64 卡刚好是
   **1 个 64 卡 worker（无副本）**，`tp ∈ {2..64}` → `num_heads ∈ {64,32,16,8,4,2}`。
2. 剩余显存仅 7~19 GB，KV cache 容量很小 → 并发数受限，
   `--ttft 1800 --tpot 100` 这类较严 SLA **很可能无解**。
3. 要看到有实际意义的帕累托前沿，建议按 **128 卡**（`ep=64` + 2 副本，或 `ep=128`）跑；
   上游 NVIDIA 侧也是在 ≥128 卡规模上评估 V4-Pro 的。
4. 无论 64 还是 128 卡，`ep=64` 的 MoE 行和 `num_heads ≤ 64` 的注意力行都是必须的。

### 8.1 支持性检查

```bash
aic-npu support --model deepseek-ai/DeepSeek-V4-Pro --system ascend_910b --backend vllm
```

> 实测输出：`Version: 0.23.0`，`Aggregated Support: YES`，`Disaggregated Support: YES`。
> 注意 `support` 子命令的 `--backend` 传的是支持矩阵里登记的 `vllm`；
> `default` 子命令在 `ascend_910b` 上**只能用 `--backend vllm-ascend`**。

### 8.2 配置搜索

硅数据齐了以后用 **SILICON**（默认即 SILICON，`--database-mode` 取值**大写**）：

```bash
aic-npu default \
  --model-path /workspace/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json \
  --total-gpus 64 --system ascend_910b \
  --backend vllm-ascend --backend-version 0.23.0 \
  --database-mode SILICON \
  --isl 4096 --osl 1024 --ttft 1800 --tpot 100 \
  --top-n 10 --save-dir results
```

硅数据补齐前临时用 `HYBRID` 跑通链路（缺表处回落 SOL）：

```bash
aic-npu default ... --database-mode HYBRID --total-gpus 64
```

> 默认 SLA（`--ttft 2000 --tpot 30`）在当前 SOL 估算下没有可行解，
> 想先看到帕累托表可以放宽，例如 `--ttft 100000 --tpot 1000`。
> `--ttft 1800 --tpot 100` 属于较严的 SLA，若 SILICON 下无解，先放宽确认链路，再收紧。

Python API 等价写法：

```python
from aiconfigurator_npu.sdk.task import TaskConfig
from aiconfigurator_npu.sdk.common import DatabaseMode, GEMMQuantMode, MoEQuantMode

task = TaskConfig(
    model="deepseek-ai/DeepSeek-V4-Pro",
    backend="vllm-ascend",
    system="ascend_910b",
    database_mode=DatabaseMode.SILICON,   # 硅数据补齐前用 HYBRID
    isl=4096,
    osl=1024,
    num_requests=100,
    gemm_quant_mode=GEMMQuantMode.float16,
    moe_quant_mode=MoEQuantMode.float16,
)
results = task.run()
results.pareto_analysis(ttft_sla_ms=1800, tpot_sla_ms=100)
results.print_pareto_table()
```

---

## 9. 已知限制

1. **MegaMoE 未建模**：上游的 `DeepSeekV4MegaMoEModule` 依赖 Blackwell（SM≥100）专有 kernel，
   Ascend 不适用，aic-npu 直接走常规 `MoE` + `MoEDispatch`。
2. **CP（context parallel）未建模**：上游仅 sglang 支持 V4 的 CP；vllm-ascend 不涉及。
   因此 `dsv4_paged_mqa_logits` / `dsv4_csa_topk_calib` 两张上游表在 aic-npu 里没有消费者（§2.1）。
3. **mHC sinkhorn 迭代按解析式建模**：`hc_sinkhorn_iters=20` 参与 FLOPs，实测可能偏差较大，
   优先用 `mhc_module_perf.txt` 覆盖。
4. **`expert_dtype: fp4` 不适用**：NPU 侧按 bf16 / W8A8 建模，忽略 FP4 专家。
5. **MoE 并行约束**：vllm-ascend 不支持同时按 TP 和 EP 切 MoE 权重，搜索空间 `moe_tp_list` 固定 `[1]`。
6. **采集脚本的硬件验证状态（2026-10-12）**：`collect_mhc.py` **尚未**在 NPU 上跑过；
   `collect_dsv4_attn.py` 的 `--module-source framework` 路径已在 NPU 上跑通真实 metadata 前向
   （踩坑见 §4.0），`reference` 路径一直可用。framework 路径强依赖 vllm-ascend 0.23.0 的内部布局，
   **升级 vllm-ascend 后需按 §4.0 的症状表回归**。
7. **DSv4 framework 采集是 TP=1 单模块近似**：生产环境跑 DSv4-Pro 使用 finegrained o_lora TP
   （`_forward_o_proj` 走 `olora_tp` 分支 + all_to_all），采集器 TP=1 走单卡 `else` 分支（分组 bmm）。
   两者 FLOPs 等价，但**不含 o_proj 的 TP 通信开销**；SDK 建模同样按两段 GEMM 计，口径一致。
8. **SWA 层被截断**：HF config 的 `compress_ratios` 有 62 项、末项为 `0`（SWA），
   按 `compress_ratios[:num_hidden_layers]` 截断后只剩 61 项 → Pro 实际建模为
   **30 层 CSA + 31 层 HCA**（与上游行为一致）。`get_kvcache_*()` 用的是 `len(compress_ratios)`（61），同样不含 SWA。
9. **MoE 的 `power_law_alpha` 分歧**：aic-npu 与上游一致用 `1.01`（DeepSeek 系），
   而 `MOEModel`（Qwen2/3-MoE、DeepSeek-V2-Lite 等）用 `1.2`。一张 `moe_perf.txt` 同时服务两者，
   因此转换脚本支持 `--moe-distribution both` 同时写两个 key（§4.4）。

---

## 10. 参考资料

- 上游模型实现：`aiconfigurator/aic-core/src/aiconfigurator_core/sdk/models/deepseek_v4.py`
- 上游算子：`aiconfigurator/aic-core/rust/aiconfigurator-core/src/operators/dsv4.rs`、`mhc.rs`
- 上游性能表：`aiconfigurator/aic-core/rust/aiconfigurator-core/src/perf_database/dsv4.rs`、`mhc.rs`
- 上游表清单注释：`aiconfigurator/aic-core/src/aiconfigurator_core/sdk/common.py:1206-1228`、
  `sdk/operations/dsv4.py:174-190`（说明哪些表是退役占位）
- 同类适配（V3.2 / GLM-5）：`docs/GLM5_ADAPTATION_DESIGN.md`
- 同类适配（R1）：`docs/DeepSeek-R1_ADAPTATION.md`
- 采集脚本：`collector/npu/collect_dsv4_attn.py`、`collect_mhc.py`、`collect_moe.py`、`dsv4_factory.py`
- 转换脚本：`tools/convert_to_aiconfigurator.py`
- 校验脚本：`tools/check_dsv4_data.py`
