# DeepSeek-V4-Pro 适配文档（aic-npu / Ascend 910B + vllm-ascend 0.23.0）

目标：让 `aic-npu` 能完成 `deepseek-ai/DeepSeek-V4-Pro` 在 `ascend_910b` + `vllm-ascend 0.23.0` 上的配置搜索
（即 `aic-npu support` 与 `aic-npu default` 两条命令全链路跑通）。

上游参照：`aiconfigurator`（NVIDIA 版）已支持 `deepseek-ai/DeepSeek-V4-Pro`，架构名 `DeepseekV4ForCausalLM`，
模型族 `DEEPSEEKV4`，对标 vLLM 0.24.0。本文档记录把这套建模"借"到 aic-npu 的**架构拆解与对齐**过程。

创建日期：2026-10-10
分支：`feature/deepseek-v4-pro-support`

---

## 0. 结论与工作量

DeepSeek-V4-Pro 与 DeepSeek-R1/V3/V3.2 **都不同**：

| 维度 | V3 / R1 | V3.2 / GLM-5 | **V4-Pro** |
|------|---------|--------------|------------|
| 架构名 | `DeepseekV3ForCausalLM` | `DeepseekV32ForCausalLM` / `GlmMoeDsaForCausalLM` | `DeepseekV4ForCausalLM` |
| 模型族 | `DEEPSEEK` | `DEEPSEEKV32` | **`DEEPSEEKV4`**（新增） |
| 注意力 | 内核级 MLA 表 | 模块级 DSA 表 | **模块级 DSv4 表（CSA/HCA 两类）+ mHC 模块** |
| KV  latent | `kv_lora_rank=512` + `qk_nope 128` + `v_head 128` | 同 V3 | **`head_dim=512` 单一 latent，`num_key_value_heads=1`（MQA）** |
| 输出投影 | `o_proj: [h*v_dim → h]` | 同 V3 | **分组低秩 `o_groups=16` / `o_lora_rank=1024` 两段投影** |
| 稀疏选择 | 无 | indexer topk（每层的 indexer） | **CSA（compress=4）带 indexer；HCA（compress=128）无 indexer；SWA（compress=0）并入 HCA** |
| 残差流 | 普通 residual | 普通 residual | **mHC（manifold-constrained hyper-connections），`hc_mult=4`，每层 pre/post 两处** |
| Dense 层 | `first_k_dense_replace=3` | — | **无 dense 层，61 层全 MoE** |
| MoE | 256 experts / topk 8 / inter 2048 | 同 V3 | **384 experts / topk 6 / inter 3072** |

因此**不能**像 R1 那样"复用 DEEPSEEK 族了事"，必须新增模型族与算子。

| 类别 | 任务数 | 已完成 | 需 NPU 硬件 |
|------|-------|-------|------------|
| A. 架构登记与配置解析 | 4 | 4 | 否 |
| B. 算子与建模（SDK） | 6 | 6 | 否 |
| C. `aic-npu default` / 支持矩阵适配 | 5 | 5 | 否 |
| D. 性能数据采集 | 2 | 0（脚本已就绪，待硬件采集） | **是** |
| E. 本地冒烟验证 | 3 | 3 | 否（无卡也能跑配置搜索） |

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

`compress_ratios` 语义（对齐上游 `DeepSeekV4Model`）：

| ratio | 含义 | 层数（Pro） | 是否有 indexer |
|-------|------|------------|----------------|
| `0` | SWA（纯滑窗） | 1 | 否 |
| `4` | CSA（Compressed Sparse Attention，4× 压缩 + indexer topk） | 30 | 是 |
| `128` | HCA（Hash/Highly-Compressed Attention，128× 压缩） | 30 | 否 |

建模时 **SWA(0) 折进 HCA(128)**（上游 `deepseek_v4.py:128`：
`ratio_counts[128] += ratio_counts.pop(0, 0)`），最终注意力算子聚合为 **CSA 一组（30 层）+ HCA 一组（31 层）**。

### 1.1 注意力算子拆解（每层）

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

> 注意 `head_dim + qk_rope_head_dim = 576`，与 V3 的 `kv_lora_rank + qk_rope = 576` 完全一致；
> V 维度 = `head_dim = 512`，与 V3 的 `kv_lora_rank = 512` 一致。
> 也就是说 **V4 的注意力核心就是 V3 的 MLA，只是 KV 被压缩、o_proj 换成两段分组低秩**。

### 1.2 mHC 算子拆解（每层 pre / post 各一处）

对齐上游 `operators/mhc.rs`：

```
hc_dim  = hc_mult × h
mix_hc  = (2 + hc_mult) × hc_mult
sites   = 2                      # attn mHC + FFN mHC
pre_ops  = sites × (2·T·hc_dim·mix_hc + T·hc_dim·3 + T·(hc²+2hc)·sinkhorn + 2·T·hc·h)
post_ops = sites × (2·T·hc·hc·h + 2·T·hc·h)
weights  = 2 × (mix_hc × hc_dim + mix_hc + 3) × bytes_per_element
```

mHC 与 router / logits 一样，量化**固定 bf16**（上游 `deepseek_v4.py` 多处 `GEMMQuantMode.bfloat16`）。

### 1.3 KV cache 拆解（非线性，分段）

对齐上游 `DeepSeekV4Model.get_kvcache_bytes_per_sequence()`。每层每序列：

```
sliding 部分：min(seq_len, sliding_window) × head_dim          # 所有层都有
压缩部分  ：ratio != 0 时  (seq_len // ratio) × head_dim
CSA 额外  ：2 × ratio × coff × head_dim 个 FP32（coff = 2 if ratio==4 else 1）  # compressor decode state
CSA       ：+ 压缩项 × indexer 条目（FP4 = index_head_dim × 0.5 字节/元素）
CSA       ：+ 2 × ratio × 2 × index_head_dim 个 FP32（第二个 indexer compressor decode state）
```

aic-npu 的内存模型是线性的（`kvcache = tokens × kvcache_per_token × bytes`），
因此拆成两项供 backend 使用：

- `get_kvcache_elements_per_token()` = `Σ_layers head_dim / ratio`（ratio=0 记 0）→ Pro = `30×128 + 30×4 = 3960`
- `get_kvcache_sliding_window_elements()` = `layers × sliding_window × head_dim` → Pro = `61 × 128 × 512`

---

## 2. 任务清单

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
| B1 | `PerfDataFilename` 新增 5 张表（dsv4 csa/hca × context/generation + mhc） | `sdk/common.py` | ✅ |
| B2 | 新增 `load_context_dsv4_module_data` / `load_generation_dsv4_module_data` / `load_mhc_module_data` | `sdk/perf_database.py` | ✅ |
| B3 | 新增 `query_context_dsv4_module` / `query_generation_dsv4_module` / `query_mhc_module`（含 SOL 解析估算） | `sdk/perf_database.py` | ✅ |
| B4 | 新增 `DeepSeekV4MHCModule` / `ContextDeepSeekV4AttentionModule` / `GenerationDeepSeekV4AttentionModule` | `sdk/operations.py` | ✅ |
| B5 | 新增 `DeepSeekV4Model`（含 compress_ratios 聚合、mHC pre/post、共享专家、MTP、KV cache 建模） | `sdk/models.py` | ✅ |
| B6 | `get_model()` 分派、`check_is_moe`、`_apply_model_quant_defaults` 适配 | `sdk/models.py` | ✅ |

### C 类：`aic-npu default` / 支持矩阵适配（无需硬件）

| # | 任务 | 位置 | 状态 |
|---|------|------|------|
| C1 | 支持矩阵把 `vllm` 版本提升到 `0.23.0`，并登记 V4-Pro 的 agg / disagg | `src/aiconfigurator_npu/systems/support_matrix.csv` | ✅ |
| C2 | MTP（nextn=1）白名单加 `DEEPSEEKV4` | `sdk/task.py:_base_common_layer` | ✅ |
| C3 | `validate()` 的注意力表 key 选择加 V4 分支 | `sdk/task.py:validate` | ✅ |
| C4 | backend 激活/KV cache 内存建模加 `DEEPSEEKV4` 分支（V4 的 `head_size=512` 不能直接参与激活估算） | `sdk/backends/trtllm_backend.py`、`sglang_backend.py` | ✅ |
| C5 | `supported_quant_mode` 注册 V4 表 | `sdk/perf_database.py` | ✅ |
| C6 | vllm-ascend 上把 HF FP8/FP4 推导出的 NVIDIA 量化模式重映射到 Ascend 可用的表项（否则 `default` 会因 `fp8_block` 无对应表直接报错） | `sdk/models.py:_apply_model_quant_defaults` | ✅ |

**C6 说明**：DeepSeek-R1 / V3 / V4-Pro 的 `quantization_config.quant_method = "fp8"` 会被
`_infer_quant_modes_from_raw_config()` 推导成 `gemm_quant_mode = fp8_block`，而 ascend 的
`gemm_perf.txt` 只有 `float16` / `sq` 两档，`TaskConfig.validate()` 会直接抛
`Unsupported gemm quant mode 'fp8_block'`。这是**在此之前就存在的**（R1 的 `default` 命令
同样跑不通），本次按 `tools/convert_to_aiconfigurator.py` 的既有约定做了重映射：

| 目标 | NVIDIA FP8/FP4 模式 | Ascend 表项 |
|------|--------------------|-------------|
| GEMM | `fp8` / `fp8_static` / `fp8_block` / `fp8_ootb` / `nvfp4` | `sq`（Ascend W8A8 动态量化就落在这一档；`w8a8_dynamic` 也在 `_normalize_gemm_quant_mode_for_table()` 里归一化到 `sq`） |
| MoE | `fp8` / `fp8_block` / `nvfp4` / `w4afp8` / `w4a16_mxfp4` / `w4a8_mxfp4_mxfp8` | `float16`（Ascend W8A8 MoE 采集未完成，表里只有 float16 行） |
| FMHA | `fp8` | `float16` |

### D 类：性能数据采集（**需 NPU 硬件**）

| # | 数据 | 状态 | 说明 |
|---|------|------|------|
| D1 | `gemm_perf.txt` / `moe_perf.txt` / `custom_allreduce_perf.txt` | ✅ 已有（0.23.0 目录） | 维度：384 experts / topk 6 / 7168 / 3072 需补采 |
| D2 | `mhc_module_perf.txt` | ⬜ 待采集 | 脚本 `collector/npu/collect_mhc.py` 已就绪 |
| D3 | `dsv4_{csa,hca}_{context,generation}_module_perf.txt` | ⬜ 待采集 | 脚本 `collector/npu/collect_dsv4_attn.py` 已就绪 |

缺失时的影响（与 DSA 一致）：

| 模式 | 行为 |
|------|------|
| `SILICON` | 缺表 → `PerfDataNotAvailableError`，跑不了 |
| `HYBRID` | 回落 SOL 解析估算 × 1/0.5 |
| `EMPIRICAL` / `SOL` | 直接用解析估算 |

**没有硅数据之前，用 `--database-mode HYBRID` 即可跑通全链路**（见第 4 节）。

#### D2：`mhc_module_perf.txt`

`op_name` 只能是 `pre` / `post`（`query_mhc_module()` 里写死），`num_tokens` 是唯一被插值的轴，
`hc_mult` / `hidden_size` 是精确 key。

```bash
python collector/npu/collect_mhc.py \
  --op-types pre post \
  --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 \
  --hc-mult 4 \
  --hidden-size 7168 \
  --sinkhorn-iters 20 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_mhc \
  --resume
```

2 op × 14 tokens = 28 个点，纯 GEMM + elementwise，几分钟可完成。先冒烟：

```bash
python collector/npu/collect_mhc.py \
  --num-tokens-list 1 128 --op-types pre \
  --warmup-iters 1 --bench-iters 3 \
  --output-dir ./data/dsv4_mhc_smoke
```

#### D3：`dsv4_{csa,hca}_{context,generation}_module_perf.txt`

四个组合（csa/hca × context/generation）**必须各自一个 `--output-dir`**：
checkpoint 文件名共享，同一目录跑两批会互相覆盖已完成记录。

`--num-heads-list` 要覆盖 `128 // tp_size`，即 TP=1/2/4/8 → `128/64/32/16`，且是**精确 key**（不插值）。
context 表的 `s = isl`，generation 表的 `s = isl + step`（脚本写 step=0）。

```bash
# 1) CSA context
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds csa \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 \
  --num-heads-list 128 64 32 16 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_csa_ctx --resume

# 2) CSA generation
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_generation --attn-kinds csa \
  --batch-list 1 2 4 8 16 32 64 128 256 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/dsv4_csa_gen --resume

# 3) / 4) HCA context + generation：把 --attn-kinds 换成 hca，换输出目录
```

`--module-source`：

| 取值 | 说明 |
|------|------|
| `framework`（默认） | 实例化 vllm-ascend 自己的 DSv4 attention 类。类名靠候选模块探测（`dsv4_factory._discover_attention_cls`），可用 `--attention-cls module.path:ClassName` 显式指定 |
| `reference` | 内置 torch 参考实现（与 SDK 的 SOL 拆解完全同构），不依赖 vllm-ascend 内部布局，可先跑通拿量级 |

先冒烟再全量（每个组合 2 个点）：

```bash
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds csa \
  --batch-list 1 --seq-len-list 128 --num-heads-list 128 \
  --module-source reference \
  --warmup-iters 1 --bench-iters 3 \
  --output-dir ./data/dsv4_smoke
```

#### 落位

```bash
cd src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend

cp ./data/dsv4_mhc/mhc_module_perf.txt                                   0.23.0/
cp ./data/dsv4_csa_ctx/dsv4_csa_context_module_perf.txt                   0.23.0/
cp ./data/dsv4_hca_ctx/dsv4_hca_context_module_perf.txt                   0.23.0/
cp ./data/dsv4_csa_gen/dsv4_csa_generation_module_perf.txt                0.23.0/
cp ./data/dsv4_hca_gen/dsv4_hca_generation_module_perf.txt                0.23.0/
```

或用转换工具（会顺带改写 framework/version/device 三列）：

```bash
python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/dsv4_all \
  --output-dir src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0 \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --mla-output none --dsv4-output --mhc-output
```

---

## 3. 建模路径

```
config.json architectures[0] = "DeepseekV4ForCausalLM"
    ↓ common.ARCHITECTURE_TO_MODEL_FAMILY              → "DEEPSEEKV4"
    ↓ utils._parse_hf_config_json()                    → extra_params = DeepSeekV4Config(...)
    ↓ sdk/models.py get_model()                        → DeepSeekV4Model
    ↓ context_ops / generation_ops
    │     ├─ DeepSeekV4MHCModule          → mhc_module_perf.txt
    │     ├─ Context/GenerationDeepSeekV4AttentionModule (CSA + HCA)
    │     │                               → dsv4_{csa,hca}_{context,generation}_module_perf.txt
    │     ├─ GEMM（shared gate/up/down、router、logits）→ gemm_perf.txt
    │     ├─ MoE / MoEDispatch            → moe_perf.txt
    │     └─ Embedding / ElementWise / P2P
    ↓ PerfDatabase.query_*()
```

---

## 4. 运行

### 4.1 支持性检查

```bash
aic-npu support --model deepseek-ai/DeepSeek-V4-Pro --system ascend_910b --backend vllm
```

> 实测输出：`Version: 0.23.0`，`Aggregated Support: YES`，`Disaggregated Support: YES`。
> 注意 `support` 子命令的 `--backend` 传的是支持矩阵里登记的 `vllm`；
> `default` 子命令在 `ascend_910b` 上**只能用 `--backend vllm-ascend`**（`main.py` 会校验系统/后端组合）。

### 4.2 配置搜索（**当前推荐 HYBRID**）

DSv4 / mHC 硅数据尚未采集，SILICON 模式会在注意力查询处抛 `PerfDataNotAvailableError`，
所以先用 HYBRID 让新算子走 SOL 解析估算。注意 `--database-mode` 的取值是**大写**
（`SILICON | HYBRID | EMPIRICAL | SOL`）：

```bash
aic-npu default \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --system ascend_910b \
  --backend vllm-ascend \
  --database-mode HYBRID \
  --total-gpus 128
```

> `--total-gpus` 参考：V4-Pro 是 384 experts / inter 3072 / 61 层，bf16 权重约 3 TB，
> Ascend 910B 单卡 64 GB，8 卡放不下；上游 NVIDIA 侧也是在 128 卡规模上跑的。
> 默认 SLA（`--ttft 2000 --tpot 30`）在当前 SOL 估算下没有可行解，
> 想先看到帕累托表可以放宽，例如 `--ttft 100000 --tpot 1000`。

Python API 等价写法：

```python
from aiconfigurator_npu.sdk.task import TaskConfig
from aiconfigurator_npu.sdk.common import DatabaseMode, GEMMQuantMode, MoEQuantMode

task = TaskConfig(
    model="deepseek-ai/DeepSeek-V4-Pro",
    backend="vllm-ascend",
    system="ascend_910b",
    database_mode=DatabaseMode.HYBRID,   # 硅数据补齐后改 SILICON
    isl=4096,
    osl=512,
    num_requests=100,
    gemm_quant_mode=GEMMQuantMode.float16,
    moe_quant_mode=MoEQuantMode.float16,
)
results = task.run()
results.pareto_analysis(ttft_sla_ms=3000, tpot_sla_ms=50)
results.print_pareto_table()
```

### 4.3 硅数据补齐后切 SILICON

数据文件就位后去掉 `--database-mode hybrid` 即可（默认 SILICON）。

---

## 4.4 验证记录（本次开发，无卡环境）

| # | 命令 | 结果 |
|---|------|------|
| E1 | `aic-npu support --model deepseek-ai/DeepSeek-V4-Pro --system ascend_910b --backend vllm` | ✅ `Version 0.23.0`，agg / disagg 均 `YES` |
| E2 | `aic-npu default --model deepseek-ai/DeepSeek-V4-Pro --system ascend_910b --backend vllm-ascend --database-mode HYBRID --total-gpus 128 --ttft 100000 --tpot 1000` | ✅ 跑完 agg（518 条）+ disagg（121 条），输出帕累托表 |
| E3 | `python tools/smoke_dsv4_model.py`（tp=1/2/4/8 构造模型） | ✅ 每相 2 个 `*_attention` op（CSA=30 层 / HCA=31 层）、mHC pre/post、共享专家/MoE 齐全；bf16 权重 tp1=2890 GiB → tp8=363 GiB |

> Windows 控制台（`gbk` 编码）下打印最终汇总表时会报
> `UnicodeEncodeError: ... '\u2022'`，这是 CLI 汇总框里的项目符号在 GBK 终端下无法编码，
> 与本次改动无关（设置 `PYTHONIOENCODING=utf-8` 即可消除）。

回归（支持矩阵版本从 0.18.0 升到 0.23.0 后）：R1 / V3 / GLM-5 / Qwen3-235B 的
`support` 均仍为 `YES`；R1 的 `default` 命令在 C6 修复后首次跑通。

---

## 5. 已知限制

1. **MoE 维度需要补采**：V4 是 384 experts / topk 6 / inter 3072，而现有 `moe_perf.txt` 是按
   256 experts / topk 8 / inter 2048 采的。HYBRID 下会走 SOL 估算，SILICON 下会插值失败。
2. **MegaMoE 未建模**：上游的 `DeepSeekV4MegaMoEModule` 依赖 Blackwell（SM≥100）专有 kernel，
   Ascend 不适用，aic-npu 直接走常规 `MoE` + `MoEDispatch`。
3. **CP（context parallel）未建模**：上游仅 sglang 支持 V4 的 CP；vllm-ascend 不涉及。
4. **mHC sinkhorn 迭代按解析式建模**：`hc_sinkhorn_iters=20` 参与 FLOPs，实测可能偏差较大，
   优先用 `mhc_module_perf.txt` 覆盖。
5. **`expert_dtype: fp4` 不适用**：NPU 侧按 bf16 / W8A8 建模，忽略 FP4 专家。
6. **MoE 并行约束**：vllm-ascend 不支持同时按 TP 和 EP 切 MoE 权重，搜索空间 `moe_tp_list` 固定 `[1]`。
7. **采集脚本尚未在硬件上跑过**：`collector/npu/collect_mhc.py` 与
   `collector/npu/collect_dsv4_attn.py`（+ `dsv4_factory.py`）是照现有 MLA/DSA 采集器的
   模式写的，输出 schema 与 `perf_database` 的加载器严格对齐，但**没有在 NPU 上验证过**。
   `collect_dsv4_attn.py --module-source framework` 依赖 vllm-ascend 0.23.0 内部的
   DSv4 attention 类名，首次在硬件上跑建议先冒烟，必要时用
   `--attention-cls module.path:ClassName` 或 `--module-source reference` 兜底。
8. **SWA 层被截断**：HF config 的 `compress_ratios` 有 62 项、末项为 `0`（SWA），
   按 `compress_ratios[:num_hidden_layers]` 截断后只剩 61 项 → Pro 实际建模为
   **30 层 CSA + 31 层 HCA**（与上游行为一致，上游同样 count(0)==0）。
   `get_kvcache_*()` 用的是 `len(compress_ratios)`（61），同样不含 SWA。

---

## 6. 参考资料

- 上游模型实现：`aiconfigurator/aic-core/src/aiconfigurator_core/sdk/models/deepseek_v4.py`
- 上游算子：`aiconfigurator/aic-core/rust/aiconfigurator-core/src/operators/dsv4.rs`、`mhc.rs`
- 上游性能表：`aiconfigurator/aic-core/rust/aiconfigurator-core/src/perf_database/dsv4.rs`、`mhc.rs`
- 同类适配（V3.2 / GLM-5）：`docs/GLM5_ADAPTATION_DESIGN.md`
- 同类适配（R1）：`docs/DeepSeek-R1_ADAPTATION.md`
