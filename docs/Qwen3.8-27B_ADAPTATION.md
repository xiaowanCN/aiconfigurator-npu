# Qwen3.8-27B 适配与数据采集文档（aic-npu / Ascend 910B + vllm-ascend 0.23.0）

目标：让 `aic-npu` 能完成 `Qwen3.8-27B`（`model_configs/qwen3_8_27B_config.json`）在 `ascend_910b` +
`vllm-ascend 0.23.0` 上的配置搜索，并且**结果全部落在硅数据上**。

上游参照：`aiconfigurator`（NVIDIA 版）已支持 `Qwen3.5` 系列（架构名 `Qwen3_5ForConditionalGeneration`，
模型族 `QWEN35`，对标 vLLM 0.24.0）。**Qwen3.8-27B 与 Qwen3.5 共用同一套架构**（`qwen3_5`），
因此 aic-npu 侧的**架构登记、算子建模、支持矩阵都已就绪**，本次适配只差**数据**：

| 类别 | 状态 |
|---|---|
| A/B/C/E（架构登记、算子建模、支持矩阵、本地冒烟） | ✅ 已完成，无需硬件 |
| **D1 `gdn_perf.txt`** | ❌ **缺一张表 —— 当前最大阻塞项（需新采集器）** |
| **D2 `context_attention_perf.txt` / `generation_attention_perf.txt`** | ⚠️ 已存在但**只有 `head_dim=128` 的行**；模型需要 `head_dim=256` 且 `num_heads∈{24,12,6,3}` 的 GQA 行 —— **需补采** |
| `Qwen35Model` 的 GDN 查询键 | ⚠️ 当前传**全局头数**，`tp>1` 时查不到本地头数的行 —— **需修** |

> 一句话：`aic-npu default ... --database-mode HYBRID` 现在能跑通（GDN 落 SOL、attention 用 128 行外推），
> 但 `SILICON` 会因 `gdn_perf.txt` 缺失直接抛 `PerfDataNotAvailableError`，且 attention 的
> `head_dim=256` 行缺失会让 16 层 GQA 走 SOL。补完 D1/D2 并修掉 GDN 查询键后，`SILICON` 才能全链路命中。

---

## 0. TL;DR：当前状态与阻塞项

| 类别 | 状态 |
|---|---|
| 架构建模（QWEN35 族、`Qwen35Model`、GDN `GDNKernel`、支持矩阵、MTP 白名单） | ✅ 全部完成 |
| `gdn_perf.txt`（GDN 4 个 kernel 的性能） | ❌ **未采集 —— 新增 `collector/npu/collect_gdn.py`** |
| attention `head_dim=256` 的 GQA 行 | ⚠️ **未采集 —— 用现有 `collect_attn.py` 补采** |
| `Qwen35Model` GDN 查询传全局头数 | ⚠️ **已修复为 per-tp 本地头数**（`models.py`） |
| 落位到 `0.23.0/` 数据目录 | ❌ `gdn_perf.txt` 还没有；attention 需追加 `head_dim=256` 行 |

**两个必须踩的点（本次确认）**：

1. **GDN 查询键是 per-tp 本地头数，不是全局头数**。NVIDIA 侧 `qwen35.py` 对 GDN 传
   `gdn_nk_per_tp = nk // tp`、`gdn_nv_per_tp = nv // tp`（见 §4.0 的 kernel 名与 §5 F1）。
   aic-npu 原 `Qwen35Model` 误传全局 `nk=16 / nv=48`，导致 `tp>1` 时查询键 `(16,48)` 与采集器写出的
   `(16//tp, 48//tp)` 对不上 → 回落 SOL。**已修**（§5）。
2. **attention 需要 `head_dim=256`**。模型 `head_dim=256`，而现有 0.23.0 attention 表只有 `head_dim=128`
   的行。GQA 全注意力层（`num_attention_heads=24 / num_key_value_heads=4`）在 `tp∈{1,2,4,8}` 下需要
   收集 rank-local `(num_heads, num_kv_heads) ∈ {(24,4),(12,2),(6,1),(3,1)}`、`head_dim=256` 的
   context + generation 行（§3.2 / §4.2）。

---

## 1. 架构拆解：Qwen3.8-27B 关键参数

来源：`model_configs/qwen3_8_27B_config.json`（已同步到 aic-npu）。

| 参数 | 值 |
|------|-----|
| `architectures` | `Qwen3_5ForConditionalGeneration` |
| 模型族 | `QWEN35`（与 Qwen3.5 共用） |
| `hidden_size` | 5120 |
| `num_hidden_layers` | 64 |
| **layer_types** | 每 4 层一个 `full_attention` + 3 个 `linear_attention` 循环 → **48 层 GDN + 16 层 GQA** |
| `num_attention_heads` / `num_key_value_heads` | 24 / **4**（GQA，仅 full_attention 层用） |
| `head_dim` | **256**（GQA 层） |
| `intermediate_size` | 17408（dense SwiGLU FFN，非 MoE） |
| **GDN 几何** | `linear_num_key_heads=16`、`linear_key_head_dim=128`、`linear_num_value_heads=48`、`linear_value_head_dim=128`、`linear_conv_kernel_dim=4` |
| `mamba_ssm_dtype` | `float32`（GDN recurrence 状态精度，对应 `query_gdn` 的 SOL 注释） |
| `mtp_num_hidden_layers` | 1（MTP / nextn=1） |
| `vocab_size` | 248320 |
| `max_position_embeddings` | 262144 |
| MoE | **无**（`num_experts` 不存在 → 纯 dense 27B） |

> 与 Qwen3.5 的区别：Qwen3.8-27B 是 **dense**（无 MoE），GDN 头数 `(16,48)`、GQA `(24,4)`、
> `head_dim=256`。Qwen3.5 的其它变体（35B-A3B / 397B）是 MoE，但**建模与采集逻辑完全通用**，
> 本适配只针对 27B 的几何补齐数据。

### 1.1 两种注意力算子（每层）

| 层类型 | 子算子 | 说明 |
|--------|--------|------|
| `linear_attention`（48 层） | `GDNKernel` × 4（context conv / scan + generation conv / recurrence）+ in/out proj GEMM + ElementWise norm + CustomAllReduce | 走 **`gdn_perf.txt`** |
| `full_attention`（16 层） | `ContextAttention` / `GenerationAttention`（GQA，`head_dim=256`）+ QKV/proj GEMM | 走 **`context_attention_perf.txt` / `generation_attention_perf.txt`** |

> **这条直接决定采集清单**：48 层 GDN 读 `gdn_perf.txt`，16 层 GQA 读 attention 表。
> 只采 attention 不采 GDN = 48/64 层走 SOL；GDN 表只写全局头数 = `tp>1` 全部回落 SOL。

### 1.2 GDN 4 个 kernel 的几何（per-tp 本地头数）

`query_gdn` 的模型键为 `(d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv)`，
其中 `d_model=5120` 为**全局**，头数为 **rank-local**。27B 在 `tp∈{1,2,4,8}` 下的本地几何：

| tp | `num_k_heads` (nk) | `num_v_heads` (nv) | conv_dim = 2·nk·hk + nv·hv |
|----|--------------------|--------------------|------------------------------|
| 1  | 16 | 48 | 10240 |
| 2  | 8  | 24 | 5120 |
| 4  | 4  | 12 | 2560 |
| 8  | 2  | 6  | 1280 |

> `head_k_dim=head_v_dim=128`。`conv_dim` 是 packed-QKV 卷积的输入通道数（Q+K+V，
> 生产 `gdn.py` 对 `mixed_qkv` 做深度卷积）。

---

## 2. 目标命令会读哪些表（采集清单的唯一依据）

`aic-npu default ... --isl 4096 --osl 1024` 的查询路径：

```
config.json architectures[0] = "Qwen3_5ForConditionalGeneration"
    ↓ common.ARCHITECTURE_TO_MODEL_FAMILY              → "QWEN35"
    ↓ utils._parse_hf_config_json()                    → extra_params = Qwen35Config(...)
    ↓ sdk/models.py get_model()                        → Qwen35Model
    ↓ context_ops / generation_ops
    │     ├─ GDNKernel ×4（48 层）                       → gdn_perf.txt
    │     ├─ ContextAttention / GenerationAttention     → context/generation_attention_perf.txt
    │     ├─ GEMM（in/out proj、logits）                 → gemm_perf.txt（已有）
    │     ├─ ElementWise / Embedding / CustomAllReduce / P2P
    ↓ PerfDatabase.query_*()
```

| # | 表文件 | 采集脚本 | 状态 |
|---|--------|----------|------|
| D1 | `gdn_perf.txt` | **`collector/npu/collect_gdn.py`（新增）** | ❌ 待采 |
| D2 | `context_attention_perf.txt` / `generation_attention_perf.txt` | `collector/npu/collect_attn.py`（现有） | ⚠️ 需补 `head_dim=256` 行 |
| D3 | `gemm_perf.txt` | `collector/npu/collect_gemm.py`（现有） | ✅ 通用表，覆盖 5120 / 17408 |

`check_model_readiness.py`（QWEN35 分支）会按上述清单逐项校验，是落位前的正式闸门（§6）。

---

## 3. 采集矩阵：每张表必须覆盖的 key

### 3.1 GDN（`gdn_perf.txt`，`data[kernel_source][phase][model_key]`，model_key 见 §1.2）

| 轴 | 目标命令取值 | 要求网格 |
|---|-------------|---------|
| `kernel_source` | **4 个全部要**：`causal_conv1d_fn` / `chunk_gated_delta_rule`（context）、`causal_conv1d_update` / `fused_sigmoid_gating_delta_rule_update`（generation） | 4 个逻辑名，对应 vllm-ascend 0.23.0 真实 kernel（见 §4.1） |
| `phase` | context / generation | 都要 |
| `d_model` | **5120（全局，精确 key）** | 只 5120 |
| `num_k_heads` / `num_v_heads` | **per-tp 本地头数**（§1.2 表） | tp ∈ {1,2,4,8} → `(16,48)/(8,24)/(4,12)/(2,6)` |
| `head_k_dim` / `head_v_dim` | 128（精确 key） | 只 128 |
| `d_conv` | 4（精确 key） | 只 4 |
| context `batch_size` × `seq_len` | `isl=4096` → context 网格必须含 4096 | `b∈{1,2,4,8,16,32,64}` × `s∈{1..32768}`（见 §4.1 容量上限） |
| generation `batch_size` | 受 OOM 限制，实测较小 | `b∈{1,2,4,8,16,32,64,128,256,512,1024}` |

> ⚠️ **`query_gdn` 的 fallback 很弱**：`model_key` 找不到时会按「同 `d_model` 下最近 `num_v_heads`」回填，
> 但 `d_model` 不匹配（5120 之外）直接回落 SOL；`_interp_2d_linear` 对 context、`_nearest_1d` 对 generation。
> 因此**每个 (kernel, phase, tp) 的 `(b, s)` 网格要足够密**，尤其 context 要在目标 `isl` 附近有 ≥2 个 `s` 点。

### 3.2 attention（`context/generation_attention_perf.txt`）

| 轴 | 目标命令取值 | 要求网格 |
|---|-------------|---------|
| `head_dim` | **256**（模型 `head_dim=256`） | **必须含 256**（现有只有 128） |
| `num_heads`（rank-local q 头） | `24 // tp` → `{24,12,6,3}` | 4 个值都要 |
| `num_key_value_heads`（rank-local kv 头） | `ceil(4 / tp)` → `{4,2,1,1}` | 对应 `{4,2,1}`（tp=4 与 tp=8 都是 1） |
| `attn_dtype` / `kv_cache_dtype` | `float16`（bf16 测量，映射 float16） | 只 float16 |
| context `isl`（= `s`） | **4096** | 必须含 4096；建议 `128…16384` 全序列 |
| context `batch_size` | 实测较小（≤32） | `1 2 4 8 16 32` 足够 |
| generation `step`（= `s-1`，`isl=1`） | `4096~5120` 上界 | 必须含 4096 与 ≥5120 上界 |

> ⚠️ **GQA 约束**：`num_heads % num_kv_heads == 0`。本模型 4 组 `(24,4)/(12,2)/(6,1)/(3,1)` 都满足。
> `num_kv_heads=0` 在 aiconfigurator 约定里表示 MHA（kv==heads），本模型不是，按实际 kv 头数写。

---

## 4. 采集命令（可直接复制执行）

### 4.0 公共前提（NPU 环境，vllm-ascend 0.23.0）

GDN 采集走 vllm-ascend 的**真实 kernel**（`npu_causal_conv1d_custom`、`chunk_gated_delta_rule`、
`npu_recurrent_gated_delta_rule`），这些 AscendC 自定义算子来自 vllm-ascend 自带的 custom-op 包，
**必须在加载路径上**：

```bash
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
```

`collector/npu/collect_gdn.py` 在 `_init_runtime()` 内也会调 `bootstrap_custom_op_env(include_vendor_lib=True)`
+ `enable_custom_op()` 做进程内兜底。**缺失症状**：报 `npu_causal_conv1d_custom ... not in libopapi.so` 或
`torch.ops._C_ascend` 命名空间不存在。

> **kernel 名 ↔ 真实入口映射**（写入 CSV 的 `kernel_source` 必须是 SDK 查询用的逻辑名）：
>
> | `kernel_source`（CSV） | vllm-ascend 0.23.0 真实入口 |
> |------------------------|------------------------------|
> | `causal_conv1d_fn`（context conv） | `torch.ops._C_ascend.npu_causal_conv1d_custom`（`run_mode=0`） |
> | `chunk_gated_delta_rule`（context scan） | `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule` |
> | `causal_conv1d_update`（generation conv） | `torch.ops._C_ascend.npu_causal_conv1d_custom`（`run_mode=1`） |
> | `fused_sigmoid_gating_delta_rule_update`（generation recur） | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` |
>
> 这是 910B 上 `vllm_ascend/ops/gdn.py` 的 `_forward_core` 生产路径（conv 走 `npu_causal_conv1d_custom`、
> scan 走 `chunk_gated_delta_rule`、recurrence 走 `npu_recurrent_gated_delta_rule` 的 csrc 自定义算子）。
> g/beta 门控（`fused_gdn_gating`）不是 SDK 的 GDN kernel，时间上归到 ElementWise norm，
> 故采集器把 g/beta 作为随机张量放在计时区之外（与 NVIDIA `collect_gdn.py` 一致）。

### 4.1 D1：GDN（新增 `collect_gdn.py`）

采集器**直接写出** `gdn_perf.txt`（列定义与 `perf_database.load_gdn_data` 完全一致，无需转换）。

```bash
cd /workspace/collector/npu && \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python collect_gdn.py \
  --model-name qwen3_8_27B \
  --d-model 5120 --d-conv 4 \
  --num-k-heads 16 --head-k-dim 128 --num-v-heads 48 --head-v-dim 128 \
  --tp-sizes 1 2 4 8 \
  --max-context-tokens 1000000 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/gdn_qwen38 --resume
```

要点：

- **tp 各跑一遍**：`--tp-sizes 1 2 4 8`，采集器内部按 `nk//tp`、`nv//tp` 计算本地头数并写出对应行。
- **context 容量上限**：`--max-context-tokens 1000000` 会跳过 `b×s > 1M` 的点（如 `b=64×s=32768`），
  避免单卡 HBM OOM；这些尾部点用插值即可。采集器对**每个 spec 独立 try/except**，
  OOM/报错记 FAILED 后继续（`--resume` 可续采）。
- **网格完整性**：context 网格为 `b∈{1,2,4,8,16,32,64} × s∈{1..32768}`，generation `b∈{1..1024}`。
  每个 (kernel, phase, tp) 的 `(b,s)` 至少要够密，保证目标 `isl=4096` 落在凸包内。
- **NPU graph 是默认口径**（与 `benchmark_npu` 一致）；失败时回落 eager。
- **冒烟建议**（换环境首次跑，2 个点）：
  ```bash
  cd /workspace/collector/npu && python collect_gdn.py \
    --tp-sizes 1 --num-k-heads 16 --num-v-heads 48 \
    --warmup-iters 1 --bench-iters 3 --output-dir ./data/gdn_smoke
  ```
- **自检**：看日志里延迟是否随 `seq_len` / `batch_size` 单调变化；若不同 `(nk,nv)` 的值几乎相同，
  说明测到的是主机开销而非 kernel 时间（应不会发生——采集器的计时闭包只含单卡前向）。

落位（直接 `cp`，因为该表只服务 QWEN35、且是新建）：

```bash
DATA_DIR=/workspace/src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
cp ./data/gdn_qwen38/gdn_perf.txt "$DATA_DIR/"
```

### 4.2 D2：attention `head_dim=256`（现有 `collect_attn.py`，补采）

现有 attention 表只有 `head_dim=128` 的行。用现有采集器补 `head_dim=256` 的 GQA 行：

```bash
cd /workspace/collector/npu && \
python collect_attn.py \
  --op-types context attention \
  --head-size 256 \
  --num-heads-list 24 12 6 3 \
  --num-kv-heads-list 4 2 1 \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/attn256_ctx --resume

cd /workspace/collector/npu && \
python collect_attn.py \
  --op-types generation attention \
  --head-size 256 \
  --num-heads-list 24 12 6 3 \
  --num-kv-heads-list 4 2 1 \
  --batch-list 1 2 4 8 16 32 64 128 256 512 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir ./data/attn256_gen --resume
```

> ⚠️ **910B FlashInfer attention 是否支持 `head_dim=256` 需在采集机上验证**：若 kernel 拒绝
> `head_dim=256`（某些 FIA 实现上限 128/192），对应行会 FAILED，16 层 GQA 在 `SILICON` 下回落 SOL。
> 验证方法：冒烟 `head-size 256 --num-heads-list 24 --num-kv-heads-list 4 --seq-len-list 4096` 看是否成功。

**落位（attention 表是共享表，绝不能整体 `cp` 覆盖现有 128 行）**：

`convert_to_aiconfigurator.py` 的 `convert_context_attention` / `convert_generation_attention` 默认**覆盖写**
目标文件。因此要先转到**隔离目录**得到只含 256 行的临时表，再**追加数据行**（去表头）到共享表：

```bash
# 1) 转到隔离目录（产出只含 head_dim=256 行的临时 attention 表）
#    注意：collect_attn.py 在 collector/npu/ 下运行，输出在 collector/npu/data/...
cd /workspace && \
python tools/convert_to_aiconfigurator.py \
  --input-dir collector/npu/data/attn256_ctx --output-dir collector/npu/data/attn256_ctx_out \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0
python tools/convert_to_aiconfigurator.py \
  --input-dir collector/npu/data/attn256_gen --output-dir collector/npu/data/attn256_gen_out \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0

# 2) 追加数据行（去掉表头）到共享表
DATA_DIR=src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
SRC_CTX=collector/npu/data/attn256_ctx_out/context_attention_perf.txt
SRC_GEN=collector/npu/data/attn256_gen_out/generation_attention_perf.txt
DST_CTX="$DATA_DIR/context_attention_perf.txt"
DST_GEN="$DATA_DIR/generation_attention_perf.txt"

# 保险：确保目标文件以换行结尾，否则追加会把最后一行和第一行粘在一起
for f in "$DST_CTX" "$DST_GEN"; do
  if [ -s "$f" ] && [ -n "$(tail -c1 "$f")" ]; then printf '\n' >> "$f"; fi
done
tail -n +2 "$SRC_CTX" >> "$DST_CTX"
tail -n +2 "$SRC_GEN" >> "$DST_GEN"
```

> ⚠️ **`--append` 不幂等**：同一批 CSV 跑两次会得到重复行（`load_attention_data` 只保留先到的，
> 不报错但数据脏）。重跑前请先 `git checkout` 或备份 attention 表。

---

## 5. 已发现的缺陷与修复

| # | 缺陷 | 影响 | 修复 | 状态 |
|---|------|------|------|------|
| **F1** | `Qwen35Model` 的 `GDNKernel` 调用传**全局头数**（`nk=16 / nv=48`），而采集器按 per-tp 本地头数写入、NVIDIA 侧也按本地头数查询 | `tp>1` 时 GDN 查询键 `(16,48)` ≠ 采集键 `(16//tp,48//tp)` → 48 层全部回落 SOL | `models.py` 的 `Qwen35Model._build_context_ops` / `_build_generation_ops` 改为 `gdn_nk_per_tp = nk // tp`、`gdn_nv_per_tp = nv // tp`，并加整除校验 | ✅ 已修复 |
| **F2** | attention 表只有 `head_dim=128`，模型需要 `head_dim=256` 且 `num_heads∈{24,12,6,3}` | 16 层 GQA 在 `SILICON` 下落 SOL | §4.2 补采 `head_dim=256` 的 GQA 行并追加 | ⬜ 待执行（需 NPU） |
| **F3** | `gdn_perf.txt` 尚未采集（QWEN35 族唯一缺失表） | `SILICON` 直接抛 `PerfDataNotAvailableError` | 新增 `collector/npu/collect_gdn.py`（§4.1） | ⬜ 待执行（需 NPU） |
| **F4** | `check_model_readiness.py` 在 0.23.0 下对 QWEN35 报 `dense attention collected head_dim=[128] but the model needs 256` + 缺 `gdn_perf.txt` | 落位前无法自动发现缺口 | 已具备；落位后跑它校验（§6） | ✅ 已具备 |

---

## 6. 落位后校验（必做）

```bash
python tools/check_model_readiness.py \
  model_configs/qwen3_8_27B_config.json \
  --system ascend_910b --backend vllm-ascend --version 0.23.0
```

该脚本按 §3 的 key 矩阵检查 0.23.0 目录下 QWEN35 家族需要的 4 张表（gemm / context_attention /
generation_attention / gdn），并逐项报告 `head_dim` 网格、`num_heads` 网格、GDN 表是否存在。
落位前会 BLOCKING 报 `gdn_perf.txt` 缺失 + attention `head_dim=256` 缺；落位后应全部 PASS。

---

## 7. 运行

### 7.1 支持性检查

```bash
aic-npu support --model qwen3_8_27B --system ascend_910b --backend vllm
```

> `support` 子命令的 `--backend` 传支持矩阵里登记的 `vllm`；`default` 子命令在 `ascend_910b` 上
> **只能用 `--backend vllm-ascend`**。

### 7.2 配置搜索

硅数据齐了以后用 **SILICON**（默认即 SILICON，`--database-mode` 取值**大写**）：

```bash
aic-npu default \
  --model-path /workspace/model_configs/qwen3_8_27B_config.json \
  --total-gpus 64 --system ascend_910b \
  --backend vllm-ascend --backend-version 0.23.0 \
  --database-mode SILICON \
  --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 \
  --top-n 10 --save-dir results
```

硅数据补齐前临时用 `HYBRID` 跑通链路（缺表处回落 SOL）：

```bash
aic-npu default ... --database-mode HYBRID --total-gpus 64
```

> 27B 是 dense 模型，bf16 权重 ≈ 27×...（远低于 V4-Pro 的 2.9 TB），64 卡放得下；
> 但 GDN 的 `(16,48)` 头数 + `head_dim=128` 在 `tp=8` 时本地仅 `(2,6)`，注意 attention 的
> `num_kv_heads=1`（tp≥4）已由 `ceil(num_kv_heads/tp)` 处理。

---

## 8. 已知限制

1. **GDN 采集是 TP=1 单卡近似**：生产环境按 `tp` 切分 GDN 头，采集器在单卡上用本地头数模拟
   （等价于单卡跑 `tp` 份 rank-local 几何）。SDK 建模同样按本地头数计，口径一致。
2. **`chunk_gated_delta_rule` 独立调用需打桩**：910B 的 triton chunk 在 `chunk_gated_delta_rule_fwd` 里
   无条件读 `get_forward_context().attn_metadata`，采集器用上下文管理器把它 shim 成
   `attn_metadata=None`（纯 prefill、num_decodes=0）以便脱离 vLLM forward pass 单跑。
3. **context 容量上限**：`--max-context-tokens` 默认 1M，跳过 `b×s` 过大的尾部点（避免 HBM OOM）；
   这些点靠插值即可，不写行。
4. **attention `head_dim=256` 的 kernel 支持**：需在采集机上验证 910B FIA 是否支持 256；
   若不支持则对应 16 层回落 SOL（§4.2 已在采集脚本层面给出冒烟验证方法）。
5. **MoE 不适用**：27B 是 dense 模型，`num_experts` 不存在，不涉及 MoE / MoEDispatch 采集。

---

## 9. 参考资料

- 上游模型实现：`aiconfigurator/aic-core/src/aiconfigurator_core/sdk/models/qwen35.py`
- 上游采集器：`aiconfigurator/collector/vllm/collect_gdn.py`、`collector/case_generator.py`
  （`get_common_gdn_test_cases`）、`collector/cases/base_ops/gdn.yaml`
- 上游性能表字段：`aiconfigurator/aic-core/src/aiconfigurator_core/systems/data/h100_sxm/linear_attention/`
- NPU 侧建模：`sdk/models.py`（`Qwen35Model`）、`sdk/operations.py`（`GDNKernel`）、
  `sdk/perf_database.py`（`load_gdn_data` / `query_gdn`）、`sdk/common.py`（`Qwen35Config`）
- vllm-ascend 生产 GDN 路径：`vllm_ascend/ops/gdn.py`（`_forward_core`）、
  `vllm_ascend/ops/triton/fla/chunk.py`（`chunk_gated_delta_rule`）、
  `vllm_ascend/ops/triton/mamba/causal_conv1d.py`（`PAD_SLOT_ID`）、
  `csrc/torch_binding.cpp`（`npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule`）
- 采集脚本：`collector/npu/collect_gdn.py`（新增）、`collector/npu/collect_attn.py`（现有）
- 校验脚本：`tools/check_model_readiness.py`
