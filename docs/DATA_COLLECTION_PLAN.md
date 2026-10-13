# 三模型数据补采计划（Qwen3.8-27B / DeepSeek-R1 / DeepSeek-V4-Pro）

本文档只覆盖上述三个模型。所有命令均按当前仓库脚本的**真实参数**编写（已逐项核对
`collector/npu/*.py` 与 `tools/convert_to_aiconfigurator.py` 的 argparse 定义），可直接复制执行。
现状审计见 `docs/MODEL_ADAPTATION_AUDIT.md`。

约定：以下命令在 **NPU 机器上**执行，工作目录为仓库根目录。

```bash
OUT=src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
PY_PREFIX="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=collector"
```

---

## 0. 一页速览

| 模型 | 当前 SILICON 结果 | 缺什么 | 是否需要硬件 | 主要命令 |
|---|---|---|---|---|
| **DeepSeek-V4-Pro** | agg ✅ 312.70 tokens/s；disagg ❌ | MoE 缺 `ep∈{2,4,8}`（ ep=16/32/64 已有 ） | 是 | §2 两条命令即可（**脚本里已配好，不用改代码**） |
| **DeepSeek-R1** | ❌ 无任何结果 | ① MoE 的 `experts=256` 形状只有 `ep=8`；② 该行挂在 `power_law_1.2` 下而 R1 族查 `power_law_1.01`；③ MLA / `mla_bmm` 缺 `num_heads∈{4,8}` | 是 | §3（**需先改 `MODEL_CONFIGS`**） |
| **Qwen3.8-27B** | ❌ 无任何结果 | ① `gdn_perf.txt` 完全不存在、**且没有采集脚本**；② dense attention 缺 `head_dim=256` | 是 + 需开发新采集器 | §4 |

> **为什么 R1 只有 ep=8 就完全跑不出来**：搜索空间强制 `dp*tp == moe_tp*moe_ep`，
> 而 vLLM 系后端把 `moe_tp` 钉死为 1 ⇒ **`ep` 就等于 worker 宽度**。
> R1 权重约 671 GB，8 张 910B（64 GB/卡）装不下；更宽的 worker 又缺 `ep` 数据 ⇒ 全部配置失效。

---

## 1. 两个必踩的坑（先读）

### 坑 1：转换器的 `--version` 默认是 `0.18.0`

`tools/convert_to_aiconfigurator.py:619` → `default="0.18.0"`。
**每次转换都必须显式写 `--version 0.23.0`**，否则产物里的版本号是错的。

### 坑 2：MoE 的 `distribution` 是 EXACT 查询键

`PerfDatabase._resolve_moe_distribution` 不做模糊匹配。`models.py` 里各族的 `_power_law_alpha`：

| 模型族 | alpha | 查询的键 |
|---|---|---|
| `DeepSeekModel`（V3 / **R1**） | 1.01 | `power_law_1.01` |
| `DeepSeekV4Model`（**V4-Pro**） | 1.01 | `power_law_1.01` |

⇒ **MoE 一律用 `--moe-distribution both` 转换**（一条测量写两行不同 distribution，全族通吃）。
现有的 R1 行只有 `power_law_1.2`，所以现在 R1 在运行时是"借用"了别的 alpha 标签的行
（代码里会有 WARN 提示 `reusing sibling`）——这正是要修掉的东西。

> 附带说明：仓库根的 `systems/data/ascend_910b/vllm-ascend/0.18.0/` 里虽然也有
> `GroupedMatmul_MoE_BF16.csv`，但实测**与 0.23.0 表不是同一次测量**（91 个重合单元格全部不一致，
> 最大偏差 461%），不能拿来补 0.23.0。详见 `systems/README.md`。

---

## 2. DeepSeek-V4-Pro：补 ep = 2 / 4 / 8

**好消息：`collect_moe.py` 里这些条目早就写好了**，只是没采。
`collector/npu/collect_moe.py:104-109`：

```python
(7168, 3072, 384, 6, 64, "deepseek-v4-pro-ep64"),
(7168, 3072, 384, 6, 32, "deepseek-v4-pro-ep32"),
(7168, 3072, 384, 6, 16, "deepseek-v4-pro-ep16"),
(7168, 3072, 384, 6,  8, "deepseek-v4-pro"),
(7168, 3072, 384, 6,  4, "deepseek-v4-pro-ep4"),
(7168, 3072, 384, 6,  2, "deepseek-v4-pro-ep2"),
```

表里现有的 V4 行是 `ep ∈ {16,32,64}`（45 行，`tools/check_dsv4_data.py` 输出可证），
所以**待采的是 `deepseek-v4-pro`（ep=8）、`-ep4`、`-ep2` 这三条**，脚本里已经定义好了。

### 命令

```bash
# ① 采集 ep=8 / 4 / 2（单卡模拟 EP，无需多机）
$PY_PREFIX python collector/npu/collect_moe.py \
  --models deepseek-v4-pro deepseek-v4-pro-ep4 deepseek-v4-pro-ep2 \
  --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  --quant-types bf16 \
  --output-dir ./data/moe_v4_ep248 \
  --resume

# ② 转换并追加（务必：--version 0.23.0 + --moe-distribution both + --append）
python tools/convert_to_aiconfigurator.py \
  --input-dir  ./data/moe_v4_ep248 \
  --output-dir $OUT \
  --version 0.23.0 \
  --moe-distribution both \
  --append
```

> 说明
> - `--models` 的名字**必须与 `MODEL_CONFIGS` 里的第 6 个字段完全一致**。
> - `--token-list` 显式带到 32768：`DEFAULT_TOKEN_LIST` 只到 4096
>   （`collect_moe.py:54`），而运行时 prefill 会查到 16384 以上，否则只能靠 SOL 外推。
> - `--append` 是必须的：不用它会**覆盖**整张 `moe_perf.txt`，把 R1/Qwen3 等其它形状的行丢掉。
> - `--quant-types bf16`：若要 W8A8 需显式加 `w8a8_dynamic`（默认只有 `bf16`）。

### 验收

```bash
python tools/check_model_readiness.py model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json
# 期望：不再报 "MoE collected ... at ep=[16, 32, 64]"

aic-npu default --model-path ./model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json \
  --total-gpus 64 --system ascend_910b --backend vllm-ascend --backend-version 0.23.0 \
  --database-mode SILICON --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 --top-n 10
# 期望：agg 仍约 312.70 tokens/s / 4.89 tokens/s/gpu（不得回退），且 Worker 宽度选择变多
```

> **关于 disagg 仍无结果**：这不是数据缺口。V4-Pro bf16 权重 ≈2894 GiB，
> PD 分离要求 prefill 与 decode 各自持有一份完整权重，64 张 910B 连一份都勉强。
> 想用 disagg 需要量化（W8A8）或更大集群，补 `ep` 数据不会改变这一点。

---

## 3. DeepSeek-R1：补 ep 网格 + 修 distribution + 补 MLA heads

### 3.1 先改代码：给 R1 补 `ep` 条目（一次性）

`collector/npu/collect_moe.py` 的 `MODEL_CONFIGS`（第 63 行起）里，
R1/V3 目前只有一条 `ep=8`（第 69 行）：

```python
    (7168, 2048, 256, 8, 8, "deepseek-v3"),
```

在它**下面**插入以下条目（格式：`hidden, inter, num_experts, topk, ep_size, name`）：

```python
    # DeepSeek-V3 / R1 —— ep 必须与运行时可能的 worker 宽度对齐
    (7168, 2048, 256, 8, 64, "deepseek-v3-ep64"),
    (7168, 2048, 256, 8, 32, "deepseek-v3-ep32"),
    (7168, 2048, 256, 8, 16, "deepseek-v3-ep16"),
    (7168, 2048, 256, 8,  8, "deepseek-v3"),        # 保留原名；若原始 CSV 还在可避免重采
    (7168, 2048, 256, 8,  4, "deepseek-v3-ep4"),
    (7168, 2048, 256, 8,  2, "deepseek-v3-ep2"),
    (7168, 2048, 256, 8,  1, "deepseek-v3-ep1"),
```

显存可行性（256 experts × inter 2048，bf16）：

| ep | 本地专家数 | 常驻权重 | 910B 64 GB | 建议 |
|---:|---:|---:|---|---|
| 64 | 4 | ~0.4 GiB | 富余 | 必采 |
| 32 | 8 | ~0.8 GiB | 富余 | 必采 |
| 16 | 16 | ~1.6 GiB | 富余 | 必采 |
| 8 | 32 | ~3.2 GiB | 安全 | 已有数据 |
| 4 | 64 | ~6.3 GiB | 安全 | 必采 |
| 2 | 128 | ~12.7 GiB | 安全 | 必采 |
| 1 | 256 | ~25.3 GiB | 紧张但可行 | 选采 |

### 3.2 采集命令

```bash
# ① MoE ep 网格
$PY_PREFIX python collector/npu/collect_moe.py \
  --models deepseek-v3 deepseek-v3-ep2 deepseek-v3-ep4 \
           deepseek-v3-ep16 deepseek-v3-ep32 deepseek-v3-ep64 \
  --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  --quant-types bf16 \
  --output-dir ./data/moe_r1_ep \
  --resume

# ② 转换并追加 —— --moe-distribution both 同时写 power_law_1.2 与 power_law_1.01，
#    这一步就是修掉 R1 查不到 distribution 的根本手段
python tools/convert_to_aiconfigurator.py \
  --input-dir  ./data/moe_r1_ep \
  --output-dir $OUT \
  --version 0.23.0 \
  --moe-distribution both \
  --append
```

### 3.3 补 MLA / mla_bmm 的 `num_heads = 4 / 8`

R1 是 128 头模型，`tp` 切分后每卡注意力头数 = `128 / tp`：

| tp | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| rank-local heads | 128 | 64 | 32 | 16 | **8** | **4** | 2 |
| 现有 MLA / mla_bmm 表 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

> `heads=2`（tp=64）**不要补**：Ascend `SparseAttnSharedkv` 要求 `q_head_num` 是 4 的倍数，
> kernel 层面 tiling 直接失败（详见 `docs/DeepSeek-V4-Pro_ADAPTATION.md` 的 F13）。

**注意 R1 的真实维度**（取自 `model_configs/deepseek-ai--DeepSeek-R1_config.json`，
与 `collect_mla.py` 的默认值**不同**，必须显式传）：
`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`
（脚本默认是 `192 / 256`，那是给 V3.2 用的）。

```bash
# ③ MLA context + generation（--output-format mla 会直接产出最终的 *_perf.txt，无需转换）
$PY_PREFIX python collector/npu/collect_mla.py \
  --model deepseek-ai/DeepSeek-R1 \
  --architecture DeepseekV3ForCausalLM \
  --op-types context generation \
  --num-heads-list 4 8 16 32 64 128 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
  --output-format mla \
  --output-dir ./data/mla_r1 \
  --resume

# → 产出 ./data/mla_r1/context_mla_perf.txt 与 generation_mla_perf.txt
cp ./data/mla_r1/context_mla_perf.txt     $OUT/context_mla_perf.txt
cp ./data/mla_r1/generation_mla_perf.txt  $OUT/generation_mla_perf.txt

# ④ mla_bmm（该脚本直接产出 mla_bmm_perf.txt；--framework/--version 默认已是 vllm-ascend / 0.23.0）
$PY_PREFIX python collector/npu/collect_mla_bmm.py \
  --op-types pre post \
  --num-heads-list 4 8 16 32 64 128 \
  --kv-lora-rank 512 --head-dim 128 \
  --output-dir ./data/mla_bmm_r1 \
  --resume

cp ./data/mla_bmm_r1/mla_bmm_perf.txt $OUT/mla_bmm_perf.txt
```

> `--num-heads-list` 必须显式写：`collect_mla.py` 默认只有 `[64]`，
> `collect_mla_bmm.py` 默认只有 `[128,64,32,16]`。

### 3.4 验收

```bash
python tools/check_model_readiness.py model_configs/deepseek-ai--DeepSeek-R1_config.json
# 期望：不再出现 "MLA table has num_heads=..." 与 "ep=[8] only"

aic-npu default --model-path ./model_configs/deepseek-ai--DeepSeek-R1_config.json \
  --total-gpus 64 --system ascend_910b --backend vllm-ascend --backend-version 0.23.0 \
  --database-mode SILICON --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 --top-n 10
# 期望：agg 与 disagg 均有结果，且**不需要** AIC_NPU_MOE_TRANSFER 环境变量
```

---

## 4. Qwen3.8-27B：工作量最大（需要新增采集器）

`model_configs/qwen3_8_27B_config.json` → `Qwen3_5ForConditionalGeneration` → `Qwen35Model`，
`layer_types` 里 `linear_attention : full_attention = 3 : 1`。

| 缺口 | 现状 | 怎么补 |
|---|---|---|
| 🔴 `gdn_perf.txt` | **连文件都没有**，且 `collector/npu/` 下**没有任何脚本能产出它** | 需新增 `collector/npu/gdn_factory.py` + `collect_gdn.py`，对标 NVIDIA 侧 `data/<gpu>/linear_attention/` 的粒度 |
| 🔴 `head_dim=256` 的 dense attention | `context/generation_attention_perf.txt` 只有 `head_dim=128` | 现有脚本支持，只是 `--head-size` 默认 128，**重跑一次即可** |

### 4.1 立即可做：补 `head_dim=256`

```bash
# 注意 Qwen3.8-27B 只有 24 个注意力头 —— tp 必须是 24 的约数，否则
# BaseModel 的 assert num_heads % tp_size == 0 直接终止该配置
$PY_PREFIX python collector/npu/collect_attn.py \
  --op-types context generation \
  --head-size 256 \
  --num-heads-list 3 6 12 24 8 16 32 40 64 \
  --num-kv-heads-list 0 1 4 8 \
  --output-dir ./data/attn_hd256 \
  --resume

python tools/convert_to_aiconfigurator.py \
  --input-dir  ./data/attn_hd256 \
  --output-dir $OUT \
  --version 0.23.0
```

> `collect_attn.py` 产出的是 CSV，**必须**走转换器（不像 MLA 脚本直出 `_perf.txt`）。
> `--num-kv-heads-list 0` 表示 MHA。

### 4.2 需要开发：GDN 采集器

先确认可用的 kernel 入口。现有可参考的实现与文档：

- `collector/npu/mla_factory.py`（结构上最接近：构造 metadata + 调 `torch_npu` 算子）
- `docs/ckw/docs/` 下关于 NPU linear-attention / GDN kernel 的记录
- NVIDIA 侧 `aiconfigurator/aic-core/.../systems/data/<gpu>/linear_attention/` 的表格字段，
  用来对齐最终 `gdn_perf.txt` 的列定义

产出目标文件 `$OUT/gdn_perf.txt`，列定义需与 `PerfDataFilename` 中 GDN 的加载器一致。
**建议先和 MLA 分支的创作同学确认 kernel 可获得性，再列详细工期** —— 这是三个模型里唯一
有"能不能做"不确定性的项。

### 4.3 验收

```bash
python tools/check_model_readiness.py model_configs/qwen3_8_27B_config.json
# 期望：不再报 "requires gdn_perf.txt" 与 "collected head_dim=[128] but the model needs 256"
```

---

## 5. 通用补采（三个模型共同受益）

### 5.1 多机通信：16 / 32 / 64 卡

现在 `nccl_perf.txt` / `custom_allreduce_perf.txt` 的 `num_gpus` 只到 **8** ——
64 卡集群的 TP/EP 通信全靠外推。`collect_hccl.py` 的 `num_gpus` 取自 `dist.get_world_size()`，
所以**必须多机 torchrun**：

```bash
# 16 卡示例（2 节点 × 8 卡）；32/64 卡同理调 --nnodes
torchrun --nnodes=2 --nproc_per_node=8 --master_addr=$MASTER --master_port=29500 \
  collector/npu/collect_hccl.py --ops all_reduce all_gather reduce_scatter \
  --output-dir ./data/hccl_16p

cp ./data/hccl_16p/nccl_perf.txt             $OUT/nccl_perf.txt
cp ./data/hccl_16p/custom_allreduce_perf.txt $OUT/custom_allreduce_perf.txt
```

### 5.2 量化 dtype（目前 `gemm_perf.txt` 只有 `float16` / `sq`）

```bash
$PY_PREFIX python collector/npu/collect_gemm.py --quant-types bf16 w8a8_dynamic --output-dir ./data/gemm_q
$PY_PREFIX python collector/npu/collect_moe.py  --quant-types bf16 w8a8_dynamic --output-dir ./data/moe_q
```

对 §2 的 V4-Pro disagg 尤其关键：W8A8 把 ~2894 GiB 权重砍一半，PD 分离才可能成立。

---

## 6. 推荐执行顺序

| 顺序 | 做什么 | 依据 | 耗时门槛 |
|---:|---|---|---|
| 1 | **V4-Pro 补 ep=2/4/8** | 脚本已配好，两条命令，零代码改动 | 最低，收益最快 |
| 2 | **V4-Pro 采 W8A8** | 这是让 V4 disagg 变得可行的唯一现实路径 | 中 |
| 3 | **R1 改 `MODEL_CONFIGS` + 采 ep 网格** | R1 从"完全不能跑"变为可用 | 中 |
| 4 | **R1 补 MLA / mla_bmm 的 heads=4/8** | 解锁 tp=16/32 | 中 |
| 5 | **Qwen3.8-27B 补 `head_dim=256`** | 解锁一半阻塞项 | 低 |
| 6 | **Qwen3.8-27B 开发 GDN 采集器** | 该模型能否上线的决定性项 | 最高，且有不确定性 |
| 7 | 多机通信 16/32/64 卡 | 提升大模型部署精度 | 需要多机窗口 |

---

## 7. 每次补采后的固定验收

```bash
# ① 静态体检（秒级，不需要硬件）
python tools/check_model_readiness.py

# ② 三模型 SILICON 回归（不应依赖任何环境变量逃生舱）
for m in deepseek-ai--DeepSeek-R1 deepseek-ai--DeepSeek-V4-Pro qwen3_8_27B; do
  aic-npu default --model-path ./model_configs/${m}_config.json \
    --total-gpus 64 --system ascend_910b --backend vllm-ascend --backend-version 0.23.0 \
    --database-mode SILICON --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 --top-n 10
done

# ③ 基线防回退：V4-Pro agg 应仍为约 312.70 tokens/s / 4.89 tokens/s/gpu
```

补采完成后，同步更新 `src/aiconfigurator_npu/systems/support_matrix.csv` 与
`docs/MODEL_ADAPTATION_AUDIT.md` 的结论表。
