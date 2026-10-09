# DeepSeek-R1 适配文档

目标：让 `aic-npu` 能完成 `deepseek-ai/DeepSeek-R1` 在 `ascend_910b` + `vllm-ascend` 上的配置搜索。

创建日期：2026-10-09

---

## 0. 结论

DeepSeek-R1 与 DeepSeek-V3 **同架构（`DeepseekV3ForCausalLM`）、同维度、同 MoE 配置**，
因此算子建模代码无需改动，直接复用已有的 `DEEPSEEK` 模型族与 `DeepSeekModel`。
适配工作量集中在 **配置登记** 与 **MLA 性能数据采集** 两类。

| 类别 | 任务数 | 已完成 | 需 NPU 硬件 |
|------|-------|-------|------------|
| A. 模型配置登记 | 4 | 4 | 否 |
| B. 代码适配（采集/转换链路） | 2 | 2 | 否 |
| C. 性能数据采集 | 3 | 1（GEMM/MoE/Comm 复用现有） | **是** |
| D. 验证 | 2 | 0 | 是 |

---

## 1. 架构与建模路径

```
config.json architectures[0] = "DeepseekV3ForCausalLM"
    ↓ common.ARCHITECTURE_TO_MODEL_FAMILY        → "DEEPSEEK"
    ↓ sdk/models.py get_model()                  → DeepSeekModel (sdk/models.py:1207)
    ↓ context_ops / generation_ops               → GEMM / MoE / ContextMLA / GenerationMLA / MLABmm / Comm
    ↓ PerfDatabase.query_*()                     → gemm_perf.txt / moe_perf.txt / context_mla_perf.txt ...
```

DeepSeek-R1 关键参数：

| 参数 | 值 | 与 V3 是否一致 |
|------|-----|--------------|
| `hidden_size` | 7168 | ✅ |
| `num_hidden_layers` | 61 | ✅ |
| `num_attention_heads` / `num_key_value_heads` | 128 / 128 | ✅ |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | ✅ |
| `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim` | 128 / 64 / 128 | ✅ |
| `n_routed_experts` / `num_experts_per_tok` | 256 / 8 | ✅ |
| `moe_intermediate_size` | 2048 | ✅ |
| `first_k_dense_replace` | 1 | ✅ |
| `num_nextn_predict_layers` | 1 | ✅ |
| `vocab_size` | 129280 | ✅ |

`DeepSeekModel` 中 MLA 相关维度是硬编码的 DSV3 值
（`2112 / 24576 / 1536 / 32768 / 512 / 128`，见 `sdk/models.py:1274-1298`），
与 R1 完全一致，因此**无需新增模型类**。

---

## 2. 任务清单

### A 类：模型配置登记（无需硬件）

| # | 任务 | 位置 | 状态 |
|---|------|------|------|
| A1 | 新增 R1 离线 HF config | `model_configs/deepseek-ai--DeepSeek-R1_config.json` | ✅ |
| A2 | 补齐 V3 离线 HF config（support_matrix 已登记但缺缓存，离线会失败） | `model_configs/deepseek-ai--DeepSeek-V3_config.json` | ✅ |
| A3 | 支持矩阵登记 R1（agg + disagg） | `src/aiconfigurator_npu/systems/support_matrix.csv` | ✅ |
| A4 | `DefaultHFModels` 登记 R1；`SupportedSystems` 补 `ascend_910b` | `src/aiconfigurator_npu/sdk/common.py` | ✅ |

**A3 附带修复**：`support_matrix.csv` 中 DeepSeek-V3 行的架构名原为 `DeepSeekV3ForCausalLM`（大写 S），
与 `ARCHITECTURE_TO_MODEL_FAMILY` 的键 `DeepseekV3ForCausalLM`（小写 s）不一致。
`check_support()` 的架构兜底分支是精确字符串比较，大小写不一致会导致按架构推断失效，已统一。

**A4 说明**：`SupportedSystems` 原先只含 NVIDIA GPU 系统，`aic-npu support --system all`
不会遍历 `ascend_910b`，已补入。

> 配置文件命名规则见 `sdk/utils.py:_find_pre_downloaded_hf_file()`：
> `{hf_id 把 / 替换为 --}_config.json`，查找目录为 `aiconfigurator_npu/model_configs`（包内）
> 与 `./model_configs`（CWD）。

### B 类：代码适配（无需硬件）

| # | 任务 | 位置 | 状态 |
|---|------|------|------|
| B1 | `collect_mla.py` 新增 `mla` output-format，直接产出 kernel 级 MLA 表 | `collector/npu/collect_mla.py` | ✅ |
| B2 | `convert_to_aiconfigurator.py` 新增 `convert_mla()` + `--mla-output` 开关 | `tools/convert_to_aiconfigurator.py` | ✅ |

**背景**：DeepSeek-R1 属于 `DEEPSEEK` 族，attention 走**内核级 MLA 表**；
而 GLM-5 / DeepSeek-V3.2 属于 `DEEPSEEKV32` 族，走**模块级 DSA 表**。
原采集脚本只有 `tensorcast` 和 `dsa_module` 两种输出，无法产出 R1 需要的表，故补齐。

B1 产出：

| 文件 | 列 |
|------|-----|
| `context_mla_perf.txt` | `framework,version,device,op_name,mla_dtype,kv_cache_dtype,batch_size,isl,num_heads,latency` |
| `generation_mla_perf.txt` | 同上 + `step` |

对应 `perf_database.py` 的索引结构：

```
context:    data[FMHAQuantMode][KVCacheQuantMode][num_heads][isl][batch]
generation: data[KVCacheQuantMode][num_heads][batch][isl + step]
```

### C 类：性能数据采集（**需 NPU 硬件**）

| # | 数据 | 状态 | 说明 |
|---|------|------|------|
| C1 | `gemm_perf.txt` | ✅ 已有 | 覆盖 R1 线性层维度 |
| C2 | `moe_perf.txt` | ✅ 已有 | `GroupedMatmul_MoE_*.csv` 已含 256 experts / topk 8 / 7168 / 2048 |
| C3 | `custom_allreduce_perf.txt`、`nccl` | ✅ 已有 | 通信 |
| C4 | `context_mla_perf.txt` / `generation_mla_perf.txt` | ⬜ **待采集** | R1 唯一缺口 |
| C5 | `mla_bmm_perf.txt` | ⬜ 待采集 | 仅 generation 的 bmm pre/post 使用；缺失时 HYBRID 走 SOL 估算 |

#### C4 采集命令（在 NPU 机器上执行）

```bash
# R1 维度：128 heads / kv_lora_rank 512 / qk_nope 128 / qk_rope 64 / v_head 128
python collector/npu/collect_mla.py \
  --output-format mla \
  --architecture DeepseekV3ForCausalLM \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 \
  --qk-nope-head-dim 128 \
  --qk-rope-head-dim 64 \
  --v-head-dim 128 \
  --framework vllm-ascend \
  --version 0.18.0 \
  --device "Ascend 910B" \
  --mla-dtype float16 \
  --kv-cache-dtype float16 \
  --output-dir ./data/dsr1_mla

# 拷贝到性能数据库目录
cp data/dsr1_mla/context_mla_perf.txt \
   src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/
cp data/dsr1_mla/generation_mla_perf.txt \
   src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0/
```

`--num-heads-list` 需要覆盖 `128 // tp_size`，即 TP=1/2/4/8 分别对应 128/64/32/16。

#### C4 备选：先采 TensorCast CSV 再转换

```bash
python collector/npu/collect_mla.py \
  --architecture DeepseekV3ForCausalLM \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
  --output-dir ./data/dsr1_mla_raw

python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/dsr1_mla_raw \
  --output-dir ./src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.18.0 \
  --device "Ascend 910B" --framework vllm-ascend --version 0.18.0 \
  --mla-output mla
```

> `--mla-output` 取值：`dsa`（默认，V3.2/GLM-5 模块表）/ `mla`（V3/R1 内核表）/ `both` / `none`。

### D 类：验证（需 NPU/运行环境）

| # | 任务 | 命令 |
|---|------|------|
| D1 | 配置解析 + 支持矩阵校验 | `aic-npu support --model deepseek-ai/DeepSeek-R1 --system ascend_910b --backend vllm` |
| D2 | 配置搜索跑通 | 见下节 |

---

## 3. 运行配置搜索

### 3.1 MLA 数据缺失时：先用 HYBRID 模式

`PerfDatabase._default_database_mode` 默认是 `SILICON`，
缺 MLA 数据时 `query_context_mla()` / `query_generation_mla()` 会抛异常
（`perf_database.py:_query_silicon_or_hybrid()`，SILICON 不回退）。
因此采集完成前必须显式用 HYBRID，让 MLA 走 SOL 解析估算：

```bash
aic-npu default \
  --model deepseek-ai/DeepSeek-R1 \
  --system ascend_910b \
  --backend vllm \
  --database-mode hybrid \
  --total-gpus 16
```

Python API 等价写法：

```python
from aiconfigurator_npu.sdk.task import TaskConfig
from aiconfigurator_npu.sdk.common import DatabaseMode, GEMMQuantMode, MoEQuantMode

task = TaskConfig(
    model="deepseek-ai/DeepSeek-R1",
    backend="vllm-ascend",
    system="ascend_910b",
    database_mode=DatabaseMode.HYBRID,   # MLA 数据补齐后改 SILICON
    isl=4096,
    osl=512,
    num_requests=100,
    gemm_quant_mode=GEMMQuantMode.float16,      # W8A8 用 w8a8_dynamic
    moe_quant_mode=MoEQuantMode.float16,
)
results = task.run()
results.pareto_analysis(ttft_sla_ms=3000, tpot_sla_ms=50)
results.print_pareto_table()
```

### 3.2 MLA 数据补齐后：切 SILICON

数据文件就位后去掉 `--database-mode hybrid` 即可（默认 SILICON）。

---

## 4. 已知限制

1. **`first_k_dense_replace=1` 未建模**：R1 第 0 层是 dense MLP（`intermediate_size=18432`），
   `DeepSeekModel` 按全部 61 层均为 MoE 建模，误差约 1/61。
2. **MTP 已建模**：`num_nextn_predict_layers=1` 由 `_mtp_scale_factor` 处理，无需额外配置。
3. **W8A8 在 decode 更慢**：小 M（≤128）时 W8A8 比 BF16 慢约 30%
   （详见 `docs/GLM5_ADAPTATION_DESIGN.md` 8.3）。建议 decode 用 `float16`，prefill 再开 `w8a8_dynamic`。
4. **`mla_bmm_perf.txt` 无采集脚本**：目前没有对应的 collector；
   在 HYBRID 模式下由 SOL 估算（`query_mla_bmm()` 的 `get_sol`），SILICON 模式仍会报错。
   若需要 SILICON 精度，需补一个 BMM 采集（`op_name` 必须是 `mla_gen_pre` / `mla_gen_post`）。
5. **MoE 并行约束**：vllm-ascend 不支持同时按 TP 和 EP 切分 MoE 权重
   （`sdk/utils.py:enumerate_parallel_config()`），R1 只能选纯 TEP / 纯 DEP / 纯 TP。

---

## 5. 与 GLM-5 适配的差异

| 维度 | GLM-5 / V3.2 | DeepSeek-R1 / V3 |
|------|-------------|-----------------|
| 架构名 | `GlmMoeDsaForCausalLM` / `DeepseekV32ForCausalLM` | `DeepseekV3ForCausalLM` |
| 模型族 | `DEEPSEEKV32` | `DEEPSEEK` |
| 模型类 | `DeepSeekV32Model` | `DeepSeekModel` |
| Attention 表 | 模块级 `dsa_*_module_perf.txt` | 内核级 `context/generation_mla_perf.txt` |
| 额外结构参数 | 需要 `index_topk` / `index_n_heads` / `index_head_dim` 等 DSA 字段 | 不需要（维度硬编码） |
| 采集脚本 | `collect_mla_module.py` | `collect_mla.py --output-format mla` |

---

## 6. 参考资料

- 模型族映射：`src/aiconfigurator_npu/sdk/common.py:ARCHITECTURE_TO_MODEL_FAMILY`
- 模型实现：`src/aiconfigurator_npu/sdk/models.py:DeepSeekModel`
- MLA 表加载：`src/aiconfigurator_npu/sdk/perf_database.py:load_context_mla_data` / `load_generation_mla_data` / `load_mla_bmm_data`
- GLM-5 适配（同类工作参考）：`docs/GLM5_ADAPTATION_DESIGN.md`
