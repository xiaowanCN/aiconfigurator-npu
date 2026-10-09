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
| C. 性能数据采集 | 5 | 4（C1–C4；C5 脚本已就绪，待硬件采集） | **是** |
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
| C4 | `context_mla_perf.txt` / `generation_mla_perf.txt` | ✅ 已采集 | context 224 点 + generation 256 点 |
| C5 | `mla_bmm_perf.txt` | ⬜ 待采集 | 仅 generation 的 bmm pre/post 使用；采集脚本 `collect_mla_bmm.py` 已就绪，缺失时 HYBRID 走 SOL 估算 |

#### C4 采集命令（在 NPU 机器上执行）

> 镜像基线是 `quay.io/ascend/vllm-ascend:v0.23.0.post1`，所以 `--version` 写 `0.23.0`，
> 数据也落到 `0.23.0/` 目录（见下面的「0.23 数据落位」）。

R1 维度：`128 heads / kv_lora_rank 512 / qk_nope 128 / qk_rope 64 / v_head 128`。

**Context 与 generation 必须分开跑** —— 两者显存模型完全不同：prefill 的输入是
`batch × seq × heads × dim`，`batch=128, seq=8192` 会到数十 GB 直接 OOM；decode 的
输入是 `batch × heads` 加上 KV block，占用低，可以跑满默认范围。

```bash
# 1) context（prefill）—— 限制 batch×seq 防止 OOM
python collector/npu/collect_mla.py \
  --output-format mla \
  --architecture DeepseekV3ForCausalLM \
  --model deepseek-ai/DeepSeek-R1 \
  --op-types context \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8096 \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 \
  --qk-nope-head-dim 128 \
  --qk-rope-head-dim 64 \
  --v-head-dim 128 \
  --framework vllm-ascend \
  --version 0.23.0 \
  --device "Ascend 910B" \
  --mla-dtype float16 \
  --kv-cache-dtype float16 \
  --output-dir ./data/dsr1_mla_ctx \
  --resume

# 2) generation（decode）—— 显存占用低，可跑满默认范围
python collector/npu/collect_mla.py \
  --output-format mla \
  --architecture DeepseekV3ForCausalLM \
  --model deepseek-ai/DeepSeek-R1 \
  --op-types generation \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 \
  --qk-nope-head-dim 128 \
  --qk-rope-head-dim 64 \
  --v-head-dim 128 \
  --framework vllm-ascend \
  --version 0.23.0 \
  --device "Ascend 910B" \
  --mla-dtype float16 \
  --kv-cache-dtype float16 \
  --output-dir ./data/dsr1_mla_gen \
  --resume

python collector/npu/collect_mla.py \
  --output-format mla \
  --architecture DeepseekV3ForCausalLM \
  --model deepseek-ai/DeepSeek-R1 \
  --op-types generation \
  --batch-list 1 2 4 8 16 32 64 128 256 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 32768 \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --mla-dtype float16 --kv-cache-dtype float16 \
  --output-dir ./data/dsr1_mla_gen
```

要点：

- `--num-heads-list` 要覆盖 `128 // tp_size`，即 TP=1/2/4/8 对应 128/64/32/16。
- `--model` 默认值就是 `deepseek-ai/DeepSeek-R1`，显式写出来便于切换；它只用于构造
  合成 `VllmConfig`（只读 config、不加载权重），要求 `model_configs/` 下有缓存。
- 两个阶段要用**不同输出目录**：`mla_checkpoint.json` 是单文件 checkpoint，同目录
  分两批跑会互相覆盖已完成记录。
- 参数是**空格**分隔（`nargs="+"`），写成 `1,2` 会报 `invalid int value`。
- `--resume` 支持中断续跑；失败的 spec 不写入 checkpoint，修好后会重跑。

#### MLA 表的插值要求（决定采集网格）

`query_context_mla()` / `query_generation_mla()` 都走 `_interp_3d()`，即**三维插值**，
三个轴全部参与：

| 阶段 | 调用 | 轴顺序 | 插值方式 |
|------|------|--------|---------|
| context | `_interp_3d(num_heads, full_s, b, data, "cubic")` | num_heads / seq / batch | cubic |
| generation | `_interp_3d(num_heads, b, s, data, "bilinear")` | num_heads / batch / seq | bilinear |

由此推出采集网格的硬约束：

1. **`num_heads` 也是插值轴**，不是附属维度。`DeepSeekModel` 查的是 `128 // tp_size`
   （见 `models.py:ContextMLA/GenerationMLA`），所以必须采集 `128 / 64 / 32 / 16`
   对应 TP=1/2/4/8；每个轴上至少 2 个采样点。
2. **不能外推**：`_nearest_1d_point_helper()` 默认 `inner_only=True`，查询点落在采集
   范围之外会直接 `raise ValueError`。网格必须**覆盖**实际查询范围。
3. **context 用 cubic**（`scipy.interpolate.griddata(method="cubic")`）：只用矩形四角
   时精度很差甚至产生 NaN，每个 `num_heads` 下的 `(seq, batch)` 网格建议至少 3×3，
   默认的 7×8 很充裕。
4. **generation 用 bilinear**（`_bilinear_interpolation`）：2×2 网格即可成立，
   但同样受约束 2 的覆盖要求。

推荐网格（比默认更宽，覆盖大并发 / 长上下文场景）：

| 轴 | 取值 |
|----|------|
| num_heads | `128 64 32 16` |
| batch | `1 2 4 8 16 32 64 128`（并发更高时补 `256`） |
| context seq | `128 256 512 1024 2048 4096 8192` |
| generation seq | `128 256 512 1024 2048 4096 8192 16384`（长上下文补 `32768`） |

> 采集脚本对每个 spec 都有 `try/except`，单点失败（含 OOM）只累加 error 并继续，
> 不会中断整批。所以可以直接跑满网格，跑完再统计缺哪些点、按需补采。

#### 0.23 数据落位

镜像里实际是 vllm-ascend 0.23.0，但仓库性能库当前只有 `0.18.0/` 目录。需要新建
`0.23.0/` 并把已有数据平移过去（加载时不校验文件内 `version` 列，只按目录定位）：

```bash
cd src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend
mkdir -p 0.23.0 && cp 0.18.0/*.txt 0.23.0/

cp data/dsr1_mla_ctx/context_mla_perf.txt 0.23.0/
cp data/dsr1_mla_gen/generation_mla_perf.txt 0.23.0/
```

同时把 `support_matrix.csv` 的 `Version` 列改为 `0.23.0`（`check_support()` 按
Version 精确匹配）。注意 `get_latest_database_version()` 会选最新版本，建了 `0.23.0/`
后它就是默认版本，务必保证该目录数据完整。

#### C4 备选：先采 TensorCast CSV 再转换

```bash
python collector/npu/collect_mla.py \
  --architecture DeepseekV3ForCausalLM \
  --model deepseek-ai/DeepSeek-R1 \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
  --output-dir ./data/dsr1_mla_raw

python tools/convert_to_aiconfigurator.py \
  --input-dir ./data/dsr1_mla_raw \
  --output-dir ./src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0 \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --mla-output mla
```

> `--mla-output` 取值：`dsa`（默认，V3.2/GLM-5 模块表）/ `mla`（V3/R1 内核表）/ `both` / `none`。

#### C5：`mla_bmm_perf.txt`（MLA decode 的两个 batched 投影）

**它测什么**：MLA decode 的「未吸收（non-absorbed）」路径中两个 batched GEMM：

| `op_name` | 计算 | 形状（V3/R1：`kv_lora_rank=512`、`head_dim=128`） |
|---|---|---|
| `mla_gen_pre` | `kv_c @ W_uk` | `[H, T, 512] @ [H, 512, 128]` → `[H, T, 128]` |
| `mla_gen_post` | `attn_out @ W_uv` | `[H, T, 128] @ [H, 128, 512]` → `[H, T, 512]` |

只在 **generation（decode）** 路径使用（`models.py:1436` / `1449`），context/prefill 没有它。
索引结构为 `data[GEMMQuantMode][op_name][num_heads][num_tokens]`。

**缺失时的影响**：

| 模式 | 行为 |
|---|---|
| `SILICON` | `self._mla_bmm_data.raise_if_not_loaded()` → 抛异常，跑不了 |
| `HYBRID` | `get_empirical` = `SOL / 0.8`（`perf_database.py:5190-5192`） |

量级参考：batch=128 / heads=128 时两个 BMM 合计约 47 µs/层，占单层 decode 的百分之几。
**HYBRID 下不采也能跑**，但要 SILICON 精度（或提高 TPOT 预测准确度）就必须采。

**采集命令**（`collector/npu/collect_mla_bmm.py`）：

```bash
python collector/npu/collect_mla_bmm.py \
  --num-tokens-list 1 2 4 8 16 32 64 128 256 \
  --num-heads-list 128 64 32 16 \
  --kv-lora-rank 512 \
  --head-dim 128 \
  --op-types pre post \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --bmm-dtype float16 \
  --output-dir ./data/dsr1_mla_bmm
```

2 op × 9 tokens × 4 heads = 72 个点。纯 GEMM，没有 attention 那类 tiling 坑，一两分钟可完成。
建议先冒烟 4 个点：

```bash
python collector/npu/collect_mla_bmm.py \
  --num-tokens-list 1 128 --num-heads-list 128 \
  --warmup-iters 1 --bench-iters 3 \
  --output-dir ./data/dsr1_mla_bmm_smoke
```

**要点**：

- `--num-tokens-list` 是**唯一被插值**的轴（decode 并发），要覆盖实际查询的 batch。
- `--num-heads-list` 是精确 key，必须含 `128 // tp_size` = `128 / 64 / 32 / 16`。
- `op_name` 只能是 `mla_gen_pre` / `mla_gen_post` —— `query_mla_bmm()` 里写死，不能改。
- `--bmm-dtype` 目前仅 `float16`（实测走 bf16）。`--gemm-quant-mode w8a8_dynamic` 下
  `mla_bmm_quant_mode` 会变成 `fp8`，此时 `query_mla_bmm()` 有 fallback
  （`quant_mode_lookup = quant_mode if quant_mode in data else float16`），能跑但按 bf16 估。
- 支持 `--resume` 续跑，checkpoint 为 `mla_bmm_checkpoint.json`。

**落位**：

```bash
cp ./data/dsr1_mla_bmm/mla_bmm_perf.txt \
   src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0/
```

补齐后 `0.23.0/` 共 8 张表（`gemm` / `moe` / `context_attention` / `generation_attention` /
`custom_allreduce` / `context_mla` / `generation_mla` / `mla_bmm`），**SILICON 模式即可直接使用**。

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
4. **`mla_bmm_perf.txt` 尚未采集**：采集脚本 `collector/npu/collect_mla_bmm.py` 已就绪，
   但还没在硬件上跑过（脚本本身也未经验证）。缺表时 HYBRID 由 SOL 估算
   （`query_mla_bmm()` 的 `get_empirical` = `SOL / 0.8`），SILICON 模式会报错。
   采集方式见 **C5**。另外 `--bmm-dtype` 目前只支持 `float16`，W8A8 配置下这两个投影
   会 fallback 到 bf16 估算。
5. **MoE 并行约束**：vllm-ascend 不支持同时按 TP 和 EP 切分 MoE 权重
   （`sdk/utils.py:enumerate_parallel_config()`），R1 只能选纯 TEP / 纯 DEP / 纯 TP。

---

## 5. vllm-ascend 0.23.0 兼容性修复记录

`collector/npu/mla_factory.py` 原本是照 vllm-ascend 0.18 写的，在 0.23.0 上采集会
连续触发四个问题。修复后 context / generation 两条路径均已冒烟通过。

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | `AttributeError: 'MockConfig' object has no attribute 'tensor_parallel_size'` | 0.23 的 `AscendConfig.__init__` 新增 `get_flashcomm2_config_and_validate()`，读 `parallel_config.tensor_parallel_size` | 见 2（逐个补 mock 属性是打地鼠，直接换真实 config） |
| 2 | `...no attribute 'enforce_eager'` / `'runner_type'` | `AscendConfig.__init__`、`platform.set_additional_forward_context`、`VllmConfig.use_v2_model_runner` 多处读真实 `ModelConfig` | 放弃手工 `MockConfig`，改用真实 `VllmConfig`（复用 `mla_module_factory._create_npu_vllm_config`） |
| 3 | `TypeError: AscendMLAMetadata.__init__() missing 1 required positional argument: 'seq_lens_cpu'` | 0.23 给 `AscendMLAMetadata` 新增必填字段 `seq_lens_cpu` | `_ascend_mla_metadata_extra()`：用 `inspect.signature` 探测，仅在字段存在时补（兼容 0.18） |
| 4 | `ERR01001 ... atten mask ... has incorrect shape [1024,1024]` / `[8192,8192]` | CANN 的 `sparse_mode=3` 只接受 `[2048,2048]`、`[1,2048,2048]`、`[1,1,2048,2048]`，**且该形状与 seq_len 无关** | `_make_causal_mask()`：固定 2048×2048 上三角 mask 并缓存 |

补充观察：

- 问题 4 曾出现过"部分点能出结果"的假象：`DEFAULT_SEQ_CONTEXT` 里只有 `seq_len=2048`
  这一档恰好构造出 `[2048,2048]` 而通过了校验，其余全部 reject。
- `_forward_decode` 在 0.23 里把 `sparse_mode = 0` 和 `attn_mask = None` **写死**，
  并不读 `metadata.attn_mask`；decode 侧传的 mask 当前未被 kernel 使用。保留它是为了
  避免 `None` 在其它路径下触发 `ERR01001`，且语义上无害（decode 的 query 长度为 1，
  能看到全部历史 KV）。
- 采集过程中若 NPU 被其它进程占用，会出现 `std::logic_error` + `Aborted (core dumped)`。
  这是环境问题而非代码问题，采集前先确认卡空闲。

## 6. 与 GLM-5 适配的差异

| 维度 | GLM-5 / V3.2 | DeepSeek-R1 / V3 |
|------|-------------|-----------------|
| 架构名 | `GlmMoeDsaForCausalLM` / `DeepseekV32ForCausalLM` | `DeepseekV3ForCausalLM` |
| 模型族 | `DEEPSEEKV32` | `DEEPSEEK` |
| 模型类 | `DeepSeekV32Model` | `DeepSeekModel` |
| Attention 表 | 模块级 `dsa_*_module_perf.txt` | 内核级 `context/generation_mla_perf.txt` |
| 额外结构参数 | 需要 `index_topk` / `index_n_heads` / `index_head_dim` 等 DSA 字段 | 不需要（维度硬编码） |
| 采集脚本 | `collect_mla_module.py` | `collect_mla.py --output-format mla` |

---

## 7. 参考资料

- 模型族映射：`src/aiconfigurator_npu/sdk/common.py:ARCHITECTURE_TO_MODEL_FAMILY`
- 模型实现：`src/aiconfigurator_npu/sdk/models.py:DeepSeekModel`
- MLA 表加载：`src/aiconfigurator_npu/sdk/perf_database.py:load_context_mla_data` / `load_generation_mla_data` / `load_mla_bmm_data`
- GLM-5 适配（同类工作参考）：`docs/GLM5_ADAPTATION_DESIGN.md`
