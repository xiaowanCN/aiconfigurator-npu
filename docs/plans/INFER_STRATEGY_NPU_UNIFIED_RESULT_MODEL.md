# aic-npu / msmodeling 统一结果模型设计（草案）

> 目标：把两套框架各自不同的返回结果，对齐成**一套同构的结果模型**，
> 让 HTTP 服务（以及 CLI 的 `run.json` / `summary.txt`）对外口径一致。
> 本文基于仓库内真实产物（`results/infer-strategy-npu/...`）的列名整理，非臆测。
> 分支：`feature/infer-strategy-npu-unified-cli`

---

## 1. 两侧真实字段对照

### 1.1 aic-npu（`aic_npu/agg|disagg/best_config_topn.csv`）

agg 列：
`model, isl, osl, prefix, concurrency, request_rate, bs, global_bs, ttft, tpot, request_latency,
seq/s, seq/s/gpu, tokens/s, tokens/s/gpu, tokens/s/user, num_total_gpus, tp, pp, dp, moe_tp, moe_ep,
parallel, gemm, kvcache, fmha, moe, comm, memory, balance_score, num_ctx_reqs, num_gen_reqs,
num_tokens, ctx_tokens, gen_tokens, backend, version, system, power_w, tokens/s/gpu_cluster`

disagg 列（一行即一个完整 P+D 方案）：
`concurrency, request_rate, (p)bs, (p)global_bs, (p)workers, (d)bs, (d)global_bs, (d)workers,
ttft, tpot, seq/s, seq/s/gpu, tokens/s, tokens/s/gpu, tokens/s/user,
(p)seq/s/worker, (d)seq/s/worker, num_total_gpus,
(p)tp,(p)pp,(p)dp,(p)moe_tp,(p)moe_ep,(p)parallel,(p)gemm,...(p)system,
(d)tp,(d)pp,(d)dp,(d)moe_tp,(d)moe_ep,(d)parallel,...(d)system, power_w, tokens/s/gpu_cluster`

### 1.2 msmodeling（`msmodeling/<xPyD>/{agg,disagg_prefill,disagg_decode}/summary.csv`）

`device_name, num_devices, model_id, quantize_linear_action, quantize_attention_action,
input_length, output_length, effective_input_length, max_batched_tokens, prefill_num_chunks,
[prefill_phase_makespan_ms], concurrency, ttft, tpot, token/s, token/s/device,
parallel, batch_size, percentage_breakdowns[(p)/(d)], weight_GB, kv_cache_GB, activation_GB,
avail_GB, tp_size, pp_size, dp_size, ep_size, moe_dp_size, pp_layer_partition, pp_schedule,
pp_makespan_ms, pp_warmup_ms, pp_steady_ms, pp_cooldown_ms, pp_bubble_ratio,
pp_bottleneck_stage, pp_mixed_pd_overlap_approx`

> `parallel` 实际取值示例：`TP=32 | PP=1 | DP=4 | EP=128 | MOE-TP=1 | MOE-DP=1`

### 1.3 映射表（现状 → 缺口）

| 统一字段 | aic-npu agg | aic-npu disagg | msmodeling | 现状 |
|---|---|---|---|---|
| 总输出吞吐 | `tokens/s` | `tokens/s` | `token/s` | ✅ 已映射 |
| 单卡输出吞吐 | `tokens/s/gpu` | `tokens/s/gpu` | `token/s/device` | ✅ 已映射 |
| 请求吞吐 req/s | `seq/s` | `seq/s` | **无同名列** | ⚠️ 靠 `token/s ÷ osl` 兜底 |
| 并发 | `concurrency` | `concurrency` | `concurrency` | ✅ |
| 批大小 | `bs` | `(p)bs` / `(d)bs` | `batch_size` | ✅ |
| TTFT / TPOT | `ttft` / `tpot` | 同 | `ttft` / `tpot` | ✅ |
| TP | `tp` | `(p)tp` / `(d)tp` | `tp_size` | ✅ |
| PP | `pp` | `(p)pp` / `(d)pp` | `pp_size` | ✅ |
| DP | `dp` | `(p)dp` / `(d)dp` | `dp_size` | ✅ |
| **EP** | `moe_ep` | `(p)moe_ep` / `(d)moe_ep` | **`ep_size`** / `EP=` | ✅ **已修复**（见 §5-1/5-2） |
| MoE-TP | `moe_tp` | `(p)moe_tp` | `MOE-TP=` | ✅ 已修复 |
| MoE-DP | 无 | 无 | `moe_dp_size` / `MOE-DP=` | ✅ 已修复（新增 `moe_dp`） |
| 部署卡数 | `num_total_gpus` | `num_total_gpus` | `num_devices` | ✅ |
| **节点数** | —（agg 为 1 组） | `(p)workers` / `(d)workers` | **无**（xPyD 的 x / y） | ⚠️ 口径不同 |
| 设备 | `system`=`ascend_910b` | `(p)system` / `(d)system` | `device_name`=`ATLAS_...` | ⚠️ 名字体系不同 |
| 模型 | `model` | `model` | `model_id`（本地路径） | ⚠️ 形式不同 |
| 后端 / 版本 | `backend`/`version` | `(p)/(d)backend`/`version` | 无 | ⚠️ msmodeling 为空 |
| 量化 | `gemm`/`kvcache`/`fmha`/`moe`/`comm` | `(p)...` | `quantize_linear_action` / `..._attention_action` | ⚠️ 表示法不同 |
| 功耗 | `power_w` | `power_w` | 无 | ⚠️ 缺失置 null |
| 显存 | `memory` | `(p)/(d)memory` | `weight_GB`+`kv_cache_GB`+`activation_GB`/`avail_GB` | ⚠️ 粒度不同 |
| 集群吞吐 | `tokens/s/gpu_cluster` | 同 | 无 | ⚠️ 缺失 |
| **总 prefill 吞吐** | 需派生 | `(p)seq/s/worker`×`(p)workers`×isl | 需派生 | ❌ **未计算** |
| **总吞吐** | 需派生 | 需派生 | 需派生 | ❌ **未计算** |

**缺口优先级（评审结论）**：

1. ✅ **EP（MoE 专家并行）—— 最重要，已修复**（见 §5）
2. ⏳ **节点数 / 每节点卡数** —— 统一为 `nodes + devices_per_node + replicas`，随统一模型落地
3. ⏳ **prefill 吞吐 / 总吞吐** —— 低优先级，**单卡输出吞吐（已有）即可满足**当前需求，其余后续再补

> 关于"历史结果是否受影响"：EP 数据**从未真正丢失** —— `summary.csv` 与 `run.json` 的 `raw`
> （`ep_size`、`moe_dp_size`）以及 `parallel_str` 一直都带 EP，丢的只是归一化后的
> 结构化 `parallel` 对象。因此**不需要重跑历史任务**，老结果可直接从 `raw` 取 EP。
> 本次修复只影响**后续运行**的结构化输出与将来的 HTTP 接口。

---

## 2. 派生指标的统一计算口径

统一在归一化层计算，保证两侧算法完全一致（缺失输入则为 `null`，**不用 0 代替**，避免歧义）：

```
requests_per_s        = seq/s 或 (token/s ÷ output_length)      # msmodeling 走后者
prefill_tokens_per_s  = requests_per_s × input_length           # 总 prefill 阶段吞吐
output_tokens_per_s   = tokens/s 或 token/s                     # 总输出吞吐
output_tokens_per_s_per_device = 上述 ÷ 该侧卡数                # 单卡输出吞吐
total_tokens_per_s    = prefill_tokens_per_s + output_tokens_per_s   # 总吞吐
devices_per_node      = num_devices ÷ nodes
replicas              = num_devices ÷ (tp × pp × dp)
```

> ⚠️ 待确认口径：msmodeling `disagg_prefill` 的 `token/s` 语义是否已经是"输入 token/s"。
> 若是，则 prefill 吞吐应直接取该值而非派生。当前统一按 `req/s × isl` 派生，
> 并在结果里用 `provenance` 标注来源，便于日后校正。

---

## 3. 统一结果模型（分层）

**结论：统一返回全量，但结构上强制分层** —— `core`（核心，扁平易渲染）/ `extended`（次要，前端折叠）/ `raw`（原始行，默认展示时忽略）。
这样前端"折叠次要信息"只需折叠 `extended`，不需要后端开裁剪开关；同时列表页可用 `?view=summary` 只取 `core`。

### 3.1 顶层

```jsonc
{
  "schema_version": "1.1",
  "task_id": "20261020-143012-ZhipuAI-GLM-5.3-Flash-ATLAS_800_A2_376T_64G-128-a1b2c3d4",
  "status": "ok",                       // ok | partial | failed
  "framework": "msmodeling",            // 实际下发的框架
  "context": {                          // 与方案无关的公共信息
    "model": "ZhipuAI/GLM-5.3-Flash",
    "model_source": "/workspace/model_configs/msmodeling/GLM5.3-Flash",
    "device": "ATLAS_800_A2_376T_64G",
    "num_devices_total": 128,
    "input_length": 8192, "output_length": 1024,
    "ttft_limit": 30000, "tpot_limit": 200,
    "quantize": "W8A8",
    "backend": "msmodeling", "backend_version": null
  },
  "agg":        PlanAgg,       // PD 混部（可能为 null）
  "disagg":     [ PlanDisagg, ... ],   // PD 分离：xPyD 每组一个方案
  "recommendation": { "serving_mode": "disagg", "plan_ref": "disagg[3]", "reason": "..." },
  "errors": []
}
```

### 3.2 `PlanAgg`（PD 混部）

```jsonc
{
  "label": "agg",
  "side": DeploymentSide,      // 复用同一个"部署侧"结构
  "candidates": [ DeploymentSide, ... ]   // Top-N，前端可折叠
}
```

### 3.3 `PlanDisagg`（PD 分离，一组 PD-ratio 组合一个）

```jsonc
{
  "label": "P8D8",                     // PD-ratio 组合标签；aic-npu 为 "disagg"
  "pd_ratio": { "prefill_devices": 8, "decode_devices": 8 },   // **每实例**卡数
  "prefill": DeploymentSide,           // P 侧信息
  "decode":  DeploymentSide,           // D 侧信息
  "overall": {                         // 整条 PD 方案的端到端口径
    "output_tokens_per_s": 12345.6,
    "output_tokens_per_s_per_device": 96.4,
    "prefill_tokens_per_s": 98765.4,
    "total_tokens_per_s": 111111.0,
    "requests_per_s": 12.05,
    "concurrency": 256,
    "ttft_ms": 320.5, "tpot_ms": 48.2,
    "meets_slo": true,
    "num_devices": 128
  }
}
```

### 3.4 `DeploymentSide`（核心：agg / prefill / decode 同构）

```jsonc
{
  "role": "prefill",                   // agg | prefill | decode
  "model": "ZhipuAI/GLM-5.3-Flash",
  "device": "ATLAS_800_A2_376T_64G",
  "backend": "msmodeling", "backend_version": null,

  // ---- core：吞吐 ----
  "throughput": {
    "output_tokens_per_s": 12345.6,             // 总输出吞吐
    "output_tokens_per_s_per_device": 96.4,     // 单卡输出吞吐
    "prefill_tokens_per_s": 98765.4,            // 总 prefill 阶段吞吐
    "total_tokens_per_s": 111111.0,             // 总吞吐
    "requests_per_s": 12.05,
    "tokens_per_s_per_user": null               // 仅 aic-npu 有
  },
  // ---- core：时延 ----
  "latency": { "ttft_ms": 320.5, "tpot_ms": 48.2,
               "request_latency_ms": null, "meets_slo": true },
  // ---- core：部署配置 ----
  "deploy": {
    "num_devices": 96,                 // 该侧总卡数
    "nodes": 3,                        // P/D 节点数（aic: workers；ms: x/y）
    "devices_per_node": 32,
    "replicas": 1,
    "parallel": { "tp": 32, "pp": 1, "dp": 4,
                  "moe_tp": 1, "moe_ep": 128, "moe_dp": 1 },
    "parallel_str": "TP=32 | PP=1 | DP=4 | EP=128 | MOE-TP=1 | MOE-DP=1",
    "batch_size": 32,
    "concurrency": 128
  },
  // ---- core：量化 ----
  "quantize": { "linear": "W8A8_DYNAMIC", "attention": "INT8",
                "gemm": null, "kvcache": null, "fmha": null, "moe": null, "comm": null },

  // ---- extended：次要信息，前端默认折叠 ----
  "extended": {
    "memory": { "weight_GB": 7.9, "kv_cache_GB": 0.54,
                "activation_GB": 6.98, "avail_GB": 38.57, "total_GB": null },
    "power_w": null,
    "cluster_tokens_per_s_per_gpu": null,      // aic: tokens/s/gpu_cluster
    "breakdown": { "Mem": 49.47, "Comm": 45.03, "Cube": 5.19, "Vec": 0.31 },
    "pipeline_parallel": { "schedule": "forward", "bubble_ratio": 0.0,
                           "makespan_ms": null, "bottleneck_stage": null },
    "prefill_detail": { "num_chunks": 1, "phase_makespan_ms": null },
    "balance_score": null,
    "source_files": [ ".../msmodeling/3P1D/disagg_prefill/summary.csv" ]
  },

  // ---- raw：原始行（不建议展示，排障用）----
  "raw": { "...": "原始 CSV 行的全部列" }
}
```

**要点：agg / prefill / decode 三种角色共用同一个 `DeploymentSide`**，前端只需一套渲染组件；
"P 节点信息 / D 节点信息"就是 `disagg[i].prefill` 与 `disagg[i].decode`。

---

## 4. 为什么"统一返回全量"而不是后端裁剪

| 方案 | 优点 | 缺点 |
|---|---|---|
| 后端只返回核心字段 | 体积小 | 前端要更多字段就得改后端；排障信息拿不到 |
| **统一返回全量 + 分层**（推荐） | 前端自由折叠、排障可用、后端不改 | 单次响应偏大 |
| 全量 + `?view=` 裁剪 | 兼顾列表页 | 多一个参数 |

推荐 **全量 + 分层**，并额外提供 `?view=summary`（只返回 `core`）与 `?include_raw=false`（去掉 `raw`）供列表/移动端使用。

体积估算：`raw` 是最大头（aic-npu 约 40 列 × N 候选）。8 组 xPyD × Top-N 候选的 run.json 约数百 KB，
gzip 后几十 KB，HTTP 可接受；真出问题再用 `include_raw=false`。

---

## 5. 现有代码需要的改造

| # | 改动 | 位置 |
|---|---|---|
| 1 | `parse_parallel` 支持 `EP=`、`MOE-TP=`、`MOE-DP=` | `runners/msmodeling.py` |
| 2 | `ParallelConfig` 增加 `moe_dp`；msmodeling 用 `ep_size` 填 `moe_ep`、`moe_dp_size` 填 `moe_dp`、`tp_size/pp_size/dp_size` 直接取数值列（比解析字符串更可靠） | `schema.py` + `runners/msmodeling.py` |
| 3 | 统一派生 `prefill_tokens_per_s` / `total_tokens_per_s` / `devices_per_node` / `replicas` | `schema.py`（新增工具函数） |
| 4 | `Candidate` 增加 `nodes` / `devices_per_node` / `replicas`；aic-npu 取 `workers`，msmodeling 取 xPyD 的 x/y | `schema.py` + 两个 runner |
| 5 | 新增 `to_unified_view()`：把 `FrameworkRun(agg, disagg, disagg_variants)` 转成 §3 的分层模型 | `schema.py` 或新增 `unified.py` |
| 6 | 量化口径统一：aic 的 `gemm/kvcache/fmha/moe/comm` 与 msmodeling 的 `linear/attention` 都写进 `quantize`，缺失置 null | 两个 runner |
| 7 | `summary.txt` 终端表格增加 EP 列与 prefill 吞吐列 | `output.py` |

> 第 5 步是"新增转换层"，**不改动 `Candidate` / `FrameworkRun` 的现有语义**，
> `run.json` 仍保留旧结构（向后兼容），新结构以 `unified` 字段并列输出，或放到 `data.result.unified`。

---

## 6. 评审结论与待确认

**已确认**：

- EP 必须修复 → **已完成**（§5-1/§5-2）
- 反馈结果**全量保存解析**，细粒度展示交给前端 → 保持 `raw` 全量落盘 + 结构化字段齐全
- 节点数统一为 `nodes + devices_per_node + replicas` → 随统一模型一起落地（§5-4）
- prefill / 总吞吐非当前重点，**单卡输出吞吐即可** → 暂缓

**待确认**：

1. msmodeling `disagg_prefill` 的 `token/s` 是否已是"输入 token/s"（若要算 prefill 吞吐时需明确）
2. aic-npu 的 `system=ascend_910b` 与 msmodeling 的 `device_name=ATLAS_...` 是否要统一成"设备显示名"（建议保留各自原值，顶层 `context.device` 用请求值）
3. `extended` 里是否还要加：aic-npu 的 `num_ctx_reqs`/`num_gen_reqs`/`balance_score`、msmodeling 的 `profiling_*` 列
4. `raw` 默认是否随响应返回（建议默认带，提供 `include_raw=false` 关闭）
5. xPyD 场景下 `disagg` 是数组（多组方案），前端是否需要"推荐组"高亮（已有 `recommendation.plan_ref`）
