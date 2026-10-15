# 三模型采集数据校验与推理策略推荐分析报告

- 生成日期：2026-10-15
- 采集脚本：`tools/collect_3models_from_scratch.sh`（依据 `docs/COLLECT_3MODELS_FROM_SCRATCH_RUNBOOK.md` / `docs/DATA_FROM_SCRATCH_3_MODELS.md`）
- 采集数据路径：`D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\staging-1014\1014\staging`
- 系统 / 后端 / 版本：`ascend_910b` / `vllm-ascend` / `0.23.0`
- 模型：Qwen3.8-27B、DeepSeek-R1、DeepSeek-V4-Pro

---

## 0. 摘要（结论先行）

1. **采集完整性**：脚本 17 个阶段全部 `OK`（0 失败，总耗时 289 分钟），产出 13 张干净表共约 5,700 行。
2. **落位**：已将 `staging/out` 的 13 张表按文档 §5 的 `merge_perf_tables.py`（按"除 latency 外的全部列"为 key）合并进
   `src/.../systems/data/ascend_910b/vllm-ascend/0.23.0`，并事先备份（见 §3）。其它模型（V3/GLM-5/Mixtral 等）的既有行被完整保留。
3. **就绪度**：`check_model_readiness` 对三模型均 **0 BLOCKED**（均为 DEGRADED，仅预期内 WARNING）。
4. **颗粒度结论**：**满足文档规定的采集需求**（所需表齐全、tp≤32 的 rank-local head 网格覆盖完整、V4 MoE 的 EP 全网格补齐）。
   唯一实质性缺口是 **R1 的 MoE 仅采到 `ep=8`**（该缺口为历史遗留，本次采集未修复，见 §4.3）。
5. **策略推荐**：
   - **SILICON 模式**：R1 **无法给出任何推荐**（缺 `ep>8` 的 MoE 实测数据）；V4、Qwen 正常出结果。
   - **HYBRID 模式（推荐用于这三个前沿模型）**：三模型全部出结果，且 R1 可用 `ep=64`（SILICON 缺失）给出可行部署。
   - **EMPIRICAL / SOL**：给出相同的并行策略形状，但延迟为理论/经验估值（偏高，作上界参考）。
   - Qwen 因数据完全覆盖，**四种模式结果逐字节一致**。

---

## 1. 采集数据与校验

### 1.1 流水线状态（来自 `collect_3models_log.txt` 收尾汇总）

| # | 阶段 | 结果 | 耗时 |
|---:|------|------|-----:|
| 1 | §4.8 gdn_perf (Qwen3.8-27B) | OK | 1562s |
| 2–5 | §4.5 dsv4 CSA/HCA context+generation (V4-Pro) | OK | 1302/236/1041/282s |
| 6 | §4.5 dsv4 四表转换 → staging/out | OK | 0s |
| 7 | §4.6 mhc_module_perf (V4-Pro) | OK | 191s |
| 8 | §4.3 MoE R1 (ep1..64) | OK | 35s |
| 9 | §4.3 MoE V4-Pro (ep1..64) | OK | 175s |
| 10 | §4.3 MoE 转换 → staging/out | OK | 0s |
| 11–13 | §4.4 context/generation_mla + mla_bmm (R1) | OK | 8088/120/33s |
| 14–15 | §4.7 attention head_dim=256 (Qwen3.8-27B) | OK | 2301/0s |
| 16–17 | §4.1 gemm (三模型共用) + 转换 | OK | 2024/0s |

> 失败阶段数：**0 / 17**。日志明确提示"本次只采集未落位"。

### 1.2 产出表清单与行数（`staging/out/`）

| 表文件 | 行数 | 关键网格（去重后） |
|--------|-----:|--------------------|
| `gemm_perf.txt` | 1215 | gemm_dtype=float16；M/N/K 全网格（N/K 含 2048/3072/5120/7168/8192/12288/16384/17408） |
| `context_attention_perf.txt` | 432 | num_heads={3,6,12,24}，head_dim={256}，attn_dtype=float16 |
| `generation_attention_perf.txt` | 432 | num_heads={3,6,12,24}，head_dim={256} |
| `context_mla_perf.txt` | 430 | num_heads={4,8,16,32,64,128} |
| `generation_mla_perf.txt` | 486 | num_heads={4,8,16,32,64,128} |
| `mla_bmm_perf.txt` | 192 | num_heads={4,8,16,32,64,128} |
| `moe_perf.txt` | 210 | topk={6,8}，num_experts={256,384}，hidden=7168，inter={2048,3072}，moe_ep_size={2,4,8,16,32,64}，distribution={power_law_1.01,1.2} |
| `dsv4_csa_context_module_perf.txt` | 261 | num_heads={4,8,16,32,64,128} |
| `dsv4_hca_context_module_perf.txt` | 272 | num_heads={4,8,16,32,64,128} |
| `dsv4_csa_generation_module_perf.txt` | 367 | num_heads={4,8,16,32,64,128} |
| `dsv4_hca_generation_module_perf.txt` | 454 | num_heads={4,8,16,32,64,128} |
| `mhc_module_perf.txt` | 32 | hc_mult={4} |
| `gdn_perf.txt` | 957 | Qwen 新表（GDN 4 个 kernel） |

### 1.3 逐模型就绪度（`check_model_readiness.py`，落位后复核）

| 模型 | family | 状态 | 关键 WARNING（均为预期内） |
|------|--------|------|---------------------------|
| DeepSeek-R1 | DEEPSEEK | DEGRADED | ① MLA `tp=64` 不可定价（需 rank-local heads=2，Ascend 不支持）；② **MoE 形状 `ep=8` 仅此一档**；③ custom_allreduce/nccl 仅覆盖 8 卡，>8 卡外推 |
| DeepSeek-V4-Pro | DEEPSEEKV4 | DEGRADED | ① dsv4 CSA/HCA `tp=64` 不可定价；② custom_allreduce/nccl 仅覆盖 8 卡 |
| Qwen3.8-27B | QWEN35 | DEGRADED | ① `tp=16/32/64` 因不能整除 `num_heads=24` 被拒（设计上限 tp=8）；② custom_allreduce/nccl 仅覆盖 8 卡 |

**结论：三模型均 0 BLOCKED**，所有"缺表/缺几何"的硬阻塞项均不存在。

---

## 2. 颗粒度校验（是否满足需求）

依据 `docs/DATA_FROM_SCRATCH_3_MODELS.md` §4 与 `check_model_readiness.FAMILY_REQUIREMENTS`：

| 模型 | 必需表 | 采集覆盖 | 判定 |
|------|--------|----------|------|
| R1 (DEEPSEEK) | gemm, mla, mla_gen, mla_bmm, moe | 全部存在；MLA/MLA_bmm head 网格 {4,8,16,32,64,128} 覆盖 tp 1–32（tp=64 受 Ascend 限制，文档已知） | ✅ 满足（MoE 见下） |
| V4 (DEEPSEEKV4) | gemm, dsv4_csa, dsv4_hca, dsv4_gen, mhc, moe | 全部存在；dsv4 head 网格覆盖 tp 1–32；**MoE EP 全网格 {2,4,8,16,32,64}** | ✅ 满足 |
| Qwen (QWEN35) | gemm, attention, attention_gen, gdn | 全部存在；dense attention head_dim=256 且 num_heads {3,6,12,24} 覆盖 tp 1–8；gdn 957 行 | ✅ 满足 |

### 2.1 颗粒度亮点（相对旧 `0.23.0` 的改善）

- **V4 MoE EP 网格被补齐**：旧表 V4 形状（384 专家 / topk6 / inter3072）仅 `ep={16,32,64}` 三档；本次采集扩展到 **`ep={2,4,8,16,32,64}` 六档**，使 V4 在 128 卡上的 EP 部署搜索空间完整。
- **Qwen 两张新表落地**：`gdn_perf.txt`（957 行，新采集器）与 `head_dim=256` 的 attention 表，使 Qwen 从"缺表阻塞"变为可定价。
- 合并以 key 覆盖方式进行，**其它模型（V3/GLM-5/Mixtral/Qwen2·3-MoE 等）的同几何行被保留、非本次几何行完整不动**，无数据丢失（见 §3 落位验证）。

### 2.2 颗粒度缺口（需关注）

1. **R1 MoE 仅 `ep=8` 一档（实质性缺口）**
   - 证据：`staging/out/moe_perf.txt` 中 R1 形状（num_experts=256, topk=8, hidden=7168, inter=2048）仅 `moe_ep_size=8`；旧 `0.23.0` 中该形状同样仅有 `ep=8` —— **属历史遗留，本次采集未修复**。
   - 影响：SILICON 模式下，128 卡部署搜索会尝试 `ep∈{16,32,64}` 等档位，因缺实测而**整组失败**；只有回落到 `ep=8`（即 16 副本 × 8 卡 worker）才可能定价，且 `check_model_readiness` 报 `viable_worker_widths=[8]`。
   - 根因线索：R1 的 `MoE R1 (ep1..64)` 阶段仅耗时 35s（V4 为 175s），说明 `-ep2/-ep4/-ep16/-ep32/-ep64` 变体对 R1 形状**未产出有效行**（可能 `collect_moe.py` 的 `MODEL_CONFIGS` 对 R1 的 EP 变体在单卡模拟下未生效或跳过）。
   - **建议**：重采 R1 MoE，确保 `deepseek-v3-ep2/ep4/ep16/ep32/ep64` 对 (256,8,7168,2048) 形状各产出一行（与 V4 对齐到 6 档 EP）。

2. **`tp=64` 注意力不可采（已知限制，非缺陷）**
   - Ascend 稀疏注意力要求 `q_head_num` 为 4 的倍数，`heads=2`（即 tp=64 时的 rank-local heads）无法采集。R1/V4 的 `tp=64` 配置只能回落 SOL。文档 §8.5 已记录，属预期。

3. **多机通信表 `custom_allreduce_perf.txt` / `nccl` 仅覆盖 8 卡（设计排除项）**
   - 本次按需求**未采集** §4.2（文档 §1 明确"按需求排除"）。>8 卡集合通信走外推，对 128 卡部署的通信估算为近似。如需更精确的多机通信，可后续单独补采 16/32/64 卡（文档 §8.7）。

> **总体判定：采集颗粒度满足文档 §4 规定的需求**（必需表齐全、tp≤32 几何完整、V4 EP 已补齐）。唯一需补采的是 **R1 MoE 的 EP 多档**（历史遗留，影响 SILICON 模式 128 卡定价）。

---

## 3. 落位（landing）说明

为用新数据驱动 `aic-npu default` 并做就绪度复核，按文档 §5 执行了合并落位：

- **备份**：`0.23.0` → `0.23.0.bak.20261015_093410`（位于 `src/.../systems/data/ascend_910b/vllm-ascend/` 下），可随时回滚。
- **合并工具**：`tools/merge_perf_tables.py <existing> <new> --in-place`（13 张表逐一合并）。
- **合并结果（落位后行数）**：gemm 3105、moe 301、context_mla 453、generation_mla 486、mla_bmm 192、dsv4_csa_ctx 262、dsv4_csa_gen 416、dsv4_hca_ctx 273、dsv4_hca_gen 510、mhc 32、context_attention 672、generation_attention 1643、**gdn 957（新增）**。
- 落位后其它模型的行数均 ≥ 旧值，证明既有数据未被破坏。

> 注：`custom_allreduce_perf.txt` 未改动（本次按需求未采集 §4.2）。

---

## 4. 推理策略推荐（`aic-npu default` 实测）

命令（仅 `--model-path` 与 `--database-mode` 变化）：

```bash
aic-npu default --model-path <model_config> --total-gpus 128 --system ascend_910b \
  --backend vllm-ascend --backend-version 0.23.0 --database-mode <MODE> \
  --isl 2048 --osl 8192 --ttft 30000 --tpot 3000 --top-n 10 --save-dir results
```

`--database-mode` 可选值：`SILICON`（默认，纯实测）、`HYBRID`（实测缺失处回落 SOL+经验，推荐用于前沿模型）、`EMPIRICAL`（SOL+经验因子）、`SOL`（纯理论光速上界）。`SOL_FULL` 被排除。

### 4.1 各模型 × 各模式结果概览

| 模型 | SILICON | HYBRID | EMPIRICAL | SOL |
|------|---------|--------|-----------|-----|
| **R1** | ❌ **无结果**（MoE 缺 `ep>8` 实测） | ✅ 出结果（ep64） | ✅ 出结果 | ✅ 出结果 |
| **V4** | ✅ 出结果（top 策略同 HYBRID） | ✅ 出结果（候选更多） | ✅ 出结果 | ✅ 出结果 |
| **Qwen** | ✅ 出结果 | ✅ 与 SILICON **逐字节一致** | ✅ 一致 | ✅ 一致 |

> Qwen 因数据完全覆盖，四种模式结果完全相同；V4 SILICON 与 HYBRID 的 **top-1 策略一致**（HYBRID 仅多放出若干因插值缺口在 SILICON 下被过滤的候选）；R1 则只有 SILICON 彻底失败。

### 4.2 R1 —— 仅 HYBRID/EMPIRICAL/SOL 可用（SILICON 失败）

**SILICON**（失败原文）：
```
MoE silicon data has no row for this expert shape in moe_perf.txt:
num_experts=256, topk=8, hidden_size=7168, inter_size=2048, moe_ep_size=128 ...
Recollect moe_perf.txt covering this (num_experts x ep) grid, or evaluate this
parallel strategy in HYBRID/EMPIRICAL mode.
No successful experiment runs to compare.
```

**HYBRID（推荐）Top 配置**：

- 聚合（agg）最优：`tp4 pp1 dp16 ep64`（2 副本 × 64 卡），**83.98 tok/s/gpu**，TTFT≈21.5s，bs=40
- 分离（disagg）最优：prefill `tp16 pp1 dp2 ep32` + decode `tp1 pp1 dp64 ep64`，**97.68 tok/s/gpu**，TTFT≈758ms，req/s≈1.53

> 注意：HYBRID 下 R1 的主力策略使用了 **`ep=64`**，这恰恰是 SILICON 缺失、只能靠 SOL+经验估值补齐的档位——这正是"为何 SILICON 失败、HYBRID 可用"的直接原因。

### 4.3 V4 —— SILICON 即可用，HYBRID/EMPIRICAL/SOL 一致

**SILICON / HYBRID Top 配置（一致）**：

- 聚合（agg）最优：`tp4 pp1 dp16 ep64`（2 副本 × 64 卡），**43.76 tok/s/gpu**，TTFT≈29.9s，bs=48
- 分离（disagg）最优：prefill `tp2 pp1 dp32 ep64` + decode `tp1 pp1 dp64 ep64`，**52.34 tok/s/gpu**，TTFT≈25.6s

V4 因 MoE EP 六档齐全，SILICON 已能完整定价，HYBRID 未改变推荐形状（仅多放出候选）。

### 4.4 Qwen —— 四模式完全一致

**SILICON / HYBRID / EMPIRICAL / SOL Top 配置（完全相同）**：

- 聚合（agg）最优：`tp4 pp1`（32 副本 × 4 卡），**529.66 tok/s/gpu**，TTFT≈2.8s，bs=64
- 分离（disagg）最优：prefill `tp1`（2 worker）+ decode `tp2`，**321.25 tok/s/gpu**，TTFT≈1.27s

Qwen 为 27B 稠密+GDN 混合模型（无 MoE 瓶颈），单卡吞吐远高于 R1/V4，且数据完全覆盖，故四种模式无差异。

### 4.5 横向对比（HYBRID / 可用模式下的最优吞吐）

| 模型 | 最优 agg tok/s/gpu | 最优 disagg tok/s/gpu | 主要约束 |
|------|------------------:|---------------------:|----------|
| Qwen3.8-27B | **529.66** | 321.25 | 稠密+GDN，tp≤8，无 MoE |
| R1 (HYBRID) | 83.98 | **97.68** | MoE；SILICON 不可用 |
| V4 (SILICON) | 43.76 | 52.34 | MoE；EP 全网格已补齐 |

> R1/V4 的 tok/s/gpu 显著低于 Qwen，主要因 671B/类似量级 + MoE 专家并行通信开销；二者在 128 卡上均以 `ep=64` 的两副本 × 64 卡为最优部署形态。

---

## 5. 行动建议

1. **必做（修复 R1 SILICON 可用性）**：重采 R1 MoE，补齐 (256,8,7168,2048) 形状的 `ep∈{2,4,16,32,64}`，使 R1 在 SILICON 模式下也能对 128 卡定价。重采后再次 `merge` 落位（`LANDING=1` 或手动 `merge_perf_tables.py`）。
2. **可选（提升大集群精度）**：补采 16/32/64 卡 `custom_allreduce_perf.txt` / `nccl`，减少 >8 卡通信的外推误差。
3. **推荐用法**：对这三个前沿模型跑推荐时默认用 `--database-mode HYBRID`；若已确认硅基数据齐全（如 V4/Qwen）可用 `SILICON` 获得完全可复现结果；`EMPIRICAL`/`SOL` 仅作理论上界参考。
4. **回滚**：如需撤销本次落位，`mv 0.23.0.bak.20261015_093410 0.23.0` 即可。

---

## 附录 A：本次执行的关键产物位置

- 采集数据：`staging-1014/1014/staging/`（含 `out/` 13 张干净表、`collect_3models_log.txt`）
- 校验/落位脚本与本报告同目录：`staging-1014/1014/`
  - `inspect_granularity.py`（颗粒度与就绪度校验）
  - `land_merge.py`（备份 + 合并落位）
  - `run_all.py`（批量跑 `aic-npu default`）
  - `parse_results.py`（从日志提取干净结果表）
  - `runs/clean/*.txt`（11 份 Top 配置表，按 `<模型>_<模式>.txt` 命名）
- 落位目标：`aiconfigurator-npu/src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0`（已备份 `0.23.0.bak.20261015_093410`）
