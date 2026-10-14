# 三模型全量采集脚本运行说明（`tools/collect_3models_from_scratch.sh`）

创建日期：2026-10-14
配套文档：`docs/DATA_FROM_SCRATCH_3_MODELS.md`（采集需求与几何口径的原始出处）

本说明只讲**怎么跑** `tools/collect_3models_from_scratch.sh`。该脚本把
`DATA_FROM_SCRATCH_3_MODELS.md` §4 中除 §4.2 `custom_allreduce_perf.txt` 之外的
**全部采集点**串成一条流水线：统一 staging、统一日志、逐阶段进度、失败不中断、可续采。

---

## 1. 脚本覆盖的采集点（共 17 个阶段）

| # | 阶段名 | 对应文档 | 模型 | 产出 |
|---:|--------|----------|------|------|
| 1 | `gdn_perf` | §4.8 | Qwen3.8-27B | `gdn_perf.txt` |
| 2 | `dsv4 CSA context` | §4.5 | V4-Pro | `dsv4_csa_context_module_perf.txt` |
| 3 | `dsv4 CSA generation` | §4.5 | V4-Pro | `dsv4_csa_generation_module_perf.txt` |
| 4 | `dsv4 HCA context` | §4.5 | V4-Pro | `dsv4_hca_context_module_perf.txt` |
| 5 | `dsv4 HCA generation` | §4.5 | V4-Pro | `dsv4_hca_generation_module_perf.txt` |
| 6 | `dsv4 四表转换 -> staging/out` | §4.5 | V4-Pro | 上面 4 张表 |
| 7 | `mhc_module_perf` | §4.6 | V4-Pro | `mhc_module_perf.txt` |
| 8 | `MoE R1 (ep1..64)` | §4.3 | R1 | CSV |
| 9 | `MoE V4-Pro (ep1..64)` | §4.3 | V4-Pro | CSV |
| 10 | `MoE 转换 -> staging/out` | §4.3 | R1+V4 | `moe_perf.txt`（both 分布） |
| 11 | `context_mla` | §4.4 | R1 | `context_mla_perf.txt` |
| 12 | `generation_mla` | §4.4 | R1 | `generation_mla_perf.txt` |
| 13 | `mla_bmm` | §4.4 | R1 | `mla_bmm_perf.txt` |
| 14 | `attention head_dim=256` | §4.7 | Qwen3.8-27B | CSV |
| 15 | `attention 转换 -> staging/out` | §4.7 | Qwen3.8-27B | `context/generation_attention_perf.txt` |
| 16 | `gemm` | §4.1 | 三模型共用 | CSV |
| 17 | `gemm 转换 -> staging/out` | §4.1 | 三模型共用 | `gemm_perf.txt` |

可选阶段：`SMOKE=1` 时在最前面加 2 个冒烟阶段（GDN 2 点、attention hd256 1 点）；
`LANDING=1` 时在最后面加 2 个阶段（§5 落位 merge、§6 校验）。

**不包含**：§4.2 `custom_allreduce_perf.txt`（三模型共用，按需求排除；落位清单中同样已去掉）。

顺序按文档 §7「先难后易」：GDN → DSv4 → mHC → MoE → MLA → attention → GEMM。

---

## 2. 前置条件

- NPU 机器、vllm-ascend `0.23.0` 镜像环境，仓库已拉取完整（含 `collector/`、`tools/`）。
- vendor 自定义算子包路径存在（DSv4 / GDN 依赖）：
  `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer`
  路径不同用 `OPP_PATH=...` 覆盖。
- 脚本会自行 `cd` 到仓库根目录（由 `BASH_SOURCE` 推算），在任意目录执行均可。
- 首次执行建议先备份落位目录（脚本只在 `LANDING=1` 时写 `0.23.0/`，且会自行再备份一次）。

---

## 3. 运行命令

### 3.1 常规全量采集（推荐）

```bash
cd /path/to/aiconfigurator-npu
chmod +x tools/collect_3models_from_scratch.sh

# 后台跑
nohup bash tools/collect_3models_from_scratch.sh &

# 另开终端实时看进度
tail -f staging/collect_3models_log.txt
```

### 3.2 前台跑（脚本内部已 tee）

```bash
bash tools/collect_3models_from_scratch.sh
```

断连风险高的机器套一层 tmux/screen：

```bash
tmux new -s collect 'bash tools/collect_3models_from_scratch.sh'
# Ctrl-B D 脱离；tmux attach -t collect 重新挂回
```

### 3.3 先冒烟再全采（强烈建议首次执行）

```bash
SMOKE=1 bash tools/collect_3models_from_scratch.sh
```

确认两件事后再跑全量：

1. GDN：延迟随 `seq_len / batch_size / (nk,nv)` 单调变化（否则测到的是主机开销）；
2. attention：910B FIA 支持 `head_dim=256`（部分实现上限 128/192）。

### 3.4 只跑 / 跳过部分阶段

`ONLY` / `SKIP` 对**阶段名**做子串匹配，逗号分隔，忽略大小写：

```bash
ONLY="moe,mla" bash tools/collect_3models_from_scratch.sh   # 只跑 MoE + MLA 相关阶段
ONLY="gdn"     bash tools/collect_3models_from_scratch.sh   # 只跑 GDN
SKIP="gemm"    bash tools/collect_3models_from_scratch.sh   # 跳过 GEMM 两阶段
```

### 3.5 只看计划不执行

```bash
DRY_RUN=1 bash tools/collect_3models_from_scratch.sh
```

### 3.6 采集 + 落位 + 校验（一条龙）

```bash
LANDING=1 bash tools/collect_3models_from_scratch.sh
```

会先 `cp -r` 备份 `0.23.0/` 到 `/tmp/0.23.0.bak.<ts>`，再用
`tools/merge_perf_tables.py --in-place` 按 key 覆盖落位（**不是 append**），
最后跑 `check_model_readiness.py` 与 `check_dsv4_data.py`。

> 也可以先只采集、人工核对 `staging/out/*.txt` 之后再单独落位：
> `LANDING=1 ONLY="落位 merge" bash tools/collect_3models_from_scratch.sh`

---

## 4. 日志与进度

- **整体日志**：`staging/collect_3models_log.txt`（可用 `LOG=...` 覆盖）。
  所有阶段的输出都写这**一个**文件，阶段之间用分隔条 + `[i/N]` 标记切分。
  重跑默认清空重建；`LOG_APPEND=1` 追加到旧日志。
- **边跑边看**：每条命令 `cmd 2>&1 | tee -a "$LOG"`，同时 `PYTHONUNBUFFERED=1` + `python -u`，
  因此 `tail -f` 看到的是实时进度，不是攒批输出。
- **阶段进度日志**（示例）：

```text
==============================================================================
[4/17] 2026-10-14 11:20:31 >>> 当前阶段：§4.5 dsv4 HCA context (V4-Pro)
  已完成 3 / 共 17；剩余 13 个待采集：§4.5 dsv4 HCA generation (V4-Pro) | §4.5 dsv4 四表转换 -> staging/out | ...
  已耗时 42 分钟
------------------------------------------------------------------------------
$ python -u collector/npu/collect_dsv4_attn.py --op-types dsv4_context --attn-kinds hca ...
[4/17] <<< 完成：§4.5 dsv4 HCA context (V4-Pro)（耗时 631s）
```

- **收尾汇总**：逐阶段 `OK / FAILED rc=...` 与耗时、失败阶段数、`staging/out/*.txt` 行数清单。
  有任一阶段失败则脚本 exit 1。

---

## 5. 续采与重跑

- 所有采集器都带 `--resume`，中断后**原样重跑同一条命令**即可从 checkpoint 续采。
- 只想补某个算子：用 `ONLY="关键字"` 单独跑对应阶段。
- staging 与 `0.23.0/` 完全隔离，重采不会污染线上数据；只有 `LANDING=1` 才写 `0.23.0/`。

---

## 6. 目录布局

```
staging/
├── collect_3models_log.txt      # 整体日志（唯一）
├── out/                         # 干净表，落位输入
│   ├── gemm_perf.txt  moe_perf.txt  gdn_perf.txt
│   ├── context_mla_perf.txt  generation_mla_perf.txt  mla_bmm_perf.txt
│   ├── dsv4_{csa,hca}_{context,generation}_module_perf.txt
│   ├── mhc_module_perf.txt
│   └── context_attention_perf.txt  generation_attention_perf.txt
├── gemm/ moe_r1/ moe_v4/ mla_r1_ctx/ mla_r1_gen/ mla_bmm_r1/
├── dsv4_csa_ctx/ dsv4_csa_gen/ dsv4_hca_ctx/ dsv4_hca_gen/ mhc/
├── attn_hd256/ gdn/
└── gdn_smoke/ attn_hd256_smoke/   # 仅 SMOKE=1 时产生
```

---

## 7. 可配环境变量

| 变量 | 含义 | 默认值 |
|------|------|--------|
| `STG` | staging 根目录 | `$REPO/staging` |
| `LOG` | 整体日志文件 | `$STG/collect_3models_log.txt` |
| `LOG_APPEND` | `1` 追加到旧日志 | `0`（每次新建） |
| `DATA_DIR` | 落位目标目录 | `src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0` |
| `OPP_PATH` | vendor 自定义算子包 | `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer` |
| `PYTHON` | python 解释器 | `python` |
| `SMOKE` | `1` 先跑冒烟 | `0` |
| `LANDING` | `1` 采集后落位+校验 | `0` |
| `ONLY` / `SKIP` | 阶段名子串过滤 | 空 |
| `DRY_RUN` | `1` 只打印计划 | `0` |

---

## 8. 注意事项

1. **不要用 `cp` 整表覆盖共享表、不要对 `moe_perf.txt` 直接 `--append` 到 `0.23.0`**——
   这是历史丢数据的根因。落位一律走 `tools/merge_perf_tables.py`（脚本内已如此实现）。
2. **merge 的 key 是「除 `latency` 外的全部列」**：本次没采到的几何会保留旧值，不会留洞；
   采到的几何刷新为新值，非本次模型的其它几何行完整保留。
3. **`heads=2` / `tp=64` 在 Ascend 上不可采**（稀疏注意力要求 `q_head_num` 为 4 的倍数），
   脚本里 attention 类列表均为 `128 64 32 16 8 4`。
4. **单阶段失败不会中断整条流水线**（采集器内部对单点也有 `try/except`），
   跑完看汇总里哪些阶段 `FAILED`，修完再用 `ONLY=` 补采即可。
5. 脚本为 LF 换行。若经 Windows 编辑过，先执行
   `sed -i 's/\r$//' tools/collect_3models_from_scratch.sh`，否则 bash 会报 `^M: bad interpreter`。
6. 如需多机 16/32/64 卡通信表或 §4.2 `custom_allreduce_perf.txt`，按文档 §4.2 单独执行
   `torchrun ... collector/npu/collect_hccl.py`，产物 `cp` 到 `staging/out/` 后再走同样的 merge。
