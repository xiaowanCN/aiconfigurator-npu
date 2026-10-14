# 三模型性能数据「从头全量采集」手册（DeepSeek-R1 / DeepSeek-V4-Pro / Qwen3.8-27B）

创建日期：2026-10-13

本文档是 `docs/DeepSeek-R1_ADAPTATION.md`、`docs/DeepSeek-V4-Pro_ADAPTATION.md`、
`docs/Qwen3.8-27B_ADAPTATION.md`、`docs/DATA_COLLECTION_PLAN.md` 的**整合执行版**。
目标：把上述三个模型在 `ascend_910b / vllm-ascend / 0.23.0` 上所需的全部性能数据**一次性重新采集、
干净落位**，并规避过去"反复 `--append`"带来的脏数据问题。

---

## 0. 为什么要从头采、并用 merge 而不是 append

过去给 `0.23.0/` 补数据时，大量步骤用 `convert_to_aiconfigurator.py --append` 或 `cp` 追加，
踩过以下坑（见各适配文档的 F 缺陷表）：

1. **`--append` 不幂等**：同一批 CSV 跑两次 → `moe_perf.txt` 里同一个 key 出现多份，
   `load_moe_data` 只保留先到的那条、仅打 debug 日志，表膨胀且无法肉眼核对（`F12`）。
2. **整表 `cp` 会抹掉其它模型行**：`moe_perf.txt` 是 V3 / R1 / V3.2 / GLM-5 / MiniMax / Mixtral /
   Qwen2·3-MoE 等多个模型**共用**的一张表。`convert_moe` 默认以 `"w"` 覆盖写、又只能看到
   `--input-dir` 里本次放的 CSV，直接 `cp` 过去会把其它模型的行全部丢掉（`F10`）。
3. **旧/失效数据混入**：V4 早期 `dsv4_*` 数据里混了 profiling 兜底值、`ratio=128` 冒充 CSA、
   `heads=2` 在 Ascend 上 tiling 失败的行，重采时若只 append 不替换，脏值一直留在表里。
4. **难以审计**：表是按几何（`hidden / heads / ep / …`）keyed 的，**不按模型名**，反复增量追加后，
   根本分不清某一行属于哪个模型、是哪一次采集的。

**本手册的方法**：每个算子把本次要采的几何**独立采到 staging 目录**，转成"只含本次模型行"的干净表文件，
最后用 §3 的 **merge 脚本**（按"除 `latency` 外的全部列"做 key，新数据覆盖旧数据、非本次模型的行原样保留）
并回 `0.23.0/`。这样：

- 三个模型需要的几何被**刷新为本次真实测量值**；
- 其它模型（V3、V3.2、GLM-5、Mixtral、Qwen2·3-MoE…）的同几何行被新值覆盖（几何一致，等价），
  非本次几何的行**完整保留**（不会被误删）；
- 可重复执行且幂等（重跑 merge 结果不变），无需"`git checkout` 后再 append"。

---

## 1. 采集范围与表清单（每个模型读哪些表）

由 `tools/check_model_readiness.py` 的 `FAMILY_REQUIREMENTS` 决定：

| 模型 | 族 | 读取的表 |
|------|----|----------|
| **DeepSeek-R1** | `DEEPSEEK` | `gemm` · `moe` · `context_mla` · `generation_mla` · `mla_bmm` · `custom_allreduce` |
| **DeepSeek-V4-Pro** | `DEEPSEEKV4` | `gemm` · `moe` · `dsv4_csa_context` · `dsv4_csa_generation` · `dsv4_hca_context` · `dsv4_hca_generation` · `mhc` · `custom_allreduce` |
| **Qwen3.8-27B** | `QWEN35` | `gemm` · `context_attention`(head_dim=256) · `generation_attention`(head_dim=256) · `gdn` · `custom_allreduce` |

按"是否被多个目标模型共用"分类：

| 表 | 共用情况 | 落位方式 |
|----|----------|----------|
| `gemm_perf.txt` | 三模型共用（且被更多模型共用） | merge |
| `custom_allreduce_perf.txt` | 三模型共用 | merge（但本次几何已在默认 sweep 内，可只校验） |
| `moe_perf.txt` | R1 + V4（且被更多模型共用） | merge |
| `context_attention_perf.txt` / `generation_attention_perf.txt` | Qwen3.8-27B（且被更多 dense 模型共用） | merge |
| `context_mla_perf.txt` / `generation_mla_perf.txt` | R1（且 V3 也用） | merge |
| `mla_bmm_perf.txt` | R1（且 V3 也用） | merge |
| `dsv4_csa_*_module_perf.txt` / `dsv4_hca_*_module_perf.txt` / `mhc_module_perf.txt` | 仅 V4-Pro | merge（无历史则等同新建） |
| `gdn_perf.txt` | 仅 Qwen3.8-27B（**当前 0.23.0 目录缺失，需新采集器**） | merge（无历史则等同新建） |

> 结论：**不存在"专属于某一个目标模型、可以放心整表覆盖"的共享表**。
> 一律走 merge，避免误删其它模型的数据。

---

## 2. 工作目录与 staging 布局

在**仓库根目录**操作（NPU 机器上）。所有采集产物放到 `./staging/` 下，与仓库数据目录彻底隔离，
落位前可随意重采、重转，互不污染：

```bash
# 公共环境变量（vendor 自定义算子包必须在加载路径上）

REPO=$(pwd)
DATA_DIR=$REPO/src/aiconfigurator_npu/systems/data/ascend_910b/vllm-ascend/0.23.0
STG=$REPO/staging


export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$REPO/collector
```

`staging/` 规划：

```
staging/
├── gemm/            collect_gemm.py 输出（CSV）
├── hccl/            collect_hccl.py 输出（直接出 *_perf.txt）
├── moe_r1/          R1 MoE（CSV）
├── moe_v4/          V4 MoE（CSV）
├── mla_r1_ctx/      R1 MLA context（直接出 *_perf.txt）
├── mla_r1_gen/      R1 MLA generation（直接出 *_perf.txt）
├── mla_bmm_r1/      R1 mla_bmm（直接出 *_perf.txt）
├── dsv4_csa_ctx/    V4 CSA context（直接出 *_perf.txt）
├── dsv4_csa_gen/    V4 CSA generation（直接出 *_perf.txt）
├── dsv4_hca_ctx/    V4 HCA context（直接出 *_perf.txt）
├── dsv4_hca_gen/    V4 HCA generation（直接出 *_perf.txt）
├── mhc/             V4 mHC（直接出 *_perf.txt）
├── attn_hd256/      Qwen dense attention head_dim=256（CSV）
└── gdn/             Qwen GDN（直接出 gdn_perf.txt）
```

---

## 3. 落位方法：MERGE，不是 APPEND（核心）

把下面脚本存成 `tools/merge_perf_tables.py`（一次性，之后所有落位都用它）。
它读取"已有的 `0.23.0/<table>.txt`"和"本次 staging 产出的干净 `<table>.txt`"，
按**除 `latency` 外的全部列**做 key：**本次新行覆盖旧行；旧表里非本次几何的行原样保留**。
幂等、可重复执行。

```python
#!/usr/bin/env python3
"""按 key 合并两张 perf 表：new 覆盖 existing 的同 key 行，existing 的其它行保留。

用法:
    python tools/merge_perf_tables.py <existing.txt> <new.txt> [--in-place]
若 --in-place，则直接写回 existing；否则打印到 stdout（重定向落位）。
existing 不存在时，等同于把 new 搬过去。
"""
import argparse
import csv
import io
import sys


def _latency_col(header):
    for i, c in enumerate(header):
        if c.strip().lower() == "latency":
            return i
    return None


def load(path):
    if not path or not __import__("os").path.isfile(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def merge(existing_path, new_path):
    e_head, e_rows = load(existing_path)
    n_head, n_rows = load(new_path)
    head = n_head or e_head
    lat = _latency_col(head)
    assert lat is not None, f"no 'latency' column in header: {head}"

    def key(row):
        return tuple(v for i, v in enumerate(row) if i != lat)

    # existing 行建索引，new 覆盖
    merged = {key(r): r for r in e_rows}
    new_keys = set()
    for r in n_rows:
        k = key(r)
        merged[k] = r
        new_keys.add(k)
    # 保留 existing 顺序；仅追加 truly-new 的 key
    out = []
    seen = set()
    for r in e_rows:
        k = key(r)
        if k in seen:
            continue
        out.append(merged.pop(k, r))
        seen.add(k)
    for k, r in merged.items():
        if k not in seen:
            out.append(r)
            seen.add(k)
    return head, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("existing")
    ap.add_argument("new")
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    head, out = merge(args.existing, args.new)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    w.writerows(out)
    text = buf.getvalue()
    if args.in_place:
        with open(args.existing, "w", newline="", encoding="utf-8") as f:
            f.write(text)
        print(f"merged -> {args.existing} ({len(out)} rows)", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
```

落位时（所有表都这样，区别只在 staging 产出哪张 txt）：

```bash
python tools/merge_perf_tables.py \
  "$DATA_DIR/<table>.txt" "$STG/out/<table>.txt" --in-place
```

> 落位前建议先 `cp -r $DATA_DIR /tmp/0.23.0.bak` 备份，便于回滚。

---

## 4. 采集命令（三模型全集，已含历次修复）

> 所有命令在 NPU 机器、仓库根目录执行。DSv4 / GDN 需要 §2 的 `ASCEND_CUSTOM_OPP_PATH` /
> `LD_LIBRARY_PATH`；其余也需要正确的 vllm-ascend 0.23.0 镜像环境。
> 每个采集器对单点失败（`try/except`）只记 error 继续，可放心跑满网格，跑完看缺哪些点再补。

### 4.1 `gemm_perf.txt`（三模型共用，共享表 → 走 merge）

三模型需要的 (K, N) 维度：R1/V3 系 `7168↔2048`、V4 `7168↔3072`、Qwen `5120↔17408`
（注意默认 `DEFAULT_NK_LIST` **不含 5120 / 3072 / 17408**，必须显式传）：

```bash
cd "$REPO" && \
python collector/npu/collect_gemm.py \
  --quant-types bf16 \
  --m-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  --n-list 2048 3072 4096 5120 7168 8192 12288 16384 17408 \
  --k-list 2048 3072 4096 5120 7168 8192 12288 16384 17408 \
  --output-dir "$STG/gemm" --timing-mode graph --resume
```

转成干净表（只含本次维度），再 merge：

```bash
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/gemm" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0
# → $STG/out/gemm_perf.txt
```

### 4.2 `custom_allreduce_perf.txt`（三模型共用）

默认 `BIZ_HIDDENS=[4096,5120,6144,7168]` 已覆盖三模型 hidden；单节点 8 卡即可：

```bash
cd "$REPO" && \
torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29501 \
  collector/npu/collect_hccl.py --ops all_reduce all_gather reduce_scatter \
  --output-dir "$STG/hccl"
# → $STG/hccl/nccl_perf.txt 与 custom_allreduce_perf.txt（直接出 txt）
cp "$STG/hccl/custom_allreduce_perf.txt" "$STG/out/custom_allreduce_perf.txt"
```

> 多机 16/32/64 卡通信表当前不是三模型的硬阻塞项（通信按 worker 内 TP/EP 计），可后续补；
> 若要采，`--nnodes` 调大即可，产物同样 `cp` 到 `$STG/out/` 再 merge。

### 4.3 `moe_perf.txt`（R1 + V4，共享表 → merge）

**R1**：`collect_moe.py` 的 `MODEL_CONFIGS` 需已含 `deepseek-v3` 及 `deepseek-v3-ep{1,2,4,16,32,64}`
（见 `DATA_COLLECTION_PLAN.md` §3.1，缺失则先补）。R1 查 `power_law_1.01`：

```bash
cd "$REPO" && \
python collector/npu/collect_moe.py \
  --models deepseek-v3 deepseek-v3-ep2 deepseek-v3-ep4 deepseek-v3-ep16 \
          deepseek-v3-ep32 deepseek-v3-ep64 \
  --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  --quant-types bf16 --output-dir "$STG/moe_r1" --timing-mode graph --resume
```

**V4**：`deepseek-v4-pro` 及 `deepseek-v4-pro-ep{2,4,16,32,64}`（`ep=64` 是唯一可行部署档，必采）：

```bash
cd "$REPO" && \
python collector/npu/collect_moe.py \
  --models deepseek-v4-pro deepseek-v4-pro-ep2 deepseek-v4-pro-ep4 \
          deepseek-v4-pro-ep16 deepseek-v4-pro-ep32 deepseek-v4-pro-ep64 \
  --token-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \
  --quant-types bf16 --output-dir "$STG/moe_v4" --timing-mode graph --resume
```

转成干净表（`--moe-distribution both` 同时写 `power_law_1.01` 与 `power_law_1.2`，通吃所有族）：

```bash
# R1 先建表
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/moe_r1" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --moe-distribution both
# V4 追加到同一份 staging（不要 --append 到 0.23.0，只追加到 staging/out）
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/moe_v4" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --moe-distribution both --append
# → $STG/out/moe_perf.txt（含 R1+V4 全部几何）
```

### 4.4 `context_mla_perf.txt` / `generation_mla_perf.txt` / `mla_bmm_perf.txt`（R1，V3 也用 → merge）

R1 真实维度：`128 heads / kv_lora_rank 512 / qk_nope 128 / qk_rope 64 / v_head 128`。
`num_heads` 必须覆盖 `128 // tp_size`：`tp≤8` → `{128,64,32,16}`，`tp=16/32` → **`{8,4}`**（Ascend
`SparseAttnSharedkv` 要求 `q_head_num` 为 4 的倍数，`heads=2` 不可采）。

```bash
cd "$REPO" && \
python collector/npu/collect_mla.py --output-format mla \
  --architecture DeepseekV3ForCausalLM --model deepseek-ai/DeepSeek-R1 \
  --op-types context \
  --batch-list 1 2 4 8 16 32 64 128 256 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 32768 \
  --num-heads-list 128 64 32 16 8 4 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --mla-dtype float16 --kv-cache-dtype float16 \
  --output-dir "$STG/mla_r1_ctx" --timing-mode graph --resume

python collector/npu/collect_mla.py --output-format mla \
  --architecture DeepseekV3ForCausalLM --model deepseek-ai/DeepSeek-R1 \
  --op-types generation \
  --batch-list 1 2 4 8 16 32 64 128 256 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 32768 \
  --num-heads-list 128 64 32 16 8 4 \
  --kv-lora-rank 512 --qk-nope-head-dim 128 --qk-rope-head-dim 64 --v-head-dim 128 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --mla-dtype float16 --kv-cache-dtype float16 \
  --output-dir "$STG/mla_r1_gen" --timing-mode graph --resume
# → $STG/mla_r1_ctx/context_mla_perf.txt、 $STG/mla_r1_gen/generation_mla_perf.txt（直接出 txt）
cp "$STG/mla_r1_ctx/context_mla_perf.txt"      "$STG/out/context_mla_perf.txt"
cp "$STG/mla_r1_gen/generation_mla_perf.txt"   "$STG/out/generation_mla_perf.txt"
```

```bash
cd "$REPO" && \
python collector/npu/collect_mla_bmm.py \
  --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 \
  --num-heads-list 128 64 32 16 8 4 \
  --kv-lora-rank 512 --head-dim 128 \
  --op-types pre post \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --bmm-dtype float16 --output-dir "$STG/mla_bmm_r1" --timing-mode graph --resume
# → $STG/mla_bmm_r1/mla_bmm_perf.txt
cp "$STG/mla_bmm_r1/mla_bmm_perf.txt" "$STG/out/mla_bmm_perf.txt"
```

### 4.5 DSv4 CSA / HCA（V4-Pro，仅 V4 用 → merge）

四个组合（csa/hca × context/generation）**各用一个 `--output-dir`**（checkpoint 文件名共享，同目录会互相覆盖）：

```bash
cd "$REPO" && \
python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds csa \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/dsv4_csa_ctx" --timing-mode graph --resume

python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_generation --attn-kinds csa \
  --batch-list 1 2 4 8 16 32 64 128 256 384 512 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/dsv4_csa_gen" --timing-mode graph --resume

python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_context --attn-kinds hca \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/dsv4_hca_ctx" --timing-mode graph --resume

python collector/npu/collect_dsv4_attn.py \
  --op-types dsv4_generation --attn-kinds hca \
  --batch-list 1 2 4 8 16 32 64 128 256 384 512 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --num-heads-list 128 64 32 16 8 4 \
  --module-source framework \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/dsv4_hca_gen" --timing-mode graph --resume
```

转成干净表并落到 staging/out（用转换器"复制"这四个 txt，不污染 0.23.0）：

```bash
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/dsv4_csa_ctx" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --dsv4-output
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/dsv4_csa_gen" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --dsv4-output
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/dsv4_hca_ctx" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --dsv4-output
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/dsv4_hca_gen" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0 \
  --dsv4-output
# → $STG/out/dsv4_{csa,hca}_{context,generation}_module_perf.txt
```

> 每个 `num_heads` 下 `(s, b)` 至少各 2 个取值，否则 `_interp_2d_1d` 三次插值会
> `QhullError` 把整张表废掉（`F11`）。上面的网格已满足。

### 4.6 `mhc_module_perf.txt`（V4-Pro，仅 V4 用 → merge）

```bash
cd "$REPO" && \
python collector/npu/collect_mhc.py \
  --op-types pre post \
  --num-tokens-list 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  --hc-mult 4 --hidden-size 7168 --sinkhorn-iters 20 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/mhc" --timing-mode graph --resume
# → $STG/mhc/mhc_module_perf.txt
cp "$STG/mhc/mhc_module_perf.txt" "$STG/out/mhc_module_perf.txt"
```

### 4.7 dense attention `head_dim=256`（Qwen3.8-27B，共享表 → merge）

Qwen3.8-27B：`num_attention_heads=24`、`num_key_value_heads=4`、`head_dim=256`。
rank-local `num_heads = 24 // tp ∈ {24,12,6,3}`，`num_kv_heads = ceil(4/tp) ∈ {4,2,1}`。
（这是 27B 精确需要的头数；更宽的列表是给其它 Qwen 模型用的，这里只采 27B 需要的，其余靠 merge 保留。）

```bash
cd "$REPO/collector/npu" && \
python collect_attn.py \
  --op-types context generation \
  --head-size 256 \
  --num-heads-list 24 12 6 3 \
  --num-kv-heads-list 4 2 1 \
  --batch-list 1 2 4 8 16 32 \
  --seq-len-list 128 256 512 1024 2048 4096 8192 16384 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/attn_hd256" --timing-mode graph --resume
```

转成干净表（两个 CSV 分别转 context / generation）：

```bash
cd "$REPO" && \
python tools/convert_to_aiconfigurator.py \
  --input-dir "$STG/attn_hd256" --output-dir "$STG/out" \
  --device "Ascend 910B" --framework vllm-ascend --version 0.23.0
# → $STG/out/context_attention_perf.txt、 generation_attention_perf.txt
```

> ⚠️ 先冒烟确认 910B FlashInfer attention 支持 `head_dim=256`（部分 FIA 实现上限 128/192）。
> 冒烟：`collect_attn.py --head-size 256 --num-heads-list 24 --num-kv-heads-list 4 --seq-len-list 4096 --timing-mode graph`。

### 4.8 `gdn_perf.txt`（Qwen3.8-27B，仅 Qwen 用 → merge；**新采集器**）

GDN 4 个 kernel 的逻辑名与 vllm-ascend 0.23.0 真实入口的映射，以及 per-tp 本地头数的必要性，
见 `docs/Qwen3.8-27B_ADAPTATION.md` §4.0–§4.1。`collect_gdn.py` 直出 `gdn_perf.txt`（列与
`load_gdn_data` 一致，无需转换）。

```bash
cd "$REPO/collector/npu" && \
ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer \
LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python collect_gdn.py \
  --model-name qwen3_8_27B \
  --d-model 5120 --d-conv 4 \
  --num-k-heads 16 --head-k-dim 128 --num-v-heads 48 --head-v-dim 128 \
  --tp-sizes 1 2 4 8 \
  --max-context-tokens 1000000 \
  --framework vllm-ascend --version 0.23.0 --device "Ascend 910B" \
  --output-dir "$STG/gdn" --timing-mode graph --resume
# → $STG/gdn/gdn_perf.txt（直接出 txt）
cp "$STG/gdn/gdn_perf.txt" "$STG/out/gdn_perf.txt"
```

> 冒烟（2 个点）：`collect_gdn.py --tp-sizes 1 --warmup-iters 1 --bench-iters 3 --timing-mode graph --output-dir $STG/gdn_smoke`。
> 自检：延迟应随 `seq_len` / `batch_size` / `(nk,nv)` 单调变化；若不同头数的值几乎相同，说明测到的是
> 主机开销而非 kernel 时间（见 `Qwen3.8-27B_ADAPTATION.md` §4.0）。

---

## 5. 落位（统一走 §3 的 merge）

先备份，再把 `$STG/out/` 里每张干净表 merge 进 `$DATA_DIR/`：

```bash
cp -r "$DATA_DIR" /tmp/0.23.0.bak.$(date +%s)

for t in gemm_perf custom_allreduce_perf moe_perf \
         context_mla_perf generation_mla_perf mla_bmm_perf \
         dsv4_csa_context_module_perf dsv4_csa_generation_module_perf \
         dsv4_hca_context_module_perf dsv4_hca_generation_module_perf \
         mhc_module_perf context_attention_perf generation_attention_perf gdn_perf; do
  src="$STG/out/$t.txt"
  [ -f "$src" ] || { echo "skip (no staging): $t"; continue; }
  python tools/merge_perf_tables.py "$DATA_DIR/$t.txt" "$src" --in-place
done
```

落位后 14 张表应齐全（`0.23.0/` 现有 13 张 + 新增 `gdn_perf.txt`）。

---

## 6. 落位后校验（必做）

```bash
# 三个模型各自的 SILICON 探针（无需跑完整 default）
python tools/check_model_readiness.py \
  model_configs/deepseek-ai--DeepSeek-R1_config.json \
  model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json \
  model_configs/qwen3_8_27B_config.json \
  --system ascend_910b --backend vllm-ascend --version 0.23.0
# 期望：三个模型都 READY（无 BLOCKING）；Qwen 不再报 "requires gdn_perf.txt" 与 head_dim=256 缺

# V4 专属的网格/退化/重复行诊断
python tools/check_dsv4_data.py --isl 4096 --osl 1024
# 期望：exit 0，所有探针命中
```

---

## 7. 推荐执行顺序与时间预算

| 顺序 | 算子 / 表 | 模型 | 说明 | 硬件门槛 |
|---:|-----------|------|------|----------|
| 1 | `gdn_perf.txt` | Qwen | **新采集器，唯一尚未落地的表**，先验证 kernel 可获得性 | 高（需确认 vllm-ascend 0.23.0 自定义算子） |
| 2 | `dsv4_csa/hca_*` + `mhc` | V4 | four + one 个采集器，已踩坑固定；DSv4 framework 路径强依赖 vllm-ascend 内部布局 | 中 |
| 3 | `moe_perf.txt` | R1 + V4 | ep 网格最宽（R1 需 ep1…64），单卡模拟 EP，无需多机 | 中 |
| 4 | `context/generation_mla` + `mla_bmm` | R1 | heads 补到 {8,4} 解锁 tp16/32 | 低 |
| 5 | `context/generation_attention` head_dim=256 | Qwen | 解锁一半阻塞项 | 低（先验证 FIA 支持 256） |
| 6 | `gemm_perf.txt` | 三模型 | 维度最杂，最后统一采 + merge | 低 |
| 7 | `custom_allreduce_perf.txt` | 三模型 | 默认 sweep 已含 5120/7168，仅校验 | 低（8 卡） |

---

## 8. 已知限制与注意事项

1. **merge 的 key 是"除 `latency` 外的全部列"**：若两个不同采集批次在几何上完全一致但测量值不同，
   新值覆盖旧值——这正是期望的"刷新"。但若某几何本次**没采到**（如 OOM 失败），旧值会被保留，不会留洞。
2. **不要对共享表用 `cp` 整表覆盖**，也不要对 `moe_perf.txt` 直接 `--append` 到 `0.23.0`**——
   这正是过去丢数据的根因。始终先产干净 staging 表，再走 §3 merge。
3. **GDN 采集器的 kernel 入口需在 NPU 上首跑验证**：`npu_causal_conv1d_custom` /
   `chunk_gated_delta_rule` / `npu_recurrent_gated_delta_rule` 必须来自 vllm-ascend 0.23.0 的
   custom-op 包（§4.8 的环境变量），缺失会报 `... not in libopapi.so`。
4. **DSv4 framework 采集是 TP=1 单模块近似**，且不含量化（`float16`）；与 SDK 建模口径一致。
   升级 vllm-ascend 后需按 `DeepSeek-V4-Pro_ADAPTATION.md` §4.0 的症状表回归。
5. **`heads=2` / `tp=64` 在 Ascend 上不可采**（稀疏注意力 kernel 要求 `q_head_num` 为 4 倍数）；
   所有 attention 类采集列表均不含 `2`，对应 `tp=64` 配置只能回落 SOL（V4 规模分析见其适配文档 §8.0）。
6. **MoE 并行约束**：vllm-ascend 不支持同时按 TP 和 EP 切 MoE（`moe_tp_list` 固定 `[1]`），
   `moe_ep` 必须由 EP 承担，故 ep 网格要宽。
7. **多机通信 16/32/64 卡**不是三模型 SILICON 的硬阻塞项（按 worker 内 TP/EP 计），可单独排期补充。

---

## 9. 参考资料

- 模型适配：`docs/DeepSeek-R1_ADAPTATION.md`、`docs/DeepSeek-V4-Pro_ADAPTATION.md`、`docs/Qwen3.8-27B_ADAPTATION.md`
- 补采计划：`docs/DATA_COLLECTION_PLAN.md`
- 采集器：`collector/npu/{collect_gemm,collect_hccl,collect_moe,collect_mla,collect_mla_bmm,collect_dsv4_attn,collect_mhc,collect_attn,collect_gdn}.py`
- 转换器：`tools/convert_to_aiconfigurator.py`
- 校验：`tools/check_model_readiness.py`、`tools/check_dsv4_data.py`
- 落位合并（本文新增）：`tools/merge_perf_tables.py`
