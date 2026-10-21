# 推理策略寻优工具使用指南

---

## 1 简介

### 1.1 工具介绍

`infer-strategy-npu` 是一款面向大模型推理部署的策略寻优工具：它以内置的**算子级性能模型**为基础，通过对 GEMM、Attention、MoE 与集合通信等核心算子在目标硬件上的微基准刻画，构建可外推的性能数据库，并据此在给定模型、卡数与 SLA（TTFT / TPOT）约束下，对**张量并行、流水并行、数据并行、专家并行**的组合、批大小、以及 **PD 混部 / PD 分离** 两种服务形态做统一组合搜索，**一条命令即可同时给出两种部署形态的全部候选配置与各自最优解**，让用户在不上机试错的前提下完成部署选型与容量规划。

工具会依据模型、设备与量化开关自动完成全部搜索与评估，**用户无需关心内部细节**，也无需手动指定额外的执行命令。

### 1.2 模型适配矩阵

| 模型                            | 架构特征                         | 量化支持             | PD 混部 | PD 分离 | 适配状态 |
| ----------------------------- | ---------------------------- | ---------------- | ----- | ----- | ---- |
| `ZhipuAI/GLM-5.3`             | GLM MoE + DSA                | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `ZhipuAI/GLM-5.2`             | GLM MoE + DSA                | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `ZhipuAI/GLM-5.3-Flash`       | GLM MoE + DSA                | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `Qwen/Qwen3.8-27B`            | Qwen3.5（Hybrid Attention）    | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `deepseek-ai/DeepSeek-R1`     | DeepSeek-V3（MLA + MoE）       | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek-V4（CSA/HCA + 大规模专家） | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `moonshotai/Kimi-K3`          | MoE                          | DISABLE / ENABLE | 支持    | 支持    | 已适配  |
| `MiniMax/MiniMax-M3`          | MoE                          | DISABLE / ENABLE | 支持    | 支持    | 已适配  |

> **说明：** `infer-strategy-npu --list-models` 打印的是实时注册表；上表为当前版本矩阵。若某模型的本地配置尚未替换为真实配置，寻优结果可能不可用，建议先以 `--dry-run` 校验，`ENABLE` 条件下量化模式支持：`FP8, MXFP4, W8A8_DYNAMIC`。


---

## 2 主要场景

本工具的核心特点：**一条命令同时产出 PD 混部与 PD 分离的全部候选配置，并为每种部署方式各自给出最优解**。

### 2.1 一次寻优：PD 混部 + PD 分离全量配置

**适用场景：** 首次评估 GLM-5.3 在给定卡数下的部署方案，希望一次性看到两种服务形态的全部可行配置与各自 Top-N。

**16 卡（不量化）：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize DISABLE
```

**32 卡（不量化）：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 32 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize DISABLE
```

**约束：**

- TTFT / TPOT 上限为硬约束，不满足的配置不会进入候选；
- 若某形态无可行配置，该形态标记为不可用，不影响另一形态出结果；
- 单次运行耗时随卡数与搜索空间增长，32 卡通常在数十分钟量级；

### 2.2 量化模型寻优（ENABLE + 三个 action 参数）

**适用场景：** 显存受限或希望评估量化后的吞吐收益。GLM-5.3 开启量化（`ENABLE`）时，可显式指定线性层、KV Cache 量化动作；不指定则取工具默认。

**16 卡 ENABLE（默认动作）：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize ENABLE
```

**32 卡 ENABLE（开启 MXFP4、KV Cache FP8）：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 32 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize ENABLE \
  --quantize-linear-action MXFP4 --quantize-attention-action FP8
```


### 2.2.1 PD 分离配比搜索（xPyD）

**适用场景：** 已确定走 PD 分离，需要规划 Prefill 与 Decode 的实例数量与卡数配比。`--pd-balance-search` 默认开启。

**32 卡，限定 1P1D：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 32 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize ENABLE --quantize-linear-action MXFP4 --quantize-attention-action FP8 \
  --pd-balance-sizes 1P1D
```

> 说明：`--pd-balance-sizes 1P1D` 表示 1 个 Prefill 实例 + 1 个 Decode 实例，合计 32 卡 → 各 16 卡。若使用默认 `--pd-balance-search`（不指定 `--pd-balance-sizes`），工具会按 `--pd-balance-max`（默认 3）展开 `1P1D / 1P2D / 2P1D / ...` 等组合，每组分别评估 Prefill 与 Decode 两侧，耗时随组数线性增长。

---

## 3 结果说明

### 3.1 终端输出（汇总）

命令执行成功后，终端依次打印：

1. **输入配置**：模型、设备、卡数、ISL/OSL、SLA 上限、量化；
2. **各形态 Top 配置表**：PD 混部（agg）与 PD 分离（disagg）各一张；

下表为汇总表结构示意：

```text
  PD 混部 (agg):
  +-----+------+----------+--------------+----------+----------+--------+-------+------------------+
  | Top | Mode | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc. |     Parallel     |
  +-----+------+----------+--------------+----------+----------+--------+-------+------------------+
  |  1  | agg  |  ......  |    ......    |  ......  |  ......  | ...... |  ...  | tp*pp*...... |
  |  2  | agg  |  ......  |    ......    |  ......  |  ......  | ...... |  ...  | tp*pp*...... |
  +-----+------+----------+--------------+----------+----------+--------+-------+------------------+

  PD 分离 (disagg):
  +-----+----------------+----------+--------------+----------+----------+--------+-------+----------------------------------------------------+
  | Top |      Mode      | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc. |                      Parallel                      |
  +-----+----------------+----------+--------------+----------+----------+--------+-------+----------------------------------------------------+
  |  1  | disagg_prefill |  ......  |    ......    |  ......  |    -     | ...... |  ...  |                  tp*pp*......                  |
  |  1  | disagg_decode  |  ......  |    ......    |    -     |  ......  | ...... |  ...  |                  tp*pp*......                  |
  +-----+----------------+----------+--------------+----------+----------+--------+-------+----------------------------------------------------+
```

重点关注：

| 字段                      | 含义                                                                    |
| ----------------------- | --------------------------------------------------------------------- |
| `Top`                   | 该形态内的排名（按 `tokens/s/dev` 降序）                                          |
| `Mode`                  | `agg` / `disagg_prefill` / `disagg_decode`         |
| `tokens/s`              | 单副本吞吐                                |
| `tokens/s/dev`          | 单卡吞吐，跨配置比较的主指标                                                        |
| `TTFT(ms)` / `TPOT(ms)` | 首 token 时间 / 每输出 token 时间；阶段表中不适用的指标显示 `-`                            |
| `req/s`                 | 每秒请求数                                                                 |
| `Conc.`                 | 并发请求数                                                                 |
| `Parallel`              | 并行策略（`tp/pp/dp/ep`） |

> **说明：** PD 混部的 `tokens/s` 是**单副本**吞吐，副本数越多单副本值越小但集群总量越大；跨配置比较请看 `tokens/s/dev`，或按 6.1 换算集群总吞吐。`tp4pp1dp4` 表示单副本 4 卡（16 卡即 4 个副本）。


---

## 4 参数

### 4.1 必填参数

| 参数              | 说明                                  | 示例                   |
| --------------- | ----------------------------------- | -------------------- |
| `--model`       | ModelScope 模型 ID（见 `--list-models`） | `ZhipuAI/GLM-5.3`    |
| `--device`      | 统一设备名（见 `--list-devices`）           | `ATLAS_950_486T_96G` |
| `--num-devices` | NPU 卡数（正整数）                         | `16` / `32`          |

### 4.2 常用可选参数

| 参数                | 默认值                            | 说明                                                      |
| ----------------- | ------------------------------ | ------------------------------------------------------- |
| `--input-length`  | `1024`                         | 输入序列长度 ISL                                              |
| `--output-length` | `1024`                         | 输出序列长度 OSL                                              |
| `--tpot-limit`    | `50`                           | TPOT 上限（ms），SLA 硬约束                                     |
| `--ttft-limit`    | `1000`                         | TTFT 上限（ms），SLA 硬约束                                     |
| `--quantize`      | `DISABLE`                      | 量化总开关：`DISABLE` / `ENABLE`（旧版 `W8A8`/`FP8` 为 ENABLE 别名） |
| `--top-n`         | `10`                           | 每类返回的候选数量                                               |
| `--out-dir`       | `./results/infer-strategy-npu` | 结果输出根目录                                                 |

### 4.3 量化高级参数

| 参数                                    | 说明                                                                           | 取值                                                                                                                                                   |
| ------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--quantize-linear-action`            | 性层量化动作（ENABLE 生效）                                                         | `DISABLED` / `W8A8_DYNAMIC` / `FP8` / `MXFP4`（默认 `W8A8_DYNAMIC`） |
| `--quantize-non-expert-linear-action` | 非专家线性层量化动作（attention 投影 / dense MLP / shared expert；ENABLE 生效，默认 `DISABLED`） | 同 `--quantize-linear-action`                                                                                                                         |
| `--quantize-attention-action`         | KV Cache 量化动作（ENABLE 生效）                                                     | `DISABLED` / `INT8` / `FP8`（默认 `DISABLED`）                                                                                                           |

> **说明：** `DISABLE` 模式下上述三个 action 强制为 `DISABLED`；`ENABLE` 模式下显式传入优先，缺省取工具默认。

### 4.4 搜索与部署高级参数

| 参数                                                                        | 说明                                          |
| ------------------------------------------------------------------------- | ------------------------------------------- |
| `--pd-balance-search` / `--no-pd-balance-search`                          | xPyD 配比搜索开关（默认开启）                           |
| `--pd-balance-max`                                                        | xPyD 搜索的 P/D 节点数上限（默认 3）                    |
| `--pd-balance-sizes`                                                      | 显式指定 xPyD 组合，如 `1P1D 2P1D`                  |
| `--tp-sizes`                                                              | 显式指定 TP 候选（空格分隔；缺省按卡数自动推导）   |
| `--jobs`                                                                  | 内部计算线程并发度                                       |
| `--local-model-root`                                                      | 本地模型根目录                                     |
| `--speculative-method`                                                    | 投机解码方法：`none` / `mtp` / `dflash` / `dspark` |
| `--num-speculative-tokens` / `--acceptance-length` / `--num-draft-layers` | 投机解码细项                                      |

> **说明：** 完整参数以 `infer-strategy-npu -recommend --help` 为准。

### 4.5 辅助命令

| 命令                                  | 说明        |
| ----------------------------------- | --------- |
| `infer-strategy-npu --list-models`  | 打印支持的模型清单 |
| `infer-strategy-npu --list-devices` | 打印支持的设备清单 |
| `infer-strategy-npu --version`      | 打印版本      |


---

## 5 实验测试验证

本章以 `Qwen/Qwen3.8-27B`、8 卡 910B 机器、ISL/OSL = 128/128、SLO（TTFT ≤ 1000 ms、TPOT ≤ 50 ms）为例，分别给出**本项目**与**华为昇腾工具MsModeling**的预测结果，再用真机实测校验二者的准确率。

### 5.1 验证环境

| 项         | 取值                            |
| --------- | ----------------------------- |
| 模型        | `Qwen/Qwen3.8-27B`            |
| 设备        | `ATLAS_800_A2_376T_64G`（910B） |
| 卡数        | 8                             |
| 部署形态      | PD 混部（agg）                    |
| 输入 / 输出长度 | 128 / 128                     |
| SLO       | TTFT ≤ 1000 ms，TPOT ≤ 50 ms   |
| 量化        | `DISABLE`                     |

### 5.2 本项目预测

**命令：**

```bash
infer-strategy-npu -recommend \
  --model Qwen/Qwen3.8-27B \
  --device ATLAS_800_A2_376T_64G --num-devices 8 \
  --input-length 128 --output-length 128 \
  --ttft-limit 1000 --tpot-limit 50 \
  --quantize DISABLE
```

**PD 混部 Top-4：**

| Top | tokens/s | tokens/s/dev | TTFT (ms) | TPOT (ms) | req/s  | 并发  | 并行配置               |
| --- | -------- | ------------ | --------- | --------- | ------ | --- | ------------------ |
| 1   | 747.18   | 747.18       | 914.14    | 43.94     | 5.883  | 40  | `tp1pp1dp1etp1ep1` |
| 2   | 1172.81  | 586.40       | 875.72    | 46.10     | 9.235  | 64  | `tp2pp1dp1etp1ep1` |
| 3   | 2177.07  | 544.27       | 947.41    | 49.63     | 17.142 | 128 | `tp4pp1dp1etp1ep1` |
| 4   | 4259.26  | 532.41       | 974.31    | 46.62     | 33.537 | 240 | `tp8pp1dp1etp1ep1` |


> **说明：** Top 按 `tokens/s/dev`（单卡吞吐）降序；`tokens/s` 为该配置的**单副本吞吐**，跨配置比较应看 `tokens/s/dev`。

### 5.3 对标华为昇腾工具预测

**命令：**

```bash
python -m cli.inference.throughput_optimizer /workspace/model_configs/msmodeling/Qwen3.8-27B \
  --device ATLAS_800_A2_376T_64G \
  --num-devices 8 \
  --input-length 128 --output-length 128 \
  --quantize-linear-action disabled \
  --quantize-attention-action disabled \
  --jobs 32 \
  --tpot-limit 50 --ttft-limit 1000
```

**PD 混部 Top-3：**

| Top | 吞吐 (token/s) | TTFT (ms) | TPOT (ms) | 并发  | 卡数  | 并行配置               | batch size |
| --- | ------------ | --------- | --------- | --- | --- | ------------------ | ---------- |
| 1   | 4559.89      | 992.04    | 45.18     | 206 | 8   | TP=4 / PP=1 / DP=2 | 103        |
| 2   | 4004.39      | 998.63    | 38.21     | 153 | 8   | TP=8 / PP=1 / DP=1 | 153        |
| 3   | 3225.32      | 669.99    | 49.61     | 160 | 8   | TP=2 / PP=1 / DP=4 | 40         |

### 5.4 实测结果与准确率对比

**实测方法：** 在 910B 上按候选配置真实部署，记录端到端输出吞吐并据此重排。

**实测结果：** 最优配置为 `tp1pp1`（8 副本），排名为 **tp1 > tp2 > tp4 > tp8**。

| 对比项      | 真机实测          | 本项目预测                   | 昇腾工具预测             |
| -------- | ------------- | ----------------------- | ------------------ |
| Top-1 配置 | `tp1pp1`      | **`tp1pp1`**（一致）        | `TP=4/DP=2`（实测第 3） |
| 完整排名     | 1 > 2 > 4 > 8 | **1 > 2 > 4 > 8**（完全一致） | 4 > 8 > 2（完全错位）    |
| Spearman | —             | **1.00**                | 负相关                |

**结论：**

- 本项目在 ISL/OSL = 128 工况下，**Top-1 命中、Top-4 完整排名与实测完全一致**；
- 昇腾工具 Top-1 预测为 `TP=4 / DP=2`，实测仅列第 3，Top-3 排名完全错位；
- 该工况下的**排名准确率**显著更高；绝对指标的偏差见 5.4.1。

#### 5.4.1 实测明细（tp1pp1，ISL/OSL = 128/128）


| 指标           | 真机实测（tp1pp1） | 本项目预测（tp1pp1）    |
| ------------ | ------------ | ---------------- |
| TTFT (ms)    | 1442.67      | 914.14（-36.6%）   |
| TPOT (ms)    | 85.41        | 43.94（-48.6%）    |
| 集群吞吐 (tok/s) | 2133.74      | 5977.44（+180.1%） |
| 并发           | 32           | 40               |

> **口径说明：** 实测吞吐为单实例 `output_throughput`（266.72 tok/s），按 8 副本线性外推得 2133.74；本项目为单副本 747.18 × 8 = 5977.44。

**误差解读：**

- **配置选型准确**：本项目 Top-1 命中实测最优 `tp1pp1`（与昇腾工具的排名对比见 5.4）；
- **本项目绝对误差**：TTFT 偏低 36.6%、TPOT 偏低 48.6%、集群吞吐偏高 180.1%；
- 偏差来源：实测为短时压测（32 条请求、15.36 s），未达稳态，含排队与冷启动开销；预测模型对解码速度的估计偏乐观；
- 与 5.5 全网格统计一致（TPOT 系统性偏低、吞吐偏乐观）。

> **使用含义：** 适合用本工具的**排名**做并行策略选型；吞吐 / 延迟的**绝对值**不宜直接作为容量承诺，建议结合实测校准系数后使用。

### 5.5 工具测评准确性说明

除上述工况外，在更广的网格（ISL/OSL 128~4096、TP 1/2/4/8，16 条对齐样本）上测试本工具：

| 指标           | 结果              |
| ------------ | --------------- |
| Top-1 一致     | 11 / 16 = 68.8% |
| 完整排名一致       | 10 / 16 = 62.5% |
| Spearman 平均  | 0.738           |
| TTFT MAPE | 75.3%（偏差方向均值=-2718.7 ms，预测偏低）          |
| TPOT MAPE | 55.61%（偏差方向均值=-86.29 ms，预测偏低）     |
| 吞吐 MAPE | 265.0%（偏差方向均值=+558.4 tok/s，预测偏高） |


---

## 6 GLM-5.3 运行实例与结果查看


### 6.1 32 卡 · ENABLE(MXFP4/FP8) · 正式运行

```bash
infer-strategy-npu -recommend  \
  --model ZhipuAI/GLM-5.3  \
  --device ATLAS_950_486T_96G --num-devices 32  \
  --input-length 8196 --output-length 1024  \
  --tpot-limit 200 --ttft-limit 30000  \
  --quantize ENABLE --pd-balance-search  \
  --quantize-linear-action MXFP4 --quantize-attention-action FP8  \
  --tp-sizes 2 4 8 16 

```

```text
❯ infer-strategy-npu -recommend --model ZhipuAI/GLM-5.3 --device ATLAS_950_486T_96G --num-devices 32 --input-length 8196 --output-length 1024 --tpot-limit 200 --ttft-limit 30000 --quantize ENABLE --pd-balance-search --quantize-linear-action MXFP4 --quantize-attention-action FP8 --tp-sizes 2 4 8 16

====================================================================================================
infer-strategy-npu - recommend summary
====================================================================================================
run_id      : 20261021-152953-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-fb6fc748
status      : ok
model       : ZhipuAI/GLM-5.3
device      : ATLAS_950_486T_96G  x32
isl/osl     : 8196/1024   tpot-limit=200 ms   ttft-limit=30000 ms
model-source: model_configs/msmodeling/GLM5.3

  PD 混部 (agg):
  +-----+------+----------+--------------+----------+----------+--------+--------+---------------------------------------------------+
  | Top | Mode | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc.  |                     Parallel                      |
  +-----+------+----------+--------------+----------+----------+--------+--------+---------------------------------------------------+
  |  1  | agg  | 10183.68 |    318.24    | 29551.62 |  81.70   | 9.945  |  832   | TP=2 | PP=1 | DP=16 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  2  | agg  | 9733.75  |    304.18    | 27440.79 |  78.90   | 9.506  |  768   | TP=2 | PP=1 | DP=16 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  3  | agg  | 7504.79  |    234.53    | 18997.47 |  68.22   | 7.329  |  512   | TP=2 | PP=1 | DP=16 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  4  | agg  | 7485.12  |    233.91    | 29642.24 |  83.36   | 7.310  |  624   | TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  5  | agg  | 7404.83  |    231.40    | 28936.70 |  82.11   | 7.231  |  608   | TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  6  | agg  | 7198.55  |    224.96    | 27489.87 |  80.02   | 7.030  |  576   | TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  7  | agg  | 6706.18  |    209.57    | 24596.20 |  76.35   | 6.549  |  512   | TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  8  | agg  | 4857.25  |    151.79    | 29790.29 |  84.82   | 4.743  |  412   | TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  9  | agg  | 4831.07  |    150.97    | 29508.71 |  84.45   | 4.718  |  408   | TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  10 | agg  | 4777.08  |    149.28    | 28962.17 |  83.73   | 4.665  |  400   | TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  11 | agg  | 4689.16  |    146.54    | 27848.24 |  81.89   | 4.579  |  384   | TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  12 | agg  | 3797.97  |    118.69    | 18936.81 |  67.40   | 3.709  |  256   | TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  13 | agg  | 2593.04  |     81.03    | 29865.98 |  85.61   | 2.532  |  222   | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  14 | agg  | 2580.27  |     80.63    | 29603.65 |  85.26   | 2.520  |  220   | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  15 | agg  | 2553.98  |     79.81    | 29092.95 |  84.57   | 2.494  |  216   | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  16 | agg  | 2500.63  |     78.14    | 28053.92 |  83.18   | 2.442  |  208   | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  17 | agg  | 2400.64  |     75.02    | 25975.85 |  79.98   | 2.344  |  192   | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  18 | agg  | 1862.46  |     58.20    | 17663.58 |  68.73   | 1.819  |  128   | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  +-----+------+----------+--------------+----------+----------+--------+--------+---------------------------------------------------+

  PD 分离 (disagg) - xPyD 变体 1 组:

    [1P1D]  P=16 卡 (1 节点)   D=16 卡 (1 节点)
    +-----+--------------+----------+--------------+----------+----------+--------+-------+-------------------------------------------------------------------+
    | Top |     Mode     | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc. |                              Parallel                             |
    +-----+--------------+----------+--------------+----------+----------+--------+-------+-------------------------------------------------------------------+
    |  1  | 1P1D_prefill | 61532.81 |   3845.80    | 2131.16  |    -     | 60.091 |   16  |          TP=2 | PP=1 | DP=8 | EP=16 | MOE-TP=1 | MOE-DP=1         |
    |  1  | 1P1D_decode  | 23672.43 |   1479.53    |    -     |  41.57   | 23.118 |  984  | TP=2 | PP=1 | DP=8 | EP=16 | MOE-TP=1 | MOE-DP=1 | DFlash=8/acc=5 |
    +-----+--------------+----------+--------------+----------+----------+--------+-------+-------------------------------------------------------------------+

----------------------------------------------------------------------------------------------------

artifacts   : \workspace\results\infer-strategy-npu\20261021-152953-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-fb6fc748
run.json    : \workspace\results\infer-strategy-npu\20261021-152953-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-fb6fc748\run.json
====================================================================================================

```


---

## 附录 A 快速开始

```bash
# 1. 安装
cd <仓库根目录>
pip install -e .

# 2. 确认环境
infer-strategy-npu --list-models
infer-strategy-npu --list-devices

# 3. 参数预演（不产出结果，仅校验；本地无 NPU 时推荐）
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 --quantize DISABLE --dry-run

# 4. 正式寻优（PD 混部 + PD 分离）
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 --quantize DISABLE

# 5. 查看结果
cat ./results/infer-strategy-npu/<run_id>/summary.txt
```

---


*文档版本随工具迭代更新；参数以 `infer-strategy-npu -recommend --help` 实时输出为准。*
