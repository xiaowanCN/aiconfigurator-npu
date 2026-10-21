# 推理策略寻优 使用指南

> 本文档介绍统一推理策略寻优工具 `infer-strategy-npu` 的安装、使用与结果说明。
> 全文以 **`ZhipuAI/GLM-5.3`** 为例，覆盖 **16 卡 / 32 卡**、**`--quantize DISABLE` / `ENABLE`** 两种量化开关。
> 阅读对象为需要评估大模型服务部署方案的性能工程师、容量规划人员与平台开发者。

---

## 1 简介

### 1.1 工具介绍

`infer-strategy-npu` 是一款面向大模型推理部署的策略寻优工具：它以内置的**算子级性能模型**为基础，通过对 GEMM、Attention、MoE 与集合通信等核心算子在目标硬件上的微基准刻画，构建可外推的性能数据库，并据此在给定模型、卡数与 SLA（TTFT / TPOT）约束下，对**张量并行、流水并行、数据并行、专家并行**的组合、批大小、以及 **PD 混部 / PD 分离** 两种服务形态做统一组合搜索，**一条命令即可同时给出两种部署形态的全部候选配置与各自最优解**，让用户在不上机试错的前提下完成部署选型与容量规划。

量化方面，本工具将 `--quantize` 设计为**高层总开关**：

- `--quantize DISABLE`：完全不量化，三个量化动作强制为 `DISABLED`；
- `--quantize ENABLE`：开启量化，具体动作由 `--quantize-linear-action` / `--quantize-non-expert-linear-action` / `--quantize-attention-action` 三个参数决定（缺省时取工具默认）；
- 旧版 `W8A8` / `FP8` 仍可作为 `ENABLE` 的别名向后兼容。

工具会依据模型、设备与量化开关自动完成全部搜索与评估，**用户无需关心内部细节**，也无需手动指定额外的执行命令。

### 1.2 模型适配矩阵

| 模型 | 架构特征 | 量化支持 | PD 混部 | PD 分离 | 适配状态 |
|------|----------|----------|---------|---------|----------|
| `ZhipuAI/GLM-5.3` | GLM MoE + DSA | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `ZhipuAI/GLM-5.2` | GLM MoE + DSA | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `ZhipuAI/GLM-5.3-Flash` | GLM MoE + DSA | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `Qwen/Qwen3.8-27B` | Qwen3.5（Hybrid Attention） | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `deepseek-ai/DeepSeek-R1` | DeepSeek-V3（MLA + MoE） | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek-V4（CSA/HCA + 大规模专家） | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `moonshotai/Kimi-K3` | MoE | DISABLE / ENABLE | 支持 | 支持 | 已适配 |
| `MiniMax/MiniMax-M3` | MoE | DISABLE / ENABLE | 支持 | 支持 | 已适配 |

> **说明：** `infer-strategy-npu --list-models` 打印的是实时注册表；上表为当前版本矩阵。若某模型的本地配置尚未替换为真实配置，寻优结果可能不可用，建议先以 `--dry-run` 校验。

### 1.3 设备说明

- 支持的设备以 `infer-strategy-npu --list-devices` 实时输出为准；本文 GLM 示例统一使用 `ATLAS_950_486T_96G`。
- 不同模型 / 量化开关对设备的支持范围可能不同；若传入的组合不支持，工具会直接报错提示，无需用户记忆兼容细节。

---

## 2 主要场景（GLM-5.3，16 / 32 卡）

本工具的核心特点：**一条命令同时产出 PD 混部与 PD 分离的全部候选配置，并为每种部署方式各自给出最优解**。以下以 GLM-5.3 为例，分别给出 16 卡与 32 卡的命令。

> 本地无 NPU 时可用 `--dry-run` 仅校验参数（不实际执行），预览将要搜索的并行空间与候选配置。

### 2.1 一次寻优：PD 混部 + PD 分离全量配置（DISABLE）

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
- `--dry-run` 可提前确认参数与搜索空间是否合理，不会真正执行。

### 2.2 量化模型寻优（ENABLE + 三个 action 参数）

**适用场景：** 显存受限或希望评估量化后的吞吐收益。GLM-5.3 开启量化（`ENABLE`）时，可显式指定专家线性层、非专家线性层、KV Cache 三套量化动作；不指定则取工具默认。

**16 卡 ENABLE（默认动作）：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize ENABLE
```

**32 卡 ENABLE（专家层 MXFP4、KV Cache FP8，非专家层保持 DISABLED）：**

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 32 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize ENABLE \
  --quantize-linear-action MXFP4 --quantize-attention-action FP8
```

**约束：**

- `--quantize DISABLE` 时三个 `--quantize-*-action` 会被强制覆盖为 `DISABLED`（传入也无效）；
- `--quantize ENABLE` 时传入的 action 优先，缺省取工具默认；
- `--quantize-non-expert-linear-action` 主要用于 MoE 类模型（专家层单独指定量化动作，其余线性层如 attention 投影 / dense MLP / shared expert 默认 `DISABLED`），一般保持 `DISABLED` 即可；
- 量化动作取值大小写与下划线/连字符写法均可（如 `W8A8_DYNAMIC` 与 `w8a8-dynamic`）。
- 旧版 `--quantize W8A8` / `--quantize FP8` 仍可作为 `ENABLE` 别名。

### 2.3 PD 分离配比搜索（xPyD）

**适用场景：** 已确定走 PD 分离，需要规划 Prefill 与 Decode 的实例数量与卡数配比。`--pd-balance-search` 默认开启。

**32 卡，限定 1P1D（Prefill 16 卡 + Decode 16 卡）：**

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

1. **输入配置**：模型、设备、卡数、ISL/OSL、SLO 上限、量化；
2. **各形态 Top 配置表**：PD 混部（agg）与 PD 分离（disagg）各一张，按单卡吞吐降序；
3. **最终推荐**：`recommendation` 段给出服务模式、最优配置与理由。

下表为汇总表结构示意（列与 GLM-5.3 运行后一致；数值由工具实际计算）：

```text
  PD 混部 (agg):
  +-----+------+----------+--------------+----------+----------+--------+-------+------------------+
  | Top | Mode | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc. |     Parallel     |
  +-----+------+----------+--------------+----------+----------+--------+-------+------------------+
  |  1  | agg  |  ......  |    ......    |  ......  |  ......  | ...... |  ...  | tp4pp1dp4 ...... |
  |  2  | agg  |  ......  |    ......    |  ......  |  ......  | ...... |  ...  | tp8pp1dp2 ...... |
  +-----+------+----------+--------------+----------+----------+--------+-------+------------------+

  PD 分离 (disagg):
  +-----+----------------+----------+--------------+----------+----------+--------+-------+----------------------------------------------------+
  | Top |      Mode      | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc. |                      Parallel                      |
  +-----+----------------+----------+--------------+----------+----------+--------+-------+----------------------------------------------------+
  |  1  | disagg_prefill |  ......  |    ......    |  ......  |    -     | ...... |  ...  |                  tp4pp1dp4 ......                  |
  |  1  | disagg_decode  |  ......  |    ......    |    -     |  ......  | ...... |  ...  |                  tp4pp1dp4 ......                  |
  |  1  | disagg_overall |  ......  |    ......    |  ......  |  ......  | ...... |  ...  | prefill[...]/decode[...]                            |
  +-----+----------------+----------+--------------+----------+----------+--------+-------+----------------------------------------------------+
```

重点关注：

| 字段 | 含义 |
|------|------|
| `Top` | 该形态内的排名（按 `tokens/s/dev` 降序） |
| `Mode` | `agg` / `disagg_prefill` / `disagg_decode` / `disagg_overall` |
| `tokens/s` | 单副本吞吐；`disagg_overall` 为 P/D 合流后的端到端吞吐 |
| `tokens/s/dev` | 单卡吞吐，跨配置比较的主指标 |
| `TTFT(ms)` / `TPOT(ms)` | 首 token 时间 / 每输出 token 时间；阶段表中不适用的指标显示 `-` |
| `req/s` | 每秒请求数 |
| `Conc.` | 并发请求数 |
| `Parallel` | 并行策略（`tp/pp/dp/etp/ep`）；`disagg_overall` 为 `prefill[...]/decode[...]` |

> **说明：** PD 混部的 `tokens/s` 是**单副本**吞吐，副本数越多单副本值越小但集群总量越大；跨配置比较请看 `tokens/s/dev`，或按 6.1 换算集群总吞吐。`tp4pp1dp4` 表示单副本 4 卡（16 卡即 4 个副本）。

### 3.2 候选明细列说明

PD 混部与 PD 分离的候选明细重点列：

| 列 | 含义 |
|----|------|
| `tokens/s/gpu` | 单卡吞吐 |
| `tokens/s/user` | 单用户吞吐 |
| `TTFT` / `TPOT` | 延迟（ms） |
| `(p)workers` / `(p)parallel` / `(p)bs` | PD 分离：Prefill 侧 worker 数 / 并行 / 每 worker 批大小 |
| `(d)workers` / `(d)parallel` / `(d)bs` | PD 分离：Decode 侧 worker 数 / 并行 / 每 worker 批大小 |
| `gpus/replica` | 单副本卡数 = Prefill 卡数 + Decode 卡数 |
| `replicas` | 副本份数；实际占用卡数 = `gpus/replica × replicas` |

> **说明：** `(p)bs` 恒为 1 属于正常现象——Prefill 是计算密集型阶段，加大批大小会显著推高 TTFT；工具通过增加 Prefill worker 数量而非增大单 worker 批大小来提升吞吐，因此各候选的 TTFT 往往一致。

### 3.3 多卡数对比

单次运行可传入不同 `--num-devices`（如 `16` 与 `32`）分别寻优，对比候选并行配置表；横向比较 16 卡与 32 卡时，在 SLO 约束下 32 卡的 `tokens/s/dev` 一般更高，但单副本 `tokens/s` 取决于副本数，需用 6.1 公式换算集群总吞吐后再下结论。

### 3.4 run.json 字段

每次运行在 `<out-dir>/<run_id>/run.json` 落一份统一结果：

| 字段路径 | 含义 |
|----------|------|
| `schema_version` | 结果 schema 版本 |
| `status` | `ok` / `partial` / `failed` |
| `request` | 归一化后的请求参数（含量化开关与三个 action） |
| `frameworks[0].agg` | PD 混部结果：`best` + `candidates` |
| `frameworks[0].disagg` | PD 分离结果：`best` + `candidates` + `pd_balance` |
| `recommendation` | 跨形态最终推荐（`mode`、`best`、`reason`） |
| `errors` | 失败信息列表 |
| `artifacts` | 产物路径（`run_dir`、`run_json`、`summary`、`logs_dir`） |

### 3.5 成功标准与任务状态

| status | 含义 | 处理建议 |
|--------|------|----------|
| `ok` | agg 与 disagg 均产出结果 | 直接查看 `recommendation` |
| `partial` | 仅一种形态可用 | 查看 `errors` 与不可用形态的原因 |
| `failed` | 未产出任何结果 | 按 6.3 排查 |

---

## 4 参数

### 4.1 必填参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | ModelScope 模型 ID（见 `--list-models`） | `ZhipuAI/GLM-5.3` |
| `--device` | 统一设备名（见 `--list-devices`） | `ATLAS_950_486T_96G` |
| `--num-devices` | NPU 卡数（正整数） | `16` / `32` |

### 4.2 常用可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-length` | `1024` | 输入序列长度 ISL |
| `--output-length` | `1024` | 输出序列长度 OSL |
| `--tpot-limit` | `50` | TPOT 上限（ms），SLA 硬约束 |
| `--ttft-limit` | `1000` | TTFT 上限（ms），SLA 硬约束 |
| `--quantize` | `DISABLE` | 量化总开关：`DISABLE` / `ENABLE`（旧版 `W8A8`/`FP8` 为 ENABLE 别名） |
| `--top-n` | `10` | 每类返回的候选数量 |
| `--out-dir` | `./results/infer-strategy-npu` | 结果输出根目录 |
| `--format` | `table` | 终端输出格式：`table` / `json` |
| `--dry-run` | 关闭 | 仅校验参数，**不产出结果**（本地无 NPU 时推荐） |
| `--quiet` | 关闭 | 不打印摘要 |
| `--verbose` | 关闭 | 打印 DEBUG 日志 |

### 4.3 量化高级参数

| 参数 | 说明 | 取值 |
|------|------|------|
| `--quantize-linear-action` | 专家线性层量化动作（ENABLE 生效） | `DISABLED` / `W8A16_STATIC` / `W8A8_STATIC` / `W4A8_STATIC` / `W8A16_DYNAMIC` / `W8A8_DYNAMIC` / `W4A8_DYNAMIC` / `FP8` / `MXFP4`（默认 `W8A8_DYNAMIC`） |
| `--quantize-non-expert-linear-action` | 非专家线性层量化动作（attention 投影 / dense MLP / shared expert；ENABLE 生效，默认 `DISABLED`） | 同 `--quantize-linear-action` |
| `--quantize-attention-action` | KV Cache 量化动作（ENABLE 生效） | `DISABLED` / `INT8` / `FP8`（默认 `DISABLED`） |

> **说明：** `DISABLE` 模式下上述三个 action 强制为 `DISABLED`；`ENABLE` 模式下显式传入优先，缺省取工具默认。

### 4.4 搜索与部署高级参数

| 参数 | 说明 |
|------|------|
| `--pd-balance-search` / `--no-pd-balance-search` | xPyD 配比搜索开关（默认开启） |
| `--pd-balance-max` | xPyD 搜索的 P/D 节点数上限（默认 3） |
| `--pd-balance-sizes` | 显式指定 xPyD 组合，如 `1P1D 2P1D` |
| `--tp-sizes` | 显式指定 TP 候选（空格分隔；缺省按卡数自动推导，GLM-5.3 从 4 起跳） |
| `--jobs` | 内部并发度 |
| `--max-search-combinations` | 最大搜索组合数 |
| `--local-model-root` | 本地模型根目录 |
| `--speculative-method` | 投机解码方法：`none` / `mtp` / `dflash` / `dspark` |
| `--num-speculative-tokens` / `--acceptance-length` / `--num-draft-layers` | 投机解码细项 |
| `--compile` | 启用编译优化 |

> **说明：** 完整参数以 `infer-strategy-npu -recommend --help` 为准。

### 4.5 辅助命令

| 命令 | 说明 |
|------|------|
| `infer-strategy-npu --list-models` | 打印支持的模型清单 |
| `infer-strategy-npu --list-devices` | 打印支持的设备清单 |
| `infer-strategy-npu --version` | 打印版本 |

### 4.6 参数速查表

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize DISABLE|ENABLE \
  [--quantize-linear-action MXFP4] [--quantize-attention-action FP8] \
  [--top-n 10] [--out-dir ./results/infer-strategy-npu] \
  [--pd-balance-search] [--pd-balance-max 3] \
  [--dry-run] [--format table|json]
```

---

## 5 GLM-5.3 运行实例与结果查看（16 / 32 卡）

### 5.1 16 卡 · DISABLE · 仅预览命令

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 16 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize DISABLE --dry-run
```

`--dry-run` 不执行，可安全在本机校验参数与设备是否匹配，并预览将要搜索的并行空间与候选配置。

### 5.2 32 卡 · ENABLE(MXFP4/FP8) · 正式运行

```bash
infer-strategy-npu -recommend \
  --model ZhipuAI/GLM-5.3 \
  --device ATLAS_950_486T_96G --num-devices 32 \
  --input-length 8196 --output-length 1024 \
  --tpot-limit 200 --ttft-limit 30000 \
  --quantize ENABLE \
  --quantize-linear-action MXFP4 --quantize-attention-action FP8 \
  --out-dir ./results/glm53-32-enable
```

运行后在 `./results/glm53-32-enable/<run_id>/` 下查看：

- `run.json`：统一结果（含 `request`、`frameworks[].agg/disagg`、`recommendation`）；
- `summary.txt`：人类可读的汇总；
- `logs/`：各子任务的执行日志（含原始搜索输出）；
- 其它产物目录（按任务自动组织）。

### 5.3 结果查看建议

1. 先看 `recommendation`：工具已按 GPU 预算归一化后的集群总吞吐给出最终推荐（口径见 6.1）；
2. 再对比 `agg` 与 `disagg_overall` 的 `tokens/s/dev` 与 `TPOT`：PD 分离通常 TPOT 更优、但端到端吞吐受 P/D 配比影响；
3. 横向比较 16 卡与 32 卡：在 SLO 约束下，32 卡的 `tokens/s/dev` 一般更高，但单副本 `tokens/s` 取决于副本数，需用 6.1 公式换算集群总吞吐后再下结论；
4. 量化对比：同一卡数下分别跑 `--quantize DISABLE` 与 `--quantize ENABLE`（配合具体 action），对比 `Throughput` 与显存占用（量化主要降低显存、放宽批大小上限）。

---

## 6 补充说明

### 6.1 推荐结论的计算口径

工具同时给出 PD 混部与 PD 分离的最优配置，最终推荐按**GPU 预算归一化后的集群总吞吐**比较：

```
集群总吞吐 = 单卡吞吐 × 总卡数
          = (tokens/s/gpu) × num_devices
```

之所以不直接比较候选的 `tokens/s`，是因为不同候选的**副本数不同**（例如 `tp4pp1` 用 4 卡/副本、16 卡可部署 4 份；`tp16pp1` 用 16 卡/副本、只能部署 1 份）。若只比单副本吞吐，副本多的配置会被严重低估。归一化到同一卡数预算后，比较才公平。

### 6.2 指标含义

| 指标 | 含义 | 备注 |
|------|------|------|
| TTFT | Time To First Token，首 token 时间（ms） | 越小越好，受 Prefill 阶段影响 |
| TPOT | Time Per Output Token，每输出 token 时间（ms） | 越小越好，受 Decode 阶段影响 |
| tokens/s/gpu | 单卡吞吐 | 跨配置比较的主指标 |
| tokens/s/user | 单用户吞吐 | 衡量单用户体验 |
| concurrency | 并发请求数 | 候选配置支持的最大并发 |
| gpus/replica | 单副本卡数 | Prefill 卡数 + Decode 卡数 |
| replicas | 副本数 | 总卡数 ÷ 单副本卡数 |

### 6.3 失败排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 模型加载失败 | 模型配置路径不在容器内，或为骨架配置 | 确认容器内路径存在；骨架模型需替换真实 `config.json` |
| 无配置满足 SLA | TTFT / TPOT 上限过紧或卡数不足 | 放宽 `--tpot-limit` / `--ttft-limit`，或增大 `--num-devices` |
| 设备不支持 | `--device` 拼写错误，或该模型/量化组合不支持此设备 | 用 `--list-devices` 核对；更换为支持的设备 |
| 运行时间过长 | xPyD 组合过多 | 降低 `--pd-balance-max`，或 `--no-pd-balance-search` |
| `status = partial` | 仅一种形态可用 | 查看 `errors`，通常为某形态在约束下无解 |

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

## 附录 B 术语表

| 术语 | 说明 |
|------|------|
| PD 混部（agg） | Prefill 与 Decode 在同一实例中运行 |
| PD 分离（disagg） | Prefill 与 Decode 拆分为独立实例组网 |
| ISL / OSL | 输入序列长度 / 输出序列长度 |
| TTFT / TPOT | 首 token 时间 / 每输出 token 时间 |
| TP / PP / DP / EP | 张量并行 / 流水并行 / 数据并行 / 专家并行 |
| xPyD | P 个 Prefill 实例 + D 个 Decode 实例的配比 |
| 副本（replica） | 一套完整部署配置的拷贝 |
| Pareto 前沿 | 吞吐与延迟权衡下的非支配配置集合 |

## 附录 C 量化动作取值

| 参数 | 取值集合 | ENABLE 缺省值 |
|---|---|---|
| `--quantize-linear-action`（专家线性层） | `DISABLED, W8A16_STATIC, W8A8_STATIC, W4A8_STATIC, W8A16_DYNAMIC, W8A8_DYNAMIC, W4A8_DYNAMIC, FP8, MXFP4` | `W8A8_DYNAMIC` |
| `--quantize-non-expert-linear-action`（非专家线性层） | 同上 | `DISABLED` |
| `--quantize-attention-action`（KV Cache） | `DISABLED, INT8, FP8` | `DISABLED` |

---

## 附录 D GLM5.3 运行结果示例
```text
❯ python -m cli.inference.throughput_optimizer models/GLM5.3 --device ATLAS_950_486T_96G --num-devices 16 --tp-sizes 4 8 16 --input-length 8196 --output-length 1024 --quantize-linear-action FP8 --quantize-attention-action FP8 --tpot-limit 100 --ttft-limit 20000
                                        =================================================================
                                                            >>>>>   MindStudio   <<<<<                                                            
                                            THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE
                                        =================================================================                                         

[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!

********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration:
    Model: models/GLM5.3
    Quantize Linear action: FP8
    Quantize Attention action: FP8
    Devices: 16 ATLAS_950_486T_96G
    TTFT Limits: 20000.0 ms
    TPOT Limits: 100.0 ms
  ----------------------------------------------------------------------------
  Memory Info:
    Total device memory:  96.000 GB
    Reserved memory:      10.000 GB
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    Best Throughput: 1926.05 tokens/s
    TTFT: 19684.82 ms
    TPOT: 89.30 ms
  ----------------------------------------------------------------------------
Top 3 PD Aggregated Configurations:
+-----+----------------------+-----------+-----------+-------------+-------------+---------------------------------------------------+------------+-----------+-------------+---------------+----------+
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices |                      parallel                     | batch_size | weight_GB | kv_cache_GB | activation_GB | avail_GB |
+-----+----------------------+-----------+-----------+-------------+-------------+---------------------------------------------------+------------+-----------+-------------+---------------+----------+
|  1  |       1926.05        |  19684.82 |   89.30   |     172     |      16     |  TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1 |     43     |   51.75   |     1.46    |     31.57     |   1.23   |
|  2  |       1320.28        |  19983.65 |   90.89   |     120     |      16     |  TP=8 | PP=1 | DP=2 | EP=16 | MOE-TP=1 | MOE-DP=1 |     60     |   50.09   |    23.18    |      0.00     |  12.73   |
|  3  |        735.02        |  19980.61 |   91.15   |      67     |      16     | TP=16 | PP=1 | DP=1 | EP=16 | MOE-TP=1 | MOE-DP=1 |     67     |   49.27   |    25.88    |      0.00     |  10.85   |
+-----+----------------------+-----------+-----------+-------------+-------------+---------------------------------------------------+------------+-----------+-------------+---------------+----------+
********************************************************************************

                                      ATLAS_950_486T_96G_models/GLM5.3: Throughput vs concurrency
      ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
2023.0┤ ●TP=16 | PP=1 | DP=1 | EP=16 | MOE-TP=1 | MOE-DP=1                                                                     │
      │ ●TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1                                                                      │
      │ ●TP=8 | PP=1 | DP=2 | EP=16 | MOE-TP=1 | MOE-DP=1                                                           ●          │
      │                                                                                                           ●            │
      │                                                                                                   ●                    │
1788.7┤                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
1554.3┤                                                                     ●                                                  │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
1319.9┤                                                             ●                                                          │
      │                                                      ●                                                                 │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                       ●                                                                                │
1085.6┤                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
 851.2┤        ●                                                                                                               │
      │                                                                                                                        │
      │                                                                                                                        │
      │        ● ●●                                                                                                            │
      │                                                                                                                        │
 616.8┤                                                                                                                        │
      └┬─────────────────────────────┬─────────────────────────────┬────────────────────────────┬─────────────────────────────┬┘
     55.4                          86.7                          118.0                        149.3                       180.6
Throughput (token/s)                                          Concurrency


                                         ATLAS_950_486T_96G_models/GLM5.3: Throughput vs TPOT
      ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
2023.0┤ ●TP=16 | PP=1 | DP=1 | EP=16 | MOE-TP=1 | MOE-DP=1                                                                     │
      │ ●TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1                                                                      │
      │ ●TP=8 | PP=1 | DP=2 | EP=16 | MOE-TP=1 | MOE-DP=1                                              ●                       │
      │                                                                                              ●                         │
      │                                                                                     ●                                  │
1788.7┤                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
1554.3┤                                                         ●                                                              │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
1319.9┤                                                                                                             ●          │
      │                                                                           ●                                            │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                     ●                                                                  │
1085.6┤                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                                        │
 851.2┤        ●                                                                                                               │
      │                                                                                                                        │
      │                                                                                                                        │
      │                                                                                                    ●       ●  ●        │
      │                                                                                                                        │
 616.8┤                                                                                                                        │
      └┬─────────────────────────────┬─────────────────────────────┬────────────────────────────┬─────────────────────────────┬┘
     75.2                          79.5                          83.7                         88.0                         92.3
Throughput (token/s)                                           TPOT (ms)

All experiments completed in 88.09 seconds.
```

```text
❯ infer-strategy-npu -recommend --model ZhipuAI/GLM-5.3 --device ATLAS_950_486T_96G --num-devices 32 --input-length 8196 --output-length 1024 --tpot-limit 200 --ttft-limit 30000 --quantize FP8 --pd-balance-search
[INFO] [serving_cast.parallel_runner] Run Aggregation with ttft 30000.0 ms, tpot 200.0 ms.
[WARNING] [tensor_cast.core.model_source_security] Local model path is on filesystem type drvfs, whose POSIX permission bits may be synthetic. Owner and symlink checks are still enforced, but group/world-writable mode bits are not rejected.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[INFO] [serving_cast.parallel_runner] Run Prefill with ttft 30000.0 ms.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[INFO] [serving_cast.parallel_runner] Run Decode with tpot 200.0 ms.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = 64.0) does not match the ratio implicitly set by other parameters (implicit factor = post-yarn context length / pre-yarn context length = config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = 256.0). Using the explicit factor (64.0) in YaRN. This may cause unexpected behaviour in model usage, please correct the 'original_max_position_embeddings' fields in the model config.
[transformers] The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = 64.0) does not match the ratio implicitly set by other parameters (implicit factor = post-yarn context length / pre-yarn context length = config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = 256.0). Using the explicit factor (64.0) in YaRN. This may cause unexpected behaviour in model usage, please correct the 'original_max_position_embeddings' fields in the model config.
[transformers] The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = 64.0) does not match the ratio implicitly set by other parameters (implicit factor = post-yarn context length / pre-yarn context length = config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = 256.0). Using the explicit factor (64.0) in YaRN. This may cause unexpected behaviour in model usage, please correct the 'original_max_position_embeddings' fields in the model config.
====================================================================================================
infer-strategy-npu - recommend summary
====================================================================================================
run_id      : 20261021-111730-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-530d4094
status      : ok
model       : ZhipuAI/GLM-5.3
device      : ATLAS_950_486T_96G  x32
isl/osl     : 8196/1024   tpot-limit=200 ms   ttft-limit=30000 ms
quantize    : ENABLE
framework   : msmodeling  (model ZhipuAI/GLM-5.3 with quantize=ENABLE is served by msmodeling (aic-npu only covers DeepSeek-R1 / Qwen3.8-27B / DeepSeek-V4-Pro with quantize=DISABLE))
model-source: model_configs/msmodeling/GLM5.3

[msmodeling] status=ok
  $ python -m cli.inference.throughput_optimizer model_configs/msmodeling/GLM5.3 --device ATLAS_950_486T_96G --num-devices 32 --input-length 8196 --output-length 1024 --quantize-linear-action W8A8_DYNAMIC --quantize-non-expert-linear-action DISABLED --quantize-attention-action DISABLED --tp-sizes 4 8 16 32 --tpot-limit 200 --ttft-limit 30000
  $ python -m cli.inference.throughput_optimizer model_configs/msmodeling/GLM5.3 --device ATLAS_950_486T_96G --num-devices 16 --input-length 8196 --output-length 1024 --quantize-linear-action W8A8_DYNAMIC --quantize-non-expert-linear-action DISABLED --quantize-attention-action DISABLED --tp-sizes 4 8 16 --disagg --ttft-limit 30000
  $ python -m cli.inference.throughput_optimizer model_configs/msmodeling/GLM5.3 --device ATLAS_950_486T_96G --num-devices 16 --input-length 8196 --output-length 1024 --quantize-linear-action W8A8_DYNAMIC --quantize-non-expert-linear-action DISABLED --quantize-attention-action DISABLED --speculative-method dflash --num-speculative-tokens 7 --acceptance-length 5 --num-draft-layers 6 --tp-sizes 4 8 16 --disagg --tpot-limit 200

  PD 混部 (agg):
  +-----+------------+------+----------+--------------+----------+----------+-------+-------+---------------------------------------------------+
  | Top | Framework  | Mode | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s | Conc. |                      Parallel                     |
  +-----+------------+------+----------+--------------+----------+----------+-------+-------+---------------------------------------------------+
  |  1  | msmodeling | agg  | 1882.69  |    58.83     | 20029.68 |  135.98  | 1.839 |  256  |  TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  2  | msmodeling | agg  | 2096.11  |    65.50     | 28931.76 |  183.20  | 2.047 |  384  |  TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  3  | msmodeling | agg  | 2104.49  |    65.77     | 29454.88 |  186.27  | 2.055 |  392  |  TP=4 | PP=1 | DP=8 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  4  | msmodeling | agg  |  970.94  |    30.34     | 16082.92 |  131.83  | 0.948 |  128  |  TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  5  | msmodeling | agg  | 1102.85  |    34.46     | 23230.89 |  174.09  | 1.077 |  192  |  TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  6  | msmodeling | agg  | 1147.39  |    35.86     | 26804.88 |  195.23  | 1.120 |  224  |  TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  7  | msmodeling | agg  | 1150.35  |    35.95     | 27228.85 |  198.20  | 1.123 |  228  |  TP=8 | PP=1 | DP=4 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  8  | msmodeling | agg  |  492.15  |    15.38     | 14930.84 |  130.04  | 0.481 |   64  | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  9  | msmodeling | agg  |  548.78  |    17.15     | 21566.76 |  174.93  | 0.536 |   96  | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  10 | msmodeling | agg  |  570.87  |    17.84     | 24884.73 |  196.19  | 0.557 |  112  | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  11 | msmodeling | agg  |  573.50  |    17.92     | 25278.38 |  198.78  | 0.560 |  114  | TP=16 | PP=1 | DP=2 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  12 | msmodeling | agg  |  245.76  |     7.68     | 14172.31 |  130.21  | 0.240 |   32  | TP=32 | PP=1 | DP=1 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  13 | msmodeling | agg  |  279.01  |     8.72     | 20471.11 |  172.04  | 0.272 |   48  | TP=32 | PP=1 | DP=1 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  14 | msmodeling | agg  |  290.54  |     9.08     | 23620.51 |  192.74  | 0.284 |   56  | TP=32 | PP=1 | DP=1 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  |  15 | msmodeling | agg  |  293.09  |     9.16     | 24381.56 |  197.89  | 0.286 |   58  | TP=32 | PP=1 | DP=1 | EP=32 | MOE-TP=1 | MOE-DP=1 |
  +-----+------------+------+----------+--------------+----------+----------+-------+-------+---------------------------------------------------+

  PD 分离 (disagg) - xPyD 变体 1 组:

    [1P1D]  P=16 卡 (1 节点)   D=16 卡 (1 节点)
    +-----+------------+--------------+----------+--------------+----------+----------+--------+-------+-------------------------------------------------------------------+
    | Top | Framework  |     Mode     | tokens/s | tokens/s/dev | TTFT(ms) | TPOT(ms) | req/s  | Conc. |                      
        Parallel                             |
    +-----+------------+--------------+----------+--------------+----------+----------+--------+-------+-------------------------------------------------------------------+
    |  1  | msmodeling | 1P1D_prefill | 29402.22 |   1837.64    | 3345.05  |    -     | 28.713 |   12  |          TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1         |
    |  1  | msmodeling | 1P1D_decode  | 4694.05  |    293.38    |    -     |  34.09   | 4.584  |  160  | TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1 | DFlash=8/acc=5 |
    |  1  | msmodeling | 1P1D_overall | 4694.05  |    293.38    |    -     |  34.09   | 4.584  |  160  | TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1 | DFlash=8/acc=5 |
    +-----+------------+--------------+----------+--------------+----------+----------+--------+-------+-------------------------------------------------------------------+

----------------------------------------------------------------------------------------------------
RECOMMENDATION
  serving_mode=disagg  mode=disagg_decode  framework=msmodeling
  throughput=4694.05 tokens/s
  ttft=- ms   tpot=34.09 ms
  parallel=TP=4 | PP=1 | DP=4 | EP=16 | MOE-TP=1 | MOE-DP=1 | DFlash=8/acc=5   backend=msmodeling
  reason=highest throughput (tokens/s) among the agg / disagg best configurations

artifacts   : \workspace\results\infer-strategy-npu\20261021-111730-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-530d4094
run.json    : \workspace\results\infer-strategy-npu\20261021-111730-ZhipuAI-GLM-5.3-ATLAS_950_486T_96G-32-530d4094\run.json   
====================================================================================================
```

*文档版本随工具迭代更新；参数以 `infer-strategy-npu -recommend --help` 实时输出为准。*
