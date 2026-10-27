# infer-strategy-npu --recommend 重构设计：PD 分离改用 msmodeling PD-ratio 搜索

> 状态：**已实现**（改动已落地，待真机联调验证）
> 目标版本：`infer_strategy_npu` schema 1.0 → **1.1**
> 涉及仓库：本仓库 `aiconfigurator-npu`（整合层）；`msmodeling` 以 **whl 形式安装**，只读消费、**不改动上游**

---

## 0. 评审结论（已定）

| 编号 | 议题 | 结论 |
|---|---|---|
| Q1 | 全部候选行的获取方式 | **方案 B**（`getattr(_prepare_pd_ratio_results)` + 原始 df 兜底） |
| Q2 | 排序口径 | **沿用 msmodeling 既有公式** `concurrency_d / tpot_d * 1000` |
| Q3 | 去掉 TP 起跳规则后的耗时上升 | **接受** |
| Q4 | PD 混部 / PD 分离的开关 | 删掉 `--pd-ratio-search` / `--no-pd-ratio-search`；**两者默认都跑**；新增 **`--serving-modes`**（`agg` / `disagg` 可选可组合），可指定只跑 PD 分离或只跑 PD 混部 |
| Q5 | 组合数爆炸的相位缓存优化 | **暂不做**（保持只用公共 API） |
| Q6 | 新模型本地配置 | 由你自行提供 |
| Q7 | `P+D > 卡数` 的组合 | **不报错**，直接剔除继续搜，全部不可行时记录一条 failure（可能无结果） |
| — | 并行统一字段 | **只有 `tp` / `dp` / `pp` / `ep`**，其余（`moe_tp` / `moe_dp` / `dcp` / `pp_*`）通通归入 `extra` |
| — | 集群/单卡吞吐获取 | 见 §4，已定位到 msmodeling 的内部计算式，在整合层**按同一公式复算** |

---

## 1. 背景：新旧 PD 分离做法的本质差异

| | 旧 xPyD 自研逻辑 | msmodeling 原生 PD-ratio |
|---|---|---|
| `xPyD` / `P,D` 含义 | **节点数**，卡数按 `unit = total // (x+y)` 均摊 | **每实例卡数** |
| 命令数 | 每组合 2 条（`--disagg --ttft-limit` + `--disagg --tpot-limit`） | 每组合 **1 条** |
| 集群配比 | 不做，P/D 两侧各自独立求解 | 内部枚举 `(P实例数, D实例数)` 最大化 `balanced_qps` |
| 结果表 | 两张独立的 P/D 表 | 一张「Top N PD Ratio Configurations」 |

已核实的 msmodeling 硬约束：

1. `--enable-optimize-prefill-decode-ratio` **不能与 `--disagg` 同用**（`cli/registry/validators.py:381-392` 的 `pd_ratio_mutex_disagg`）。
2. `--num-devices` 是**集群总卡数**；每实例卡数由 `--prefill/decode-devices-per-instance` 单独给。
3. `--ttft-limit` 与 `--tpot-limit` 需**同时**在顶层下发；`_run_pd_phase()` 各取其一（`parallel_runner.py:931-942`）。
4. 投机解码参数**必须下发**，否则 PD-ratio 的 D 阶段无投机，与旧 disagg 口径不公平。

---

## 2. 需求 1：模型分发保持不变 + 新增 DeepSeek-V4.1-Flash

### 2.1 保持不变

`routing.py` 的 `AIC_NPU_MODEL_IDS` 与 `route()` **一行不改**：

```
deepseek-ai/DeepSeek-R1      + --quantize DISABLE  -> aic-npu
Qwen/Qwen3.8-27B             + --quantize DISABLE  -> aic-npu
deepseek-ai/DeepSeek-V4-Pro  + --quantize DISABLE  -> aic-npu
以上三模型的 --quantize ENABLE                      -> msmodeling
其余全部模型                                          -> msmodeling
```

### 2.2 新增

`models.py` 的 `MODELS` 新增（msmodeling-only，不入 `AIC_NPU_MODEL_IDS` ⇒ 恒走 msmodeling）：

```python
ModelSpec(
    model_id="deepseek-ai/DeepSeek-V4.1-Flash",
    ms_model_id="deepseek-ai/DeepSeek-V4.1-Flash",
    local_dir="DeepSeek-V4.1-Flash",
),
```

本地目录 `model_configs/msmodeling/DeepSeek-V4.1-Flash/config.json` 由你提供（缺失时 `msmodeling_model_source()` 会回退到 ModelScope ID 联网拉取）。

---

## 3. 需求 2：取消 `--tp-sizes` 起跳规则

### 3.1 行为

| | 改前 | 改后 |
|---|---|---|
| `--tp-sizes` 缺省 | 按模型三档起跳推导（大模型 32 / 中等 4 / 其他 1），且**始终下发** | **完全不下发** `--tp-sizes`，msmodeling 自行推导 |
| `--tp-sizes` 显式传入 | 原样下发 | 不变 |

### 3.2 msmodeling 侧"自行决定"的实际语义（已核实）

- `arg_parse()`：`tp_sizes / ep_sizes / moe_dp_sizes` 全为 None 时置 `args.tp_sizes = []`（`cli/inference/throughput_optimizer.py:61-63`）。
- `resolve_search_sizes([], target, target)` → `[1,2,4,…,2^floor(log2(target))]`（`serving_cast/service/utils.py:534-560`）。
- 候选生成再按 `target % tp == 0` 过滤（`parallel_runner.py:642-644`）。

即 **TP = 所有能整除该侧卡数的 2 的幂**；PD-ratio 下 `target` 是**每实例卡数**（`parallel_runner.py:948`），与集群总卡数解耦。

### 3.3 改动清单

| 文件 | 动作 |
|---|---|
| `config.py` | 删 `MS_TP_SIZES_START` / `_LARGE_START` / `_MEDIUM_START` / `_LARGE_MODELS` / `_MEDIUM_MODELS` |
| `routing.py` | 删 `resolve_tp_sizes()` / `is_large_tp_model()` / `is_medium_tp_model()` / `_powers_of_two()`；`tp_size_flags()` 改为「`request.tp_sizes` 为空则返回 `[]`」 |
| `cli.py` | `--tp-sizes` 保留（默认 `None`），help 改为「缺省时不下发，由 msmodeling 按卡数自行推导」 |

> 已知代价：大模型 128 卡的 TP 候选由 `[32,64,128]` 变为 `[1,2,4,8,16,32,64,128]`，小 TP 会被 early-stop 跳过，单轮耗时上升（已确认接受）。

---

## 4. 需求 3 之核心：PD 分离吞吐的获取（第 1 点答复）

### 4.1 纵轴 `Throughput (token/s)` 是从哪来的

追踪链（msmodeling 内部）：

```
disagg_throughput_optimizer.py:376-379
    latency_ms = self._fold_decode_latency_ms(latency_ms, optimizer_data)   # 投机解码折叠后的 TPOT
    tpot = latency_ms
    output_throughput = concurrency / tpot * 1000        # ← 原生 token/s，单 D 实例
disagg_throughput_optimizer.py:404
    token_s_device = output_throughput / self.dp / self.pp / self.tp   # ← 原生 token/s/device
        ⬇ 写入 D 阶段 df 的 token/s、token/s/device 列
        ⬇ 但 _run_pd_ratio() 只保留 PD 交叉表，D 阶段 df 被丢弃
optimizer_curve_plots.py:579-589 (_pd_tps_curve_df)
    work["token/s"] = concurrency_d / tpot_d * 1000      # ← PD-ratio 图按同一公式复算
optimizer_curve_plots.py:592-608 (plot_pd_ratio_terminal_curves)
    → 标题 "<device>_<model>_pd_decode_tps"
```

**关键结论**：msmodeling 的 `pd_decode_tps` 图**自己也是从 PD 交叉表的 `concurrency_d` / `tpot_d` 复算的**，PD 结果表里根本没有 `token/s_d` 这一列。证据：

- `PDRatioThroughputOptimizer.optimize()` 的 `result_cols`（`pd_ratio_throughput_optimizer.py:133-150`）只含 `pd_ratio / p_qps / d_qps / balanced_qps / ttft_p / tpot_d / parallel_p / parallel_d / num_devices_p / num_devices_d / batch_size_p / batch_size_d / concurrency_p / concurrency_d / PP 列`，**无 `token/s_d`**。
- 图的输入是 `_first_non_empty_summary_df(results)` → `res.get_summary_df()`，即**原始 PD 交叉表**（不是过滤后的候选表）。

### 4.2 我们如何拿到（msmodeling 是 whl，改不动）

**不需要上游改任何东西**，也不用碰私有方法：

1. `ParallelRunner(args).run_disagg()` 在 PD-ratio 模式返回 `summary`；
2. `summary.get_summary_df()`（公共 API）拿到 PD 交叉表，其中已含 `concurrency_d` / `tpot_d` / `num_devices_d`；
3. 在整合层按**同一公式**逐行复算即可。

```python
# infer_strategy_npu/pd_ratio_metrics.py —— 与 msmodeling _pd_tps_curve_df 同源
decode_tps_per_instance = concurrency_d / tpot_d * 1000.0          # = 图纵轴
decode_tps_per_device   = decode_tps_per_instance / num_devices_d  # = 原生 token/s/device
```

数值自校验（你的 GLM5.2 样例）：`24 / 67.10 × 1000 = 357.7`，与图上限 365.1、下限 256.9 完全吻合（图只画了 concurrency 17.5~24.5 的区间）。

### 4.3 三级吞吐口径（全部落盘）

| 指标 | 公式 | 样例值 |
|---|---|---|
| 单 D 实例 decode 吞吐 | `concurrency_d / tpot_d * 1000` | 357.7 token/s |
| 单 D 实例单卡吞吐 | 上式 `/ num_devices_d` | 44.7 token/s/device |
| D 侧集群 decode 吞吐 | 上式 `× d_instances` | 1073.1 token/s |
| **集群端到端 token 吞吐** | `balanced_qps × output_length` | 1.05 × 1024 = **1075.2 token/s** |
| **集群单卡吞吐** | 集群端到端 `/ devices_used` | 1075.2 / 32 = **33.6 token/s/device** |
| 集群 req/s | `balanced_qps` | 1.05 req/s |
| 单 P 实例 prefill 吞吐 | `p_qps × input_length` | 2.90 × 4096 = 11878 token/s |
| 单 P 实例单卡吞吐 | 上式 `/ num_devices_p` | 1484.8 token/s/device |

> 注：`_fold_decode_latency_ms()` 已把投机解码加速折进 `tpot`，所以 `concurrency_d/tpot_d*1000` 天然**含投机加速**，无需再乘接受率。

### 4.4 排序

候选汇总表按 **`decode_tps_per_instance` 降序**（即 pd_decode_tps 图纵轴口径）。集群级三个派生量同列展示，供与 aic-npu 比较，但不参与排序。

---

## 5. 需求 3：PD-ratio 搜索与 `--serving-modes`

### 5.1 CLI 参数

```python
# 删除
--pd-balance-search / --no-pd-balance-search / --pd-balance-max / --pd-balance-sizes
--pd-ratio-search   / --no-pd-ratio-search        # 评审后一并取消：PD 分离默认必跑

# 新增：形态开关（挂在通用参数区，aic-npu / msmodeling 都认）
parser.add_argument(
    "--serving-modes", type=serving_mode, nargs="+", metavar="MODE",
    default=["agg", "disagg"],
    help="要搜索的部署形态（默认 agg disagg 都跑）；"
         "--serving-modes disagg 表示只跑 PD 分离，--serving-modes agg 表示只跑 PD 混部",
)
ms.add_argument(
    "--pd-ratio-sizes", type=positive_int, nargs="+", metavar="N",
    default=[8],
    help="PD 分离搜索时每实例卡数 P/D 的枚举空间（默认 8）；"
         "取 P×D 笛卡尔积，如 8 16 32 -> (8,8)(8,16)(8,32)(16,8)…(32,32)；"
         "P+D > --num-devices 的组合自动剔除，全部不可行时 PD 分离无结果",
)
```

`--serving-modes` 语义映射：

| 取值 | msmodeling | aic-npu |
|---|---|---|
| `agg disagg`（默认） | 跑 agg + 全部 PD-ratio 组合 | 一次调用同时产出，全量输出 |
| `disagg` | **只跑 PD-ratio 组合，跳过 agg** | 仍须一次调用（上游限制），**仅过滤掉 agg 输出**并在 `run.message` 注明 |
| `agg` | 只跑 agg | 同上，仅过滤 disagg 输出 |

### 5.2 组合枚举

```python
@dataclass(frozen=True)
class PdRatioCombo:
    prefill_devices: int   # 每个 P 实例的卡数
    decode_devices: int    # 每个 D 实例的卡数
    @property
    def label(self) -> str:        # "P8D8" / "P8D16"
        return f"P{self.prefill_devices}D{self.decode_devices}"

def pd_ratio_grid(total_devices, sizes) -> list[PdRatioCombo]:
    # sizes 去重保序 → 笛卡尔积(p 外层, d 内层) → 剔除 p+d > total_devices → 剔除 p/d > total
```

| `--pd-ratio-sizes` | 组合数 | 组合 |
|---|---|---|
| `8`（默认） | 1 | (8,8) |
| `8 16` | 4 | (8,8)(8,16)(16,8)(16,16) |
| `8 16 32` | 9 | (8,8)(8,16)(8,32)(16,8)(16,16)(16,32)(32,8)(32,16)(32,32) |

标签/目录名用 `P{p}D{d}`，**不复用旧的 `1P1D`**（避免与"节点数"语义混淆）。

**全部不可行时**（如 `--num-devices 8 --pd-ratio-sizes 8 16`）：不报错，记一条 `failures`：`pd-ratio: --num-devices=8 装不下任何 PD 组合（最小 P+D=16）；可增大 --num-devices 或减小 --pd-ratio-sizes`，并在 `summary.txt` 打印被剔除的组合清单。

### 5.3 下发的 msmodeling 命令

在**现有 PD 混部命令**之上追加三参数：

```
python -m cli.inference.throughput_optimizer <model_source> \
  --device <DEV> --num-devices <TOTAL> \
  --input-length <ISL> --output-length <OSL> \
  [--compile] \
  --quantize-linear-action ... --quantize-non-expert-linear-action ... --quantize-attention-action ... \
  --enable-optimize-prefill-decode-ratio \
  --prefill-devices-per-instance <P> \
  --decode-devices-per-instance <D> \
  --ttft-limit <TTFT> --tpot-limit <TPOT> \
  [--jobs N] [--max-search-combinations N] \
  [--speculative-method ... --num-speculative-tokens ... --acceptance-length ... --num-draft-layers ...]
```

- 无 `--disagg`、无 `--tp-sizes`、每个 `(P,D)` 只有**一条**命令。
- `ms_extra_flags()` 的投机参数条件由 `mode == disagg_decode` 改为 `mode in (disagg_decode, pd_ratio)`。

### 5.4 Runner 执行流程（`runners/msmodeling.py`）

```
total_steps = (1 if "agg" in serving_modes else 0) + len(pd_ratio_grid)

1. 若 "agg" in serving_modes: 跑 agg 子任务（总卡数，逻辑不变）
2. 若 "disagg" in serving_modes:
     for combo in pd_ratio_grid:
       a. 写 msmodeling/pd_ratio/<label>/command.txt
       b. args = _parse_args(argv)
       c. ParallelRunner(args).run_disagg()          # 内部走 _run_pd_ratio()
       d. summary = summaries[0]
       e. raw_df   = summary.get_summary_df()        # 公共 API：PD 交叉表
          best_row = summary.get_best_result_row()   # 公共 API：过滤+实例分配+排序后最优行
          cand_df  = _prepared_pd_ratio_df(summary)  # 见 5.5
       f. 落盘 raw/candidates/best
       g. cand_df 逐行 → PdRatioCandidate（含 §4.3 全部派生量）→ 汇入全局列表
       h. 构造 ServingModeResult(mode=disagg, label=<label>, pd_ratio={...})
3. 全局候选按 decode_tps_per_instance 降序 → 落盘 + 挂到 FrameworkRun
```

- `run.disagg_variants` 继续承载 PD-ratio 变体（**保留字段名**，`ui/app.js` 直接读它，前端零改动）。
- `run.disagg` = 变体中"集群端到端吞吐最优"者，供 `_build_recommendation` 与 agg 比较。

### 5.5 全部候选行的获取（方案 B · 已定）

```python
def _prepared_pd_ratio_df(summary) -> pd.DataFrame:
    prepare = getattr(summary, "_prepare_pd_ratio_results", None)   # 上游私有方法
    if callable(prepare):
        try:
            df = prepare()
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return summary.get_summary_df()        # 兜底：原始 PD 交叉表（缺 P/D 实例数列则留空）
```

`_prepare_pd_ratio_results()`（`optimizer_summary.py:641-665`）依次做：SLO 过滤 → `_apply_pd_instance_distribution()`（枚举实例数最大化 `balanced_qps`）→ 每组 `(parallel_p, parallel_d)` 保留最优 → 每个 `balanced_qps` 保留一条 → `rank_pd_ratio_rows()`。产出的行即 msmodeling 终端「Top N PD Ratio Configurations」表的每一行 —— **这正是"最全"的候选集**。

> 由于 msmodeling 是只读 whl，私有方法不会被上游重构打乱；再叠一层原始 df 兜底确保永不崩。

---

## 6. 需求 4：结果存储（沿用现有方案）

沿用 `<out-dir>/<run_id>/` 一套产物 + `run.json` 单一真相源，`SCHEMA_VERSION` 1.0 → **1.1**。

### 6.1 并行字段收窄（第 2 点答复）

```python
@dataclass
class ParallelConfig:
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    ep: int | None = None      # MoE 专家并行（aic-npu 的 ep / msmodeling 的 EP=）
    # moe_tp / moe_dp / dcp / pp_layer_partition 一律不进这里 → 落 Candidate.extra
```

`runners/aic_npu.py::_parallel()` 只读 `tp/pp/dp/ep`（aic-npu 的 `etp` → `extra.moe_tp`）；
`runners/msmodeling.py::parallel_from_ms_row()` 读 `tp_size/pp_size/dp_size/ep_size`（`MOE-TP` / `MOE-DP` / `dcp` → `extra`）。

### 6.2 新增数据模型（`schema.py`）

```python
@dataclass
class PdRatioSide:
    instances: int | None = None              # P/D 实例数
    devices_per_instance: int | None = None
    devices_total: int | None = None          # instances × devices_per_instance
    parallel: ParallelConfig | None = None    # 仅 tp/pp/dp/ep
    parallel_str: str | None = None
    batch_size: int | None = None
    concurrency: int | None = None
    qps_per_instance: float | None = None     # p_qps / d_qps
    qps_allocated: float | None = None        # allocated_p_qps / allocated_d_qps
    ttft_ms: float | None = None              # 仅 P 侧
    tpot_ms: float | None = None              # 仅 D 侧
    tokens_per_s_per_instance: float | None = None
    tokens_per_s_per_device: float | None = None
    tokens_per_s_cluster: float | None = None

@dataclass
class PdRatioCandidate:
    framework: str
    serving_mode: str = "disagg"
    mode: str = "pd_ratio"
    label: str = ""                       # "P8D8"
    pd_ratio: float | None = None
    prefill: PdRatioSide | None = None
    decode: PdRatioSide | None = None
    # —— 与 aic-npu 对齐的集群级指标 ——
    throughput_tokens_per_s: float | None = None            # balanced_qps × osl
    throughput_tokens_per_s_per_device: float | None = None # / devices_used
    tokens_per_s_per_user: float | None = None              # osl / request_latency_s
    throughput_req_per_s: float | None = None               # balanced_qps
    request_latency_ms: float | None = None
    devices_used: int | None = None
    num_devices: int | None = None
    meets_slo: bool | None = None
    # —— 排序键 ——
    decode_tps_per_instance: float | None = None            # 排序用
    decode_tps_cluster: float | None = None
    ttft_ms / tpot_ms
    quantize / system / backend
    extra / raw
```

### 6.3 容器增量

```python
@dataclass
class ServingModeResult:
    ...
    pd_ratio: dict | None = None            # 原 pd_balance 改名（语义改为每实例卡数 P/D）
    pd_ratio_candidates: list[dict] = []    # 该组合的候选，供 UI 展开

@dataclass
class FrameworkRun:
    ...
    disagg_variants: list[ServingModeResult] = []   # 字段名保留，承载 PD-ratio 变体
    pd_ratio_candidates: list[dict] = []            # 跨组合合并 + 已排序（msmodeling 专有）
```

`ui/app.js` 与 `scripts/build_results_index.py` 无需改动。

### 6.4 产物布局

```
<out>/<run_id>/
  request.json
  run.json                                  # schema_version = "1.1"
  summary.txt
  logs/*.log
  aic_npu/{agg,disagg}/best_config_topn.csv  # aic-npu 分支（不变）
  msmodeling/
    agg/{command.txt,summary.csv,candidates.json}          # 不变
    pd_ratio/
      P8D8/{command.txt, pd_ratio_raw.csv, pd_ratio_candidates.csv, pd_ratio_best.json}
      P8D16/…  P16D8/…                                      # 按 --pd-ratio-sizes 展开
      all_candidates.csv      # 跨组合合并 + 按 decode 单实例吞吐降序
      all_candidates.json     # 同上，供 HTTP / UI 直接消费
```

### 6.5 API 层（`api/schemas.py`）

```python
serving_modes: list[str] = Field(default=["agg", "disagg"])   # 新增
pd_ratio_sizes: list[int] = Field(default=[8])                # 替换
# 删除 pd_balance_search / pd_balance_max / pd_balance_sizes
```

### 6.6 终端输出（`output.py`）

`render_summary()` 的 disagg 段落改为：每个 PD-ratio 变体一行标签 `P8D8（P 侧 1 实例×8 卡，D 侧 3 实例×8 卡，用卡 32/32）`，末尾追加全局排序候选表：

| Rank | P×D | P inst | D inst | P dev/inst | D dev/inst | P Parallel | D Parallel | P bs | D bs | P conc | D conc | TTFT | TPOT | **decode tps/inst** | decode tps/cluster | cluster tokens/s | tokens/s/dev | req/s | devices |

---

## 7. 需求 5：aic-npu ↔ msmodeling 字段对齐

### 7.1 对齐表（✅ 可对齐）

| 概念 | aic-npu（`disagg` 表列） | msmodeling PD-ratio | 统一字段 |
|---|---|---|---|
| P 实例数 | `(p)workers` | `P Instances` | `prefill.instances` |
| D 实例数 | `(d)workers` | `D Instances` | `decode.instances` |
| 单 P/D 实例卡数 | `(p)/(d)gpus/worker` | `P/D Devices/Instance` | `…devices_per_instance` |
| P/D 侧总卡数 | `workers × gpus/worker` | `Instances × Devices/Instance` | `…devices_total` |
| 集群用卡 | `total_gpus (used)` | `Devices Used` | `devices_used` |
| **TP** | `parallel` 中的 `tp4` | `TP=4` | `parallel.tp` |
| **PP** | `pp1` | `PP=1` | `parallel.pp` |
| **DP** | `dp16` | `DP=16` | `parallel.dp` |
| **EP** | `ep64` | `EP=64` | `parallel.ep` |
| P/D 批大小 | `(p)/(d)bs` | `P/D Batch Size` | `…batch_size` |
| P/D 并发 | `concurrency`（聚合值） | `P/D Concurrency`（单实例值） | `…concurrency` |
| 端到端 QPS | `req/s` | `Balanced QPS` | `throughput_req_per_s` |
| TTFT | `TTFT` | `TTFT (ms)` = `ttft_p` | `ttft_ms` |
| TPOT | `TPOT` | `TPOT (ms)` = `tpot_d` | `tpot_ms` |
| 请求时延 | `request_latency` | 派生 `ttft + tpot×(osl-1)` | `request_latency_ms` |
| 集群 token 吞吐 | `tokens/s` | 派生 `balanced_qps × osl` | `throughput_tokens_per_s` |
| 单卡 token 吞吐 | `tokens/s/gpu` | 派生 `…/ devices_used` | `throughput_tokens_per_s_per_device` |
| 每用户吞吐 | `tokens/s/user` | 派生 `osl / request_latency` | `tokens_per_s_per_user` |

### 7.2 非对齐项（一律进 `extra` / `raw`）

**并行维度（除 tp/dp/pp/ep 外）**：`moe_tp`（aic-npu `etp` / msmodeling `MOE-TP`）、`moe_dp`（`MOE-DP`）、`dcp`（`DCP`）、`pp_layer_partition`、`pp_schedule`、`pp_makespan_ms`、`pp_bubble_ratio`。

| 仅 aic-npu 有 | 仅 msmodeling PD-ratio 有 |
|---|---|
| `replicas`、`gpus/replica`（xPyD 最小扩缩单元） | `PD Ratio`（配比本身） |
| `tokens/s/gpu_cluster` | `P/D QPS/Instance` |
| `power_w` | `Allocated P QPS`、`Allocated D QPS` |
| `(p)/(d)seq/s/worker` | `max_batched_tokens`、`batch_range`、`concurrency_search_strategy` |
| `backend-version`（`0.23.0`）、`database_mode`（HYBRID） | 投机解码参数（dflash/dspark/mtp） |
| `isl/osl` 作为求解输入 | `prefix_cache_hit_rate` |

### 7.3 三处语义陷阱（必须在汇总时标注）

1. **并发口径**：aic-npu 的 `concurrency` 是**集群聚合值**（`640 (=640x1)`），msmodeling 的 `P/D Concurrency` 是**单实例值**，对齐需 `× instances`。
2. **用卡量**：aic-npu 恒好用满 `total_gpus`；msmodeling 的 `Devices Used` 因实例数是整数**可能 < 总卡数**（你的样例恰好 32/32，换参数会有零头）。
3. **EP 语义**：msmodeling 的 `EP` 就是 MoE 专家并行（对应 aic-npu 的 `ep`），而其 `MOE-TP` / `MOE-DP` 是**另一套** MoE 切分维度，不属于统一并行字段。

---

## 8. 改动计划（共 12 步）—— **已全部完成**

已完成的工作：1~12 全部落地。

---

## 9. 真机验证（venv + msmodeling-0.2.0 whl，已通过）

环境：项目 `.venv`（Python 3.12.11 / torch 2.10.0+cpu），
msmodeling 由 `msmodeling_offline_installer/msmodeling-0.2.0-py3-none-any.whl`
`pip install --no-deps --force-reinstall` 装入。

### 9.1 关键 API 全部存在

`ParallelRunner.run_disagg` / `_run_pd_ratio`、`OptimizerSummary.get_summary_df` /
`get_best_result_row` / **`_prepare_pd_ratio_results`** —— 方案 B 依赖的私有方法在 whl 中可用。

### 9.2 PD 结果表列名与预期完全一致

`_prepare_pd_ratio_results()` 产出的 48 列里，整合层实际消费的关键列全部命中：

```
pd_ratio  p_qps  d_qps  balanced_qps  ttft_p  tpot_d
parallel_p  parallel_d  num_devices_p  num_devices_d
batch_size_p  batch_size_d  concurrency_p  concurrency_d
p_instances  d_instances  allocated_p_qps  allocated_d_qps  allocated_devices
```

PD 结果表**确实不含 `token/s_d`**（与 §4 的判断一致），吞吐只能由 `concurrency_d/tpot_d`
复算 —— 验证通过。

### 9.3 实测结果（GLM-5.2 / ATLAS_950_486T_96G / 32 卡 / MXFP4+FP8 / P8D8）

| 项 | 实测 |
|---|---|
| `get_summary_df()` 原始 PD 交叉表 | **(126, 42)** |
| `_prepare_pd_ratio_results()` 候选表 | **(11, 48)** → 落盘 10 行（去重后） |
| 最优行 | `pd_ratio=1.002`、`p_inst=2`、`d_inst=2`、`conc_d=492`、`tpot_d=41.39` |
| `decode_tps_per_instance` | 492 / 41.39 × 1000 = **11886.9** |
| `decode_tps_cluster` | × 2 = 23773.9 |
| `throughput_tokens_per_s` | 23.1907 × 1024 = **23747.3**（受 `balanced_qps` 封顶） |
| `devices_used / num_devices` | 32 / 32 |

P/D/overall 三个视角均取自**同一行**，`_variant_headline` 渲染为
`[P8D8] P 侧 2 实例 × 8 卡（共 16 卡） | D 侧 2 实例 × 8 卡（共 16 卡）`。

### 9.4 多组合 + 排序（`--pd-ratio-sizes 8 16`）

4 组组合（P8D8 / P8D16 / P16D8 / P16D16）各跑一次，合并得 **32 条候选**，
`all_candidates.csv` 按 `decode_tps_per_instance` 降序输出，实测
`is_monotonic_decreasing = True`。

### 9.5 两个"无结果"场景（均不报错，只记 failure）

| 场景 | 行为 |
|---|---|
| 大模型 + 8 卡/实例 + 不量化 | GLM-5.2 每实例 8 卡放不下 → `summaries=[]` → 记录可操作提示（建议增大 `--pd-ratio-sizes` 或开量化） |
| `--num-devices 8` + `--pd-ratio-sizes 8 16` | 组合数 0 → 提示"装不下任何 PD 组合（最小 P+D=16）" |

### 9.6 两个版本的差异（重要）

whl 里 `throughput_optimizer.arg_parse()` 是**零参**的（内部读 `sys.argv`，
且 `--device` 是 `nargs="+"` → `args.device` 为 **list**）。
整合层已有两处对应处理，验证确认必要且有效：

1. `_parse_args()` 临时改写 `sys.argv` 再调用 `arg_parse()`；
2. `args.device = request.device`（改回字符串，否则 `ParallelRunner.__init__`
   会以 list 做 dict key 抛 `TypeError: unhashable type: 'list'`）。

### 9.7 耗时参考

| 场景 | 耗时 |
|---|---|
| 单组 PD-ratio（MXFP4，32 卡，8 进程） | **177 s** |
| 4 组（sizes 8 16） | ≈ 12 min |

> 因此默认保持 `--pd-ratio-sizes 8`（1 组）是必要的；
> `8 16 32`（9 组）预计 ≈ 27 min，与 §3.1 的成本提示一致。

验证产物保留在 `_verify_results/20261027-114626-...-9639c54a/`（可自行删除）。

| # | 文件 | 内容 |
|---|---|---|
| 1 | `models.py` | 新增 `deepseek-ai/DeepSeek-V4.1-Flash`（msmodeling-only） |
| 2 | `config.py` | 删 `MS_TP_SIZES_*`、`PD_BALANCE_*`、`get_pd_balance_granularity()`；`SCHEMA_VERSION="1.1"`；新增 `MODE_PD_RATIO`、`PD_RATIO_SIZES_DEFAULT=(8,)`、`SERVING_MODES_DEFAULT=("agg","disagg")` |
| 3 | `schema.py` | `ParallelConfig` 收窄为 tp/pp/dp/ep；删 `pd_balance_*`；新增 `PdRatioSide`/`PdRatioCandidate`；`ServingModeResult.pd_balance`→`pd_ratio` + `pd_ratio_candidates`；`FrameworkRun.pd_ratio_candidates`；`RecommendRequest.pd_balance_*` → `pd_ratio_sizes` + `serving_modes` |
| 4 | **新 `pd_ratio_metrics.py`** | PD 行 → 派生吞吐的全部公式（与 msmodeling 同源，含注释出处） |
| 5 | `routing.py` | 删 TP 起跳与 xPyD 旧逻辑；新增 `PdRatioCombo`/`pd_ratio_grid()`；`tp_size_flags()` 空则不下发；`slo_flags()` 增 `pd_ratio` 分支；投机参数覆盖 `pd_ratio`；新增 `build_msmodeling_command(mode=pd_ratio, p, d)` |
| 6 | `runners/msmodeling.py` | 新增 `_run_pd_ratio_combo()`、`_prepared_pd_ratio_df()`、`_pd_ratio_candidate_from_row()`；删 `_pd_grid()`/`_run_disagg_variant()`；进度按 `serving_modes` 计算；并行字段改名 |
| 7 | `runners/aic_npu.py` | `_parallel()` 只读 tp/pp/dp/ep；`etp` 转 `extra.moe_tp`；按 `serving_modes` 过滤输出并记录说明 |
| 8 | `cli.py` | 删 4 个旧 PD 参数 + `pd_balance_label()` + 前置校验；新增 `--serving-modes`、`--pd-ratio-sizes`；`--tp-sizes` help 更新 |
| 9 | `output.py` | `render_summary()` 改为 PD-ratio 变体 + 全局排序候选表 + 被剔除组合提示 |
| 10 | `api/schemas.py` | `serving_modes` + `pd_ratio_sizes` |
| 11 | `scripts/run_regression_*.sh`（4 个） | PD 参数注入改名 |
| 12 | `docs/plans/*.md` | 用户指南（3 份）+ 设计文档 + HTTP API 文档同步 |

### 验证命令

```bash
# msmodeling + PD-ratio 默认（1 组合，只跑 PD 分离）
infer-strategy-npu -recommend --model ZhipuAI/GLM-5.2 --device ATLAS_950_486T_96G \
  --num-devices 32 --input-length 4096 --output-length 1024 \
  --ttft-limit 2000 --tpot-limit 80 --serving-modes disagg

# 多组合 + 干跑校验命令拼装
infer-strategy-npu -recommend --model ZhipuAI/GLM-5.2 --device ATLAS_950_486T_96G \
  --num-devices 128 --pd-ratio-sizes 8 16 32 --dry-run

# 组合全部不可行：只记 failure，不报错
infer-strategy-npu -recommend --model ZhipuAI/GLM-5.2 --device ATLAS_950_486T_96G \
  --num-devices 8 --pd-ratio-sizes 8 16

# aic-npu 分支不受影响
infer-strategy-npu -recommend --model deepseek-ai/DeepSeek-R1 \
  --device ATLAS_800_A2_376T_64G --num-devices 128 --quantize DISABLE
```
