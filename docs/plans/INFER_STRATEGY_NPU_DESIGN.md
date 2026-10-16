# infer-strategy-npu 统一 CLI 设计文档

> 分支：`feature/infer-strategy-npu-unified-cli`（自 `check/model-adaptation-audit` 切出）
> 目标：在 aic-npu 容器内整合 `aic-npu` 与 `msmodeling` 两套推理部署策略搜索框架，
> 对外暴露唯一入口 `infer-strategy-npu`，并产出统一结构化的 `agg / disagg(prefill/decode)` 结果，
> 为后续 HTTP 服务封装留出纯函数接口。

---

## 1. 背景与现状

两套框架能力重叠但输入输出完全不同：

| 维度 | aic-npu（aiconfigurator-npu） | msmodeling（源码构建 whl 0.2.0） |
|---|---|---|
| 入口 | `aic-npu default ...`（console script `aiconfigurator_npu.cli.main:main`） | `python -m cli.inference.throughput_optimizer ...` |
| Python API | `aiconfigurator_npu.cli.api.cli_default(...) -> CLIResult` | `serving_cast.parallel_runner.ParallelRunner(args).run_agg()/run_disagg() -> list[OptimizerSummary]` |
| 模型来源 | `model_configs/<org>--<name>_config.json`（离线） | ModelScope/HF id 或本地目录（含 `config.json`） |
| 硬件描述 | `--system ascend_910b`（仅 1 套性能库） | `--device ATLAS_*`（26 种 DeviceProfile） |
| 结果落盘 | `--save-dir` → `<随机目录>/{agg,disagg}/best_config_topn.csv` + `pareto.csv` + `exp_config.yaml` | 无落盘，仅 stdout 打印；唯一文件产物是 `--chrome-trace-file` |
| 排序指标 | `tokens/s/gpu_cluster` | `token/s`（best throughput） |

结论：aic-npu 有文件产物但目录名带随机后缀；msmodeling 有结构化对象但无落盘。
统一层必须**同时**做两件事：① 归一化输入参数与路由；② 归一化输出结果并强制落盘。

---

## 2. 统一命令接口

```
infer-strategy-npu -recommend
    --model <ModelScope ID>
    --device <设备名>
    --num-devices <N>
    [--input-length <int>]      # 默认 1024，>0
    [--output-length <int>]     # 默认 1024，>0
    [--tpot-limit <int>]        # 默认 50，>0
    [--ttft-limit <int>]        # 默认 1000，>0
    [--quantize <DISABLE|W8A8>] # 默认 DISABLE
    [--out-dir <路径>]           # 默认 /workspace/results/infer-strategy-npu
    [--top-n <int>]             # 默认 10（透传给 aic-npu --top-n）
    [--framework auto|aic-npu|msmodeling]  # 默认 auto，强制覆盖路由（排障用）
    [--local-model-root <路径>]  # msmodeling 本地模型根目录，默认取环境变量 INFER_STRATEGY_LOCAL_MODEL_ROOT
    [--quantize-linear-action <枚举>]       # 覆盖 msmodeling --quantize-linear-action
    [--quantize-attention-action <枚举>]    # 覆盖 msmodeling --quantize-attention-action
    [--speculative-method none|dflash|dspark|mtp]   # 默认 dflash（PD分离-Decode）
    [--num-speculative-tokens N] [--acceptance-length F] [--num-draft-layers N]
    [--jobs N] [--max-search-combinations N]        # 透传 msmodeling
    [--dry-run]                  # 只打印将要下发的两套框架命令，不执行
    [--quiet] [--verbose]
```

参数取值校验：

- `--model`：白名单 6 个 ModelScope ID（大小写不敏感，支持别名）：
  `ZhipuAI/GLM-5.3-Flash`、`Qwen/Qwen3.8-27B`、`deepseek-ai/DeepSeek-R1`、
  `deepseek-ai/DeepSeek-V4-Pro`、`moonshotai/Kimi-K3`、`MiniMax/MiniMax-M3`。
- `--device`：26 个，取自 `tensor_cast/device.py` 的 `DeviceProfile` 注册名（见 §4）。
- `--num-devices` / `--input-length` / `--output-length` / `--tpot-limit` / `--ttft-limit`：正整数（>0）。
- `--quantize`：`DISABLE` / `W8A8`（大小写与连字符不敏感，归一化为 `DISABLE` / `W8A8`）。

---

## 3. 路由规则

```
model ∈ {DeepSeek-R1, Qwen3.8-27B, DeepSeek-V4-Pro} 且 quantize == DISABLE
    → aic-npu

其余全部
    ├─ model ∈ {DeepSeek-R1, Qwen3.8-27B, DeepSeek-V4-Pro} 且 quantize == W8A8
    └─ model ∈ {Kimi-K3, GLM-5.3-Flash, MiniMax-M3}（quantize 任意）
    → msmodeling
```

aic-npu 分支额外约束：

- 设备必须是 `ATLAS_800_A2_376T_64G`（唯一映射到 `--system ascend_910b`）。
  其它设备直接报错并提示改用 msmodeling 可支持的设备。
- 下发命令（与现有用法保持一致）：

```
aic-npu default \
  --model-path /workspace/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json \
  --total-gpus 128 --system ascend_910b \
  --backend vllm-ascend --backend-version 0.23.0 --database-mode HYBRID \
  --isl 4096 --osl 1024 --ttft 30000 --tpot 3000 \
  --top-n 10 --save-dir <run>/aic_npu
```

  - `--ttft` / `--tpot` 由 `--ttft-limit` / `--tpot-limit` 直接映射（aic-npu 的 ttft/tpot 是 SLO 约束，语义一致）。
  - `--backend` 必须显式给 `vllm-ascend`（默认 `trtllm` 在 ascend 上必然失败）。
  - 实现上通过 `cli_default()` Python API 调用，等价命令会记入 `run.json` 便于复现。

msmodeling 分支一次运行拆成 3 个子任务（对应统一结果的 3 个字段）：

```
# 1) agg（PD 混部）→ results.agg
python -m cli.inference.throughput_optimizer <model> \
  --device ATLAS_800_A2_376T_64G --num-devices 128 \
  --input-length 4096 --output-length 2048 --compile \
  --quantize-linear-action DISABLED --quantize-attention-action DISABLED \
  --tpot-limit 50 --ttft-limit 1000

# 2) disagg_prefill（PD分离-Prefill）→ results.disagg.prefill
python -m cli.inference.throughput_optimizer <model> \
  --device ATLAS_800_A2_376T_64G --num-devices 128 \
  --input-length 4096 --output-length 2048 --compile \
  --quantize-linear-action W8A8_DYNAMIC --quantize-attention-action INT8 \
  --disagg --ttft-limit 1000

# 3) disagg_decode（PD分离-Decode）→ results.disagg.decode
python -m cli.inference.throughput_optimizer <model> \
  --device ATLAS_800_A2_376T_64G --num-devices 128 \
  --input-length 4096 --output-length 2048 --compile \
  --quantize-linear-action W8A8_DYNAMIC --quantize-attention-action INT8 \
  --disagg --tpot-limit 50 \
  --speculative-method dflash --num-speculative-tokens 7 --acceptance-length 5 --num-draft-layers 6
```

SLO 参数口径（依据 `docs/zh/user_guide/msmodeling_throughput_optimizer_user_guide.md` 第 2 节）：

| 子任务 | 下发的 SLO 参数 |
|---|---|
| agg（PD 混部） | `--tpot-limit` **和** `--ttft-limit`（文档列出的 PD 混部关键参数就是这两个；也与 aic-npu 同时传 `--ttft/--tpot` 的语义一致） |
| disagg_prefill（PD分离-Prefill） | `--disagg --ttft-limit` |
| disagg_decode（PD分离-Decode） | `--disagg --tpot-limit` |

`--quantize` → msmodeling 动作映射：

| `--quantize` | `--quantize-linear-action` | `--quantize-attention-action` |
|---|---|---|
| `DISABLE` | `DISABLED` | `DISABLED` |
| `W8A8` | `W8A8_DYNAMIC` | `INT8` |

要点与文档一致性核对（已逐条对照用户指南）：

1. `--quantize-linear-action` 的**默认值本身就是 `W8A8_DYNAMIC`**，所以 `quantize=DISABLE` 时必须显式下发 `DISABLED`，否则会被静默当成 W8A8。
2. `--quantize-attention-action` 语义是 **KV cache 量化**（choices `DISABLED|INT8|FP8`）。`W8A8 → INT8` 对齐你给的 PD分离参考命令；文档多数示例用的是 `disabled`。要改口径可显式传 `--quantize-attention-action DISABLED`。
3. 量化取值大小写与下划线/连字符写法均可，本层统一以**大写下划线**形式下发（`W8A8_DYNAMIC` / `INT8` / `DISABLED`）。
4. `--num-draft-layers` **仅适用于 `dflash` / `dspark`，不可与 `mtp` 同用**（文档 2.2 节明确约束），代码已按此判断——选 `mtp` 时不下发该参数。
5. `--num-speculative-tokens` 任一候选为 `0` 会解析失败；`--speculative-method` 只作用于 PD分离-Decode（投机解码影响 Decode）。
6. 两个量化动作都提供了直通覆盖：`--quantize-linear-action` / `--quantize-attention-action`。

> 说明：msmodeling 侧用 `ParallelRunner` Python API 直接调用（而非 subprocess），
> 以便拿到 `OptimizerSummary` 结构化对象并落盘；等价命令同样记入 `run.json`。
> 由于 `arg_parse()` 内部有 tp/ep/dcp 候选归一化与校验，实现上复用
> `build_argparser(get_spec("throughput_optimizer")) + parse_module_args(spec, parser, argv)`
> 走同一套声明式 parser，避免自建 Namespace 缺字段。

---

## 4. 统一设备清单（`--device`）

取自 `msmodeling/tensor_cast/device.py` 中注册的 `DeviceProfile`（26 个）：

```
TEST_DEVICE
ATLAS_800_A2_376T_64G        ATLAS_800_A2_313T_64G        ATLAS_800_A2_280T_64G
ATLAS_800_A2_280T_64G_PCIE   ATLAS_800_A2_280T_32G_PCIE
ATLAS_800_A3_752T_128G_DIE   ATLAS_800_A3_560T_128G_DIE   ATLAS_800_A3_560T_128G_DIE_ROCE
ATLAS_350_425T_112G          ATLAS_350_425T_84G
ATLAS_850_486T_112G          ATLAS_850_486T_128G          ATLAS_850E_486T_96G
ATLAS_850E_425T_96G          ATLAS_850_486T_112G_ROCE     ATLAS_850_486T_128G_ROCE
ATLAS_850E_486T_96G_ROCE     ATLAS_850E_425T_96G_ROCE     ATLAS_850_486T_112G_FM16
ATLAS_850_486T_128G_FM16     ATLAS_850E_486T_96G_FM16     ATLAS_850E_425T_96G_FM16
ATLAS_950_486T_128G          ATLAS_950_486T_96G           ATLAS_950_486T_144G
```

到 aic-npu `--system` 的映射（其余设备走 msmodeling）：

| `--device` | aic-npu `--system` |
|---|---|
| `ATLAS_800_A2_376T_64G` | `ascend_910b` |
| 其它 | 不支持（报错） |

---

## 5. 统一模型注册表

| 统一模型名 | aic-npu `--model-path` | msmodeling 本地目录（**相对路径**） | 回退 model-id |
|---|---|---|---|
| `deepseek-ai/DeepSeek-R1` | `/workspace/model_configs/deepseek-ai--DeepSeek-R1_config.json` | `model_configs/msmodeling/DeepSeek-R1` | `deepseek-ai/DeepSeek-R1` |
| `Qwen/Qwen3.8-27B` | `/workspace/model_configs/qwen3_8_27B_config.json` | `model_configs/msmodeling/Qwen3.8-27B` | `Qwen/Qwen3.8-27B` |
| `deepseek-ai/DeepSeek-V4-Pro` | `/workspace/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json` | `model_configs/msmodeling/DeepSeek-V4-Pro` | `deepseek-ai/DeepSeek-V4-Pro` |
| `ZhipuAI/GLM-5.3-Flash` | — | `model_configs/msmodeling/GLM5.3-Flash` | `ZhipuAI/GLM-5.3-Flash` |
| `moonshotai/Kimi-K3` | — | `model_configs/msmodeling/Kimi-K3` | `moonshotai/Kimi-K3` |
| `MiniMax/MiniMax-M3` | — | `model_configs/msmodeling/MiniMax-M3` | `MiniMax/MiniMax-M3` |

**msmodeling 的二级模型目录**（已建，容器内即 `/workspace/model_configs/msmodeling/`）：

```
model_configs/
├── deepseek-ai--DeepSeek-R1_config.json        # aic-npu 用（HF id -> <org>--<name>_config.json）
├── qwen3_8_27B_config.json
├── deepseek-ai--DeepSeek-V4-Pro_config.json
└── msmodeling/                                  # msmodeling 用（每个模型一个目录 + config.json）
    ├── DeepSeek-R1/config.json                  # 复用 aic-npu 的真实配置
    ├── DeepSeek-V4-Pro/config.json
    ├── Qwen3.8-27B/config.json
    ├── GLM5.3-Flash/config.json                 # 占位，待填
    ├── Kimi-K3/config.json                      # 占位，待填
    └── MiniMax-M3/config.json                   # 占位，待填
```

目录约定与填充说明见 `model_configs/msmodeling/README.md`。
aic-npu 只扫描 `model_configs/*.json`（`<org>--<name>_config.json`），新增的二级目录不会干扰它。

**msmodeling 默认使用相对路径**（`model_configs/msmodeling/<目录>`，相对 CWD；容器 CWD 为 `/workspace`）：

- 便于在容器内外、不同机器上保持同一份命令与 `run.json` 可复现；
- 不会把宿主机绝对路径（`D:\...`、`/home/...`）写进结果文件；
- 需要指向别处时用 `--local-model-root <路径>` 或环境变量 `INFER_STRATEGY_LOCAL_MODEL_ROOT` 覆盖。

模型解析优先级（msmodeling 侧）：

1. `<local_model_root>/<本地目录名>` 存在 → 用该**相对路径**（默认 `model_configs/msmodeling/<目录>`，避免 HF 拉取超时）
2. `--model` 传的是已存在的本地目录 → 直接用
3. 否则用 ModelScope ID（走网络；建议设 `HF_ENDPOINT` 镜像或 `HF_HUB_OFFLINE=1` + 本地目录）

环境变量 `INFER_STRATEGY_LOCAL_MODEL_ROOT` 可覆盖根目录；默认值为相对路径 `model_configs/msmodeling`。

---

## 6. 统一结果 Schema（`run.json`，schema_version = 1.0）

```jsonc
{
  "schema_version": "1.0",
  "run_id": "20261016-153000-Qwen-Qwen3.8-27B-ATLAS_800_A2_376T_64G-128",
  "generated_at": "2026-10-16T15:30:00+08:00",
  "status": "ok",                       // ok | partial | failed
  "request": {
    "mode": "recommend",
    "model": "Qwen/Qwen3.8-27B",
    "device": "ATLAS_800_A2_376T_64G",
    "num_devices": 128,
    "input_length": 1024,
    "output_length": 1024,
    "tpot_limit": 50,
    "ttft_limit": 1000,
    "quantize": "DISABLE"
  },
  "routing": {
    "framework": "aic-npu",             // 命中的框架
    "reason": "model in aic-npu allowlist and quantize=DISABLE",
    "commands": ["aic-npu default --model-path ... --total-gpus 128 ..."]
  },
  "frameworks": [
    {
      "framework": "aic-npu",
      "status": "ok",                   // ok | skipped | failed
      "message": null,
      "artifacts": { "root": ".../aic_npu", "files": [".../agg/best_config_topn.csv"] },
      "agg":     { "available": true, "best": {CANDIDATE}, "candidates": [CANDIDATE, ...] },
      "disagg":  {
        "available": true,
        "prefill": {CANDIDATE},         // 与 aic-npu disagg 的 (p)* 列对应
        "decode":  {CANDIDATE},         // 与 aic-npu disagg 的 (d)* 列对应
        "best":    {CANDIDATE},         // 整条 disagg 方案（tokens/s/gpu_cluster 最高者）
        "candidates": [CANDIDATE, ...]
      }
    }
  ],
  "recommendation": {
    "framework": "aic-npu",
    "serving_mode": "disagg",           // agg | disagg
    "mode": "disagg",                   // agg | disagg_prefill | disagg_decode
    "best": {CANDIDATE},
    "reason": "highest throughput_tokens_per_s among all candidates"
  },
  "errors": []
}
```

### CANDIDATE（归一化候选记录）

```jsonc
{
  "framework": "aic-npu",               // aic-npu | msmodeling
  "serving_mode": "disagg",             // agg | disagg
  "mode": "disagg_decode",              // agg | disagg_prefill | disagg_decode
  "throughput_tokens_per_s": 1234.56,   // 集群总吞吐 token/s
  "throughput_tokens_per_s_per_device": 9.64,
  "throughput_req_per_s": 1.2,          // seq/s 或 QPS(req/s)，不可得为 null
  "ttft_ms": 1234.5,
  "tpot_ms": 45.6,
  "meets_slo": true,                    // ttft<=ttft_limit && tpot<=tpot_limit
  "concurrency": 128,
  "batch_size": 128,
  "num_devices": 128,
  "parallel": { "tp": 4, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 1 },
  "parallel_str": "tp4pp1dp1etp1ep1",
  "quantize": { "linear": "DISABLED", "attention": "DISABLED" },
  "backend": "vllm-ascend",             // msmodeling 侧为 "msmodeling"
  "backend_version": "0.23.0",
  "system": "ascend_910b",              // msmodeling 侧为 device 名
  "raw": { }                            // 框架原始行，便于回溯
}
```

### 字段对齐规则

| 统一字段 | aic-npu agg | aic-npu disagg | msmodeling |
|---|---|---|---|
| `throughput_tokens_per_s` | `tokens/s` | `tokens/s` | `token/s` |
| `throughput_tokens_per_s_per_device` | `tokens/s/gpu` | `tokens/s/gpu` | `token/s/device` |
| `throughput_req_per_s` | `seq/s` | `seq/s` | `qps_req_s`（QPS req/s） |
| `ttft_ms` | `ttft` | `ttft` | `ttft` |
| `tpot_ms` | `tpot` | `tpot` | `tpot` |
| `concurrency` | `concurrency` | `concurrency` | `concurrency` |
| `batch_size` | `bs` | `(p)bs`/`(d)bs` | `batch_size` |
| `parallel` | `tp/pp/dp/moe_tp/moe_ep` | `(p)tp…/(d)tp…` | 由 `parallel` 串解析 `TP=x|PP=y|DP=z` |
| `system` | `system` | `(p)system`/`(d)system` | `device` |

- aic-npu 的 `disagg` 行同时含 prefill 与 decode 列：`best` 取该行整体；
  `prefill` / `decode` 分别用 `(p)*` / `(d)*` 列构造（吞吐分别用 `(p)seq/s/worker`、`(d)seq/s/worker` 折算）。
- msmodeling 的 `disagg_prefill` 与 `disagg_decode` 是两次独立运行，各自产出一行 best；
  `disagg.best` 取 decode 行（端到端吞吐口径），`agg.best` 取 agg 行。

---

## 7. 产物文件布局

```
<out-dir>/<run_id>/
├── run.json                       # 统一结构化结果（主产物，HTTP 直接返回这个）
├── request.json                   # 归一化后的请求
├── summary.txt                    # 人类可读摘要（同终端输出）
├── aic_npu/                       # aic-npu 原始产物（save_dir）
│   └── <model>_ascend_910b_vllm-ascend_isl*_osl*_ttft*_tpot*_<rand>/
│       ├── pareto_frontier.png
│       ├── agg/{best_config_topn.csv,pareto.csv,exp_config.yaml,topN/generator_config.yaml}
│       └── disagg/{...}
└── msmodeling/
    ├── agg/{summary.csv,summary.json,command.txt}
    ├── disagg_prefill/{summary.csv,summary.json,command.txt}
    └── disagg_decode/{summary.csv,summary.json,command.txt}
```

- aic-npu 目录名含随机后缀：实现上先创建空目录，运行后取其下唯一子目录（`glob` + mtime 兜底）。
- msmodeling 无落盘能力：直接把 `OptimizerSummary.get_summary_df()` 写 `summary.csv`，
  把 best 行写 `summary.json`，并额外落一份 `candidates.json`（Top-N）。

---

## 8. 代码结构

新增顶层包 `src/infer_strategy_npu/`（不动现有 `aiconfigurator_npu` 包）：

```
src/infer_strategy_npu/
├── __init__.py          # 版本 + 导出 recommend()
├── cli.py               # argparse 定义 + main()（console script 入口）
├── config.py            # 默认常量、环境变量读取、路径解析
├── devices.py           # 26 个设备常量 + device→aic-npu system 映射
├── models.py            # 6 个模型注册表 + model→(aic-npu path / msmodeling id / 本地目录) 映射
├── schema.py            # RecommendRequest / Candidate / FrameworkRun / UnifiedRunResult dataclass ↔ dict
├── routing.py           # 路由决策（返回 framework + reason + 命令清单）
├── service.py           # recommend(request) -> UnifiedRunResult（纯函数，HTTP 复用）
├── output.py            # 落盘 writer（run.json/csv/json）、终端表格渲染、结果读取/解析
└── runners/
    ├── __init__.py
    ├── base.py          # FrameworkRunner 抽象基类
    ├── aic_npu.py       # 调 cli_default() 并解析 agg/disagg CSV
    └── msmodeling.py    # 构建 argv → arg_parse → ParallelRunner → OptimizerSummary
```

`pyproject.toml` 增加：

```toml
[project.scripts]
aic-npu = "aiconfigurator_npu.cli.main:main"
infer-strategy-npu = "infer_strategy_npu.cli:main"
```

HTTP 预留：`service.recommend()` 不依赖 argparse / stdout，后续只需在其上包一层 FastAPI
（`POST /v1/recommend` 接收 JSON → `RecommendRequest` → 返回 `UnifiedRunResult.to_dict()`）。

---

## 9. 容器与安装

新增 `docker/Dockerfile.unified`（关键片段）：

```dockerfile
FROM quay.io/ascend/vllm-ascend:v0.23.0.post1
WORKDIR /workspace
COPY collector/ ./collector/
COPY tools/ ./tools/
COPY model_configs/ ./model_configs/
COPY src/ ./src/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# msmodeling 离线 whl（由源码构建，随仓库提交；*.whl 通配，换版本不用改这里）
# 构建期必须加 TORCH_DEVICE_BACKEND_AUTOLOAD=0，且**绝不能 import transformers / torch_npu**：
# msmodeling 导入链经过 transformers，transformers 导入时调 is_torch_npu_available()
# -> import torch_npu -> 加载 _C -> 需要宿主机昇腾驱动库 libascend_hal.so，docker build
# 环境没有它，直接 ImportError。因此构建期只校验：import torch（autoload 已关）、
# msmodeling 元数据版本、torch_npu 是否存在（find_spec，不真正导入）。
# 该变量仅作用于这条 RUN，不进 ENV，运行时在 NPU 主机上仍正常自动加载 torch_npu。
COPY msmodeling_offline_installer/*.whl ./msmodeling_offline_installer/
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "import importlib.util as u; [print(m + '==' + __import__(m).__version__) for m in ('torch', 'torchvision') if u.find_spec(m)]" > /tmp/ms-constraints.txt \
 && cat /tmp/ms-constraints.txt \
 && pip install --no-cache-dir -c /tmp/ms-constraints.txt ./msmodeling_offline_installer/*.whl \
 && rm -f /tmp/ms-constraints.txt \
 && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c "import torch, importlib.util as u, importlib.metadata as md
print('torch', torch.__version__)
print('msmodeling', md.version('msmodeling'))
print('torch_npu', 'present' if u.find_spec('torch_npu') else 'MISSING')
try:
    importlib.import_module('cli')
    print('cli import ok')
except Exception as e:
    print('cli import skipped (expected without NPU driver):', type(e).__name__, e)" \
 || (echo "msmodeling build-time check failed" && exit 1)

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/collector \
    INFER_STRATEGY_LOCAL_MODEL_ROOT=model_configs/msmodeling
RUN mkdir -p /workspace/data /workspace/local_models /workspace/results
CMD ["/bin/bash"]
```

> **已知冲突（可忽略，不影响本镜像用途）**：msmodeling whl 要求 `transformers>=5.16,<5.17`，
> 而基础镜像 `vllm-ascend 0.23.0.post1` 钉的是 `transformers==5.5.4`（外加 `fastapi<0.124`、
> `pandas~=2.2` 等）。`pip install <whl>` 会把 transformers / fastapi / pandas 升上去，
> pip 会打印 dependency-conflict 告警（如 `vllm-ascend ... requires transformers==5.5.4, but you
> have 5.16.1`）。**本镜像目标只是运行 aic-npu 与 msmodeling 两套 CLI，不通过 vllm 起服务**，
> 因此这个冲突只是告警、不会让构建失败，也不影响两条 CLI 路径；若日后要在同一镜像里跑 vllm-ascend
> 服务，需要单独处理这个版本矛盾（例如另起一个只含 vllm 的 stage）。
>

### msmodeling whl：必须由源码构建

**不要用官方发布的 `msmodeling-26.1.0-py3-none-any.whl`**：它比源码落后较多，
`cli/registry/`（声明式参数注册表 SPEC / argparse_adapter / validators）与 `cli/spec_cli.py`
**完全缺失**，装了之后 `import cli.registry` 直接失败，`arg_parse()` 跑不起来 —— 而统一 CLI 的
msmodeling Runner 正是走这条路径。

| | 官方 whl 26.1.0 | 源码构建 0.2.0 |
|---|---|---|
| `cli/` 文件数 | 8 | 20（含 `cli/registry/` 共 10 个文件） |
| `cli/registry/` | ❌ | ✅ |
| `cli/spec_cli.py` | ❌ | ✅ |
| `transformers` | `<5.8.0,>=5.6.0` | `>=5.16.0,<5.17.0`（与源码一致） |

构建命令（输出到本仓库的 `msmodeling_offline_installer/`，不在源码仓库留 `dist/`）：

```bash
cd <msmodeling 源码仓库>
uv build --wheel -o "<aic-npu 仓库>/msmodeling_offline_installer"
# 等价：python -m build --wheel --outdir <同上>
```

构建后端 hatchling，打包范围 `tensor_cast / serving_cast / cli / optix / web_ui`。
当前随仓库提交的是 `msmodeling-0.2.0-py3-none-any.whl`（源码分支
`feature/analyze-throughput-optimizer`，HEAD `44c1350c`）。

Dockerfile 用 `COPY msmodeling_offline_installer/*.whl` 通配安装，**换版本不用改 Dockerfile**；
CI 会在构建前校验 whl 存在且包含 `cli/registry/`（缺失直接 fail）。
重建步骤与自检见 `msmodeling_offline_installer/README.md`。

### CI

新增 `.github/workflows/build-image-unified.yml`（与既有 `build-image-dsv4.yml` 同构，手动 `workflow_dispatch`）：

- 入参：`image_tag`、`ref`（Git ref to build，默认 `feature/infer-strategy-npu-unified-cli`）
- **关键机制**：workflow 文件只需注册在 main（Actions UI 里可见），但通过 `ref` 输入
  `actions/checkout` 任意分支/tag/sha —— 即使 main 还没有合并 feature 分支，
  也能用该分支的完整构建上下文（`src/`、`model_configs/`、msmodeling whl）构建镜像
- 流程：checkout 指定 ref → **校验 whl 存在且含 `cli/registry/`**（缺失直接 fail）→ QEMU/Buildx
  → 登录 quay.io → `docker/build-push-action` 以 `docker/Dockerfile.unified` 构建并推送
  `quay.io/18896723947/aiconfigurator-npu-unified`
- 平台 `linux/arm64`，与既有镜像保持一致

### msmodeling 运行时依赖：本地 vs 容器

whl 的 METADATA **自带 `Requires-Dist`**（`torch<=2.10,>=2.8`、`torchvision>=0.23.0`、
`transformers>=5.16.0,<5.17.0`、`modelscope>=1.37.1`、`salabim`、`blinker`、`greenlet`、`optree`、
`compressed-tensors`、`scikit-learn`、`pyswarms`、`loguru`、`strenum`、`aenum`、`overrides`、
`colorama`、`openpyxl`、`psutil`、`pillow`、`matplotlib`、`plotext`、`prettytable`、`pyyaml`、
`einops`、`alembic`、`sqlmodel`、`tomlkit`，以及 web_ui 用的 `uvicorn`/`fastapi`）。
所以**两边都要装依赖**，但策略不同：

| | Windows 本地 | Docker（vllm-ascend 基础镜像） |
|---|---|---|
| 装法 | 先 `pip install torch --index-url .../cpu`，再 `pip install <whl>` 让 pip 自动解析其余 | `pip install -c <约束文件> <whl>` |
| torch | 必须显式装 CPU 版（Windows 无 GPU/CANN 版本） | **镜像自带带 NPU 补丁的 torch，绝不能被 PyPI 版本覆盖** |
| transformers | pip 自动装（5.6–5.8） | 镜像已装（vLLM 依赖），可能被 pip 升级 |
| 其余依赖 | pip 自动装 | 镜像基本没有，需要装 |
| 风险 | 手工 pin transformers 会与 whl 的 `<5.8.0` 冲突 | pip 若拉 PyPI 的 torch 会废掉 torch_npu / vllm-ascend |

Dockerfile 因此先生成约束把 `torch` / `torchvision` 钉在镜像自带版本再装 whl：
镜像 torch 满足区间 → 正常装；不满足 → 解析直接失败并给出明确报错，避免静默装成无 NPU 的 torch。

离线环境需配合内网 PyPI 镜像，或改用 `--no-deps` + 手工最小依赖清单
（见 Windows 调试手册第 3 节）。
- 本地模型目录通过 `-v <host>:/workspace/local_models` 挂载，镜像内已设
  `INFER_STRATEGY_LOCAL_MODEL_ROOT=/workspace/local_models`，无需再传参数。

---

## 10. 风险与开放问题

1. **msmodeling 落盘能力缺失**：只能靠 `OptimizerSummary` 的 DataFrame 自行序列化，
   若上游改动 `get_summary_df()` 列名需同步 `runners/msmodeling.py` 的字段映射。
2. **`--speculative-method dflash` 默认开启**：与用户给出的 PD分离-Decode 参考命令一致，
   但 `--num-draft-layers`/`--acceptance-length` 是模型相关超参，
   对不支持的模型可能报错 → 可用 `--speculative-method none` 关闭。
   已按文档约束：`--num-draft-layers` 只在 `dflash`/`dspark` 时下发（`mtp` 不允许）。
3. **`W8A8 → (W8A8_DYNAMIC, INT8)`**：`INT8` 是 **KV cache 量化**，与你给的 PD分离参考命令一致，
   而 msmodeling 用户指南多数示例用 `disabled`。若你的口径是不量化 KV cache，
   显式加 `--quantize-attention-action DISABLED` 即可。
4. **aic-npu 仅支持 1 种设备**：非 `ATLAS_800_A2_376T_64G` 的设备只能走 msmodeling，
   本层会在路由阶段直接给出明确报错而非静默降级。
5. **aic-npu 结果目录随机后缀**：通过"预建空目录 + 取唯一子目录"规避，若未来改为多实验并存需调整。
6. **两框架吞吐口径不完全一致**：aic-npu 排序用 `tokens/s/gpu_cluster`，msmodeling 用 `token/s`；
   统一记录两者，推荐时优先比较 `throughput_tokens_per_s`（集群口径），并在 `raw` 保留原始值。
7. **PD 混部会同时施加 TTFT 与 TPOT 约束**：agg 子任务下发 `--tpot-limit` + `--ttft-limit`，
   若 `--ttft-limit`（默认 1000ms）过紧，可能出现"没有配置满足 SLO"。需要放宽时调大 `--ttft-limit`。
8. **HTTP 封装本期不做**，仅预留 `service.recommend()` 与 `run.json` 契约。

---

## 11. 实现状态

代码已落在 `src/infer_strategy_npu/`，console script 注册在 `pyproject.toml`：

```toml
[project.scripts]
aic-npu = "aiconfigurator_npu.cli.main:main"
infer-strategy-npu = "infer_strategy_npu.cli:main"
```

| 模块 | 职责 | 状态 |
|---|---|---|
| `cli.py` | argparse 定义、`-recommend` 模式、终端输出（table/json） | 已完成 |
| `config.py` | 默认常量、环境变量、POSIX/Windows 路径拼接 | 已完成 |
| `devices.py` | 26 个统一设备 + device→aic-npu `--system` 映射 | 已完成 |
| `models.py` | 6 个模型注册表（aic-npu path / msmodeling id / 本地目录） | 已完成 |
| `quantize.py` | `--quantize` 归一化与 msmodeling action 映射 | 已完成 |
| `schema.py` | `RecommendRequest` / `Candidate` / `FrameworkRun` / `UnifiedRunResult` | 已完成 |
| `routing.py` | 路由决策 + 等价命令生成（dry-run 复现） | 已完成 |
| `runners/aic_npu.py` | 调 `cli_default()`，解析 agg/disagg CSV，拆分 prefill/decode 视角 | 已完成 |
| `runners/msmodeling.py` | 复用 `arg_parse()` + `ParallelRunner`，序列化 `OptimizerSummary` | 已完成 |
| `output.py` | `run.json / request.json / summary.txt / *.csv` 落盘 + 终端表格 | 已完成 |
| `service.py` | `recommend(request) -> UnifiedRunResult`（HTTP 复用入口） | 已完成 |
| `docker/Dockerfile.unified` | aic-npu 容器 + msmodeling whl（whl 随仓库提交，按目录 `COPY`） | 已完成 |
| `model_configs/msmodeling/<模型>/config.json` | msmodeling 二级模型目录（相对路径分发，3 个复用真实配置 + 3 个占位） | 已完成 |
| `.github/workflows/build-image-unified.yml` | 手动触发构建并推送统一镜像（含 whl 下载步骤） | 已完成 |

配套文档：[`INFER_STRATEGY_NPU_WIN_DEBUG.md`](INFER_STRATEGY_NPU_WIN_DEBUG.md)（Windows 本地激活环境 / 装 whl / 跑测试 / 看结果）。

### 常用命令

```bash
# 查看支持的模型 / 设备
infer-strategy-npu --list-models
infer-strategy-npu --list-devices

# 走 aic-npu（DeepSeek-R1 / Qwen3.8-27B / DeepSeek-V4-Pro + quantize=DISABLE）
infer-strategy-npu -recommend --model Qwen/Qwen3.8-27B \
  --device ATLAS_800_A2_376T_64G --num-devices 128 \
  --input-length 1024 --output-length 1024 --tpot-limit 50 --ttft-limit 1000 --quantize DISABLE

# 走 msmodeling（Kimi-K3 / GLM-5.3-Flash / MiniMax-M3，或 W8A8 量化）
infer-strategy-npu -recommend --model ZhipuAI/GLM-5.3-Flash \
  --device ATLAS_800_A2_376T_64G --num-devices 128 --quantize W8A8 \
  --local-model-root /workspace/local_models

# 只看下发的命令，不执行
infer-strategy-npu -recommend --model deepseek-ai/DeepSeek-V4-Pro \
  --device ATLAS_800_A2_376T_64G --num-devices 128 --dry-run
```

### HTTP 封装接入点

```python
from infer_strategy_npu.schema import RecommendRequest
from infer_strategy_npu.service import recommend

result = recommend(RecommendRequest(**payload))   # payload 来自 HTTP JSON
return result.to_dict()                            # 直接作为响应体
```

`run.json` 即响应体契约，落盘路径在 `result.artifacts["run_json"]`。
