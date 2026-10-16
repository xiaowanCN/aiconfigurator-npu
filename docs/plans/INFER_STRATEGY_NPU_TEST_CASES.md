# infer-strategy-npu 测试验证手册（Linux 端）

本文档给出 `infer-strategy-npu` 在 Linux 端的全部验证命令，覆盖当前已适配的 6 个模型、
两条下发路径（aic-npu / msmodeling）与两种量化（DISABLE / W8A8）。配套脚本见
`scripts/run_infer_strategy_tests.sh`（支持后台挂起执行）。

---

## 1. 前置条件

1. **安装统一 CLI**（仓库内，从仓库根目录）：
   ```bash
   pip install -e .
   # 验证
   infer-strategy-npu --version
   # 若未生成控制台入口，可用回退: python -m infer_strategy_npu.cli
   ```
2. **msmodeling whl 已安装**：统一镜像里已装 `msmodeling-0.2.0`；裸仓库需先装离线 whl
   （`msmodeling_offline_installer/*.whl`）。
3. **模型配置可解析**：aic-npu 的 config 固定在 `/workspace/model_configs/<file>`；
   msmodeling 的 config 默认取相对路径 `model_configs/msmodeling/<dir>/config.json`。
   **推荐直接在统一镜像里运行**（默认 CWD=`/workspace`，配置齐全）；若用裸仓库 checkout，
   请从仓库根目录运行，使相对路径 `model_configs/msmodeling` 能被解析。
4. **环境变量**（按需）：
   - `TORCH_DEVICE_BACKEND_AUTOLOAD=0`：关闭 torch 后端自动加载，避免无 NPU 驱动的机器上
     启动即去探测 NPU；统一镜像/真实 NPU 主机上都不影响 msmodeling 的显式 `import torch_npu`。
   - `INFER_STRATEGY_LOCAL_MODEL_ROOT`：覆盖 msmodeling 本地模型根目录（默认 `model_configs/msmodeling`）。

---

## 2. 路由规则（决定命令下发给谁）

`infer-strategy-npu` 根据 **模型 + 量化** 自动选框架，也可用 `--framework` 强制：

| 条件 | 下发框架 |
|---|---|
| 模型 ∈ {DeepSeek-R1, Qwen3.8-27B, DeepSeek-V4-Pro} **且** `--quantize DISABLE` | **aic-npu** |
| 其余所有情况（含上述三模型用 W8A8，以及其余三模型任意量化） | **msmodeling** |
| `--framework aic-npu` 强制 | aic-npu（要求设备能映射到 `--system`，见下） |
| `--framework msmodeling` 强制 | msmodeling |

设备约束：aic-npu 路径只支持 `ATLAS_800_A2_376T_64G`（映射到 `--system ascend_910b`），
所以强制 `--framework aic-npu` 也必须用该设备；msmodeling 支持 `DEVICES` 中全部 26 种设备。

---

## 3. 已适配模型与配置现状

| 模型（--model） | aic-npu | msmodeling | msmodeling 本地 config 现状 |
|---|---|---|---|
| `deepseek-ai/DeepSeek-R1` | ✅ | ✅ | 真实配置（`DeepSeek-R1/`） |
| `Qwen/Qwen3.8-27B` | ✅ | ✅ | 真实配置（`Qwen3.8-27B/`） |
| `deepseek-ai/DeepSeek-V4-Pro` | ✅ | ✅ | 真实配置（`DeepSeek-V4-Pro/`） |
| `ZhipuAI/GLM-5.3-Flash` | ❌ | ✅ | 占位骨架（`GLM5.3-Flash/`） |
| `moonshotai/Kimi-K3` | ❌ | ✅ | 占位骨架（`Kimi-K3/`） |
| `MiniMax/MiniMax-M3` | ❌ | ✅ | 占位骨架（`MiniMax-M3/`） |

> 占位骨架 config 仅供 dry-run / 命令构造验证；真实 msmodeling 运行可能失败或产出无意义结果，
> 待补真实 config 后即可。

---

## 4. 测试用例矩阵

默认设备 `ATLAS_800_A2_376T_64G`、卡数 `64`。`--dry-run` 只构造并下发命令、不真正执行，任何环境可跑。

| ID | 模型 | 量化 | 框架(强制/自动) | 期望路由 | 分组 |
|---|---|---|---|---|---|
| T00 | — | — | — | 自检：version / list-models / list-devices | 自检 |
| T01 | DeepSeek-R1 | DISABLE | auto | aic-npu | A dry-run |
| T02 | DeepSeek-R1 | W8A8 | auto | msmodeling | A dry-run |
| T03 | Qwen3.8-27B | DISABLE | auto | aic-npu | A dry-run |
| T04 | Qwen3.8-27B | W8A8 | auto | msmodeling | A dry-run |
| T05 | DeepSeek-V4-Pro | DISABLE | auto | aic-npu | A dry-run |
| T06 | DeepSeek-V4-Pro | W8A8 | auto | msmodeling | A dry-run |
| T07 | GLM-5.3-Flash | DISABLE | auto | msmodeling | A dry-run |
| T08 | Kimi-K3 | W8A8 | auto | msmodeling | A dry-run |
| T09 | MiniMax-M3 | W8A8 | auto | msmodeling | A dry-run |
| T10 | GLM-5.3-Flash | — | msmodeling | msmodeling | A dry-run |
| T11 | DeepSeek-R1 | — | aic-npu | aic-npu | A dry-run |
| R01 | DeepSeek-R1 | DISABLE | aic-npu | aic-npu | B 真实 |
| R02 | Qwen3.8-27B | DISABLE | aic-npu | aic-npu | B 真实 |
| R03 | DeepSeek-V4-Pro | DISABLE | aic-npu | aic-npu | B 真实 |
| R04 | DeepSeek-R1 | W8A8 | auto | msmodeling | C 真实(需NPU) |
| R05 | Qwen3.8-27B | W8A8 | auto | msmodeling | C 真实(需NPU) |
| R06 | DeepSeek-V4-Pro | W8A8 | auto | msmodeling | C 真实(需NPU) |
| R07 | GLM-5.3-Flash | DISABLE | auto | msmodeling | C 真实(占位,可能失败) |
| R08 | Kimi-K3 | W8A8 | auto | msmodeling | C 真实(占位,可能失败) |
| R09 | MiniMax-M3 | W8A8 | auto | msmodeling | C 真实(占位,可能失败) |

---

## 5. 手动运行命令（可直接复制）

> 说明：`--quantize` 默认即 `DISABLE`，下面为清晰显式写出。
> 命令里的 `--out-dir` 可省略（默认 `/workspace/results/infer-strategy-npu`），这里写成独立子目录便于逐条查看。

### 0. 自检
```bash
infer-strategy-npu --version
infer-strategy-npu --list-models
infer-strategy-npu --list-devices
```

### A. dry-run（全量，安全，推荐先跑）
```bash
B=infer-strategy-npu; D=ATLAS_800_A2_376T_64G; N=64; O=./itn-test-results
$B -recommend --model deepseek-ai/DeepSeek-R1    --device $D --num-devices $N --quantize DISABLE --out-dir $O/T01 --dry-run
$B -recommend --model deepseek-ai/DeepSeek-R1    --device $D --num-devices $N --quantize W8A8   --out-dir $O/T02 --dry-run
$B -recommend --model Qwen/Qwen3.8-27B           --device $D --num-devices $N --quantize DISABLE --out-dir $O/T03 --dry-run
$B -recommend --model Qwen/Qwen3.8-27B           --device $D --num-devices $N --quantize W8A8   --out-dir $O/T04 --dry-run
$B -recommend --model deepseek-ai/DeepSeek-V4-Pro --device $D --num-devices $N --quantize DISABLE --out-dir $O/T05 --dry-run
$B -recommend --model deepseek-ai/DeepSeek-V4-Pro --device $D --num-devices $N --quantize W8A8   --out-dir $O/T06 --dry-run
$B -recommend --model ZhipuAI/GLM-5.3-Flash      --device $D --num-devices $N --quantize DISABLE --out-dir $O/T07 --dry-run
$B -recommend --model moonshotai/Kimi-K3         --device $D --num-devices $N --quantize W8A8   --out-dir $O/T08 --dry-run
$B -recommend --model MiniMax/MiniMax-M3         --device $D --num-devices $N --quantize W8A8   --out-dir $O/T09 --dry-run
$B -recommend --model ZhipuAI/GLM-5.3-Flash      --device $D --num-devices $N --framework msmodeling --out-dir $O/T10 --dry-run
$B -recommend --model deepseek-ai/DeepSeek-R1    --device $D --num-devices $N --framework aic-npu   --out-dir $O/T11 --dry-run
```

### B. 真实 aic-npu（纯 Python 估算，CPU 即可跑）
```bash
$B -recommend --model deepseek-ai/DeepSeek-R1     --device $D --num-devices $N --framework aic-npu --out-dir $O/R01
$B -recommend --model Qwen/Qwen3.8-27B            --device $D --num-devices $N --framework aic-npu --out-dir $O/R02
$B -recommend --model deepseek-ai/DeepSeek-V4-Pro --device $D --num-devices $N --framework aic-npu --out-dir $O/R03
```

### C. 真实 msmodeling（需 NPU 主机；transformers 会显式 import torch_npu）
```bash
$B -recommend --model deepseek-ai/DeepSeek-R1     --device $D --num-devices $N --quantize W8A8 --out-dir $O/R04
$B -recommend --model Qwen/Qwen3.8-27B            --device $D --num-devices $N --quantize W8A8 --out-dir $O/R05
$B -recommend --model deepseek-ai/DeepSeek-V4-Pro --device $D --num-devices $N --quantize W8A8 --out-dir $O/R06
$B -recommend --model ZhipuAI/GLM-5.3-Flash      --device $D --num-devices $N --out-dir $O/R07   # 占位config, 可能失败
$B -recommend --model moonshotai/Kimi-K3          --device $D --num-devices $N --quantize W8A8 --out-dir $O/R08   # 占位, 可能失败
$B -recommend --model MiniMax/MiniMax-M3          --device $D --num-devices $N --quantize W8A8 --out-dir $O/R09   # 占位, 可能失败
```

---

## 6. 如何解读输出

每条命令在 `--out-dir` 对应子目录下生成：
- `run.json`：完整结果（`status`、`request`、`routing.framework`、`artifacts`、`recommendation`）。
  dry-run 时 `routing.framework` 即为实际会下发的框架，可直接核对路由是否正确。
- `summary.txt`：`render_summary` 渲染的终端摘要（agg / disagg 最佳配置与吞吐）。
- msmodeling 真实运行还会产出 `msmodeling/<mode>/summary.csv` 与 `candidates.json`
  （`mode` ∈ agg / disagg_prefill / disagg_decode）。
- aic-npu 真实运行产出 `aic_npu/<mode>/best_config_topn.csv`。

**判定标准**：
- dry-run：`status` 应为 `ok` 或 `partial`，且 `routing.framework` 符合第 4 节“期望路由”。
- 真实运行：`status=ok` 且 `recommendation` 非空即视为通过。

---

## 7. 已知事项

1. **aic-npu 设备限制**：强制 `--framework aic-npu` 或非 W8A8 的三模型自动路由，都必须用
   `ATLAS_800_A2_376T_64G`；换其他设备会报 `device not supported by aic-npu` 的 `RoutingError`。
2. **真实 msmodeling 需要 NPU 主机**：其导入链经 transformers → `is_torch_npu_available()` →
   `import torch_npu`，需要宿主机挂载的 `libascend_hal.so`。纯 CPU 机器上真实 msmodeling 运行会因
   缺驱动而 import 失败（dry-run 不受影响）。
3. **占位 config**：`GLM5.3-Flash / Kimi-K3 / MiniMax-M3` 目前是骨架 config，真实 msmodeling 运行
   可能失败或结果无意义，属预期；dry-run 与路由验证不受影响。
4. **transformers 版本冲突**：msmodeling 要求 `transformers>=5.16,<5.17`，会与镜像内 vllm 的
   `transformers==5.5.4` 产生 pip 告警（非致命）；本 CLI 不经 vllm，不影响两条路径。

---

## 8. 一键脚本与挂起执行

脚本 `scripts/run_infer_strategy_tests.sh` 自动跑上述矩阵，逐条记录日志并打印汇总。

```bash
# 默认: 跑 A(dry-run 全量) + B(真实 aic-npu)
bash scripts/run_infer_strategy_tests.sh

# 额外跑 C(真实 msmodeling, 需 NPU 主机)
bash scripts/run_infer_strategy_tests.sh --with-msmodeling-real

# 挂起(后台)执行, 日志落地, 随时 tail 跟进
nohup bash scripts/run_infer_strategy_tests.sh --with-msmodeling-real > itn-test-results/run_all.log 2>&1 &
echo "PID: $!"
tail -f itn-test-results/run_all.log        # Ctrl-C 退出 tail, 不影响后台进程

# 查看汇总
cat itn-test-results/summary.txt
```

可用环境变量覆盖默认：`DEVICE`、`NUM_DEVICES`、`OUT_DIR`、`BIN`。
每个用例的独立日志在 `itn-test-results/logs/<id>.log`。
