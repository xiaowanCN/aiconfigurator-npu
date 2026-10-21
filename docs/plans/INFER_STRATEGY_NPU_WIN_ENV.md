# infer-strategy-npu：Windows 本地测试环境说明

> 配套脚本：`scripts/setup_win_env.ps1`（一键恢复，幂等）。
> 更完整的调试步骤见 `INFER_STRATEGY_NPU_WIN_DEBUG.md`。

## 1. 这个环境是用来干什么的

Windows 本地开发机只用来跑 **msmodeling 分支** 的真实推理吞吐评估 + 路由/参数/落盘逻辑校验。
aic-npu 分支在本地没有 `/workspace/model_configs`，只能靠 `--dry-run` 校验收参（见 WIN_DEBUG 第 0 节）。

要让 msmodeling 分支真实跑起来，`.venv` 里除了本项目（editable 安装）之外，还必须额外装两类依赖：
**CPU 版 torch** 和 **离线 msmodeling whl**。这两类依赖**不在本项目的 `pyproject.toml` 里**，
是当初手动装进 `.venv` 的。

## 2. 依赖清单与为什么需要

| 依赖 | 来源 | 为什么必须手动装 |
|---|---|---|
| `torch`（CPU 版） | PyTorch CPU 索引 | Windows 无 CANN/GPU，必须显式走 `--index-url https://download.pytorch.org/whl/cpu`，否则 pip 会试图拉取装不上的版本 |
| `msmodeling`（离线 whl） | `msmodeling_offline_installer/msmodeling-0.2.0-py3-none-any.whl` | 包含 `tensor_cast` / `serving_cast` 等本仓不带的模块；whl 的 METADATA 声明了 `Requires-Dist`，`pip install` 会**自动拉取** transformers/modelscope/optree 等，无需手工列清单 |

> 不要手工 pin `transformers` 版本：whl 的 METADATA 已与源码 `pyproject.toml` 一致
> （`transformers>=5.16.0,<5.17.0`），交给 pip 自动解析即可。旧版 `msmodeling-26.1.0` whl 声明不一致，**不要再用**。

## 3. 一键恢复（推荐）

```powershell
cd D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\aiconfigurator-npu

# 若 PowerShell 禁止执行脚本
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 一键装好 torch(CPU) + msmodeling whl，并做 import 自检；已装则跳过
scripts\setup_win_env.ps1
```

脚本逻辑（幂等）：
1. 找 `.venv\Scripts\python.exe`；若缺失提示先 `uv venv .venv --python 3.12`。
2. 若 `pip` 不存在则 `ensurepip --upgrade`（uv 建的 venv 默认无 pip）。
3. `import torch` 失败 → 装 CPU 版 torch；否则跳过。
4. `import tensor_cast` 失败 → 装离线 msmodeling whl；否则跳过。
5. 跑自检：`import tensor_cast.device` 等，打印注册的 device profile 数量。

## 4. ⚠️ 关键陷阱：`uv sync` 会清空这套环境

`.venv` 是 `uv` 建的。执行 `uv sync` / `uv venv`（重建）/ `uv pip sync` 会按本项目
`pyproject.toml` **重建**环境，把手动装进去的 torch / transformers / msmodeling 全部清掉，
只保留项目本身声明的轻量依赖。表现就是：cli 能启动（`infer-strategy-npu --version` 正常），
但一跑 msmodeling 分支就报 `No module named 'tensor_cast'`。

**遇到这个报错，直接重跑 `scripts\setup_win_env.ps1` 即可恢复**（第 3 节）。

> 日常开发建议用 `uv pip install -e .` 而不是 `uv sync`，可减少误清依赖的概率；
> 但即便如此，只要 `.venv` 被重建，仍需重跑上面的恢复脚本。

## 5. 冒烟测试

恢复后，任选一条验证：

```powershell
# 只看下发命令（不需要任何框架依赖，dry-run 一定可用）
.\.venv\Scripts\python.exe -m infer_strategy_npu.cli -recommend `
    --model ZhipuAI/GLM-5.3 --device ATLAS_950_486T_96G --num-devices 16 `
    --quantize ENABLE --pd-balance-sizes 1P1D --dry-run

# 真实跑 msmodeling（需要上面的 torch + whl 就绪）
.\.venv\Scripts\python.exe -m infer_strategy_npu.cli -recommend `
    --model ZhipuAI/GLM-5.3 --device ATLAS_950_486T_96G --num-devices 16 `
    --input-length 8196 --output-length 1024 --tpot-limit 200 --ttft-limit 30000 `
    --quantize ENABLE --pd-balance-search
```

预期：路由到 `msmodeling`，TP 候选从 4 起跳（GLM5.3/GLM5.2 中等模型），
`--quantize ENABLE` 默认下发 `--quantize-linear-action W8A8_DYNAMIC`
`--quantize-non-expert-linear-action DISABLED` `--quantize-attention-action DISABLED`；
传入 `--quantize DISABLE` 时三项强制 `DISABLED`。
