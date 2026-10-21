# model_configs/msmodeling — msmodeling 本地模型配置目录

msmodeling 分支默认从这里加载模型配置（**相对路径**，相对 CWD；容器内 CWD 为 `/workspace`，
因此实际路径是 `/workspace/model_configs/msmodeling/<模型目录>`）。

目录名与统一 `--model` 的对应关系见 `src/infer_strategy_npu/models.py` 的 `ModelSpec.local_dir`：

| 统一 `--model` | 本地目录 |
|---|---|
| `deepseek-ai/DeepSeek-R1` | `model_configs/msmodeling/DeepSeek-R1` |
| `Qwen/Qwen3.8-27B` | `model_configs/msmodeling/Qwen3.8-27B` |
| `deepseek-ai/DeepSeek-V4-Pro` | `model_configs/msmodeling/DeepSeek-V4-Pro` |
| `ZhipuAI/GLM-5.3-Flash` | `model_configs/msmodeling/GLM5.3-Flash` |
| `ZhipuAI/GLM-5.3` | `model_configs/msmodeling/GLM5.3` |
| `ZhipuAI/GLM-5.2` | `model_configs/msmodeling/GLM5.2` |
| `moonshotai/Kimi-K3` | `model_configs/msmodeling/Kimi-K3` |
| `MiniMax/MiniMax-M3` | `model_configs/msmodeling/MiniMax-M3` |

## 当前状态

- `DeepSeek-R1` / `Qwen3.8-27B` / `DeepSeek-V4-Pro`：已直接复用本仓库 `model_configs/` 下的
  aic-npu 配置（即真实的 HF `config.json`），可直接使用。
- `GLM5.3-Flash` / `GLM5.3` / `GLM5.2` / `Kimi-K3` / `MiniMax-M3`：**占位配置**（含 `"_placeholder": true` 标记），
  需要把真实的 HF / ModelScope `config.json` 内容填进去才能跑出正确结果。

填充后建议删掉 `"_placeholder"` 字段（不影响解析，仅作标记用）。

## 为什么用相对路径

- 便于在容器内外、不同机器上保持同一份命令与 `run.json` 可复现；
- 避免把宿主机绝对路径（`D:\...`、`/home/...`）写进结果文件；
- 容器内 `WORKDIR=/workspace`，`model_configs/` 已由镜像 `COPY` 进去，相对路径可直接命中。

如确需指向其它位置（例如挂载的 `/workspace/local_models`），用
`--local-model-root <路径>` 或环境变量 `INFER_STRATEGY_LOCAL_MODEL_ROOT` 覆盖即可；
目录不存在时会自动回退到 ModelScope ID（会联网，可能超时）。
