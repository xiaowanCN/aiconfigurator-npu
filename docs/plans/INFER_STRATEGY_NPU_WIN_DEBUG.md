# infer-strategy-npu：Windows 本地调试手册

> 适用范围：在 Windows 开发机上激活环境、安装两套框架、跑通 `infer-strategy-npu` 并查看结果。
> 生产/完整验证请在容器里做（见 `docker/Dockerfile.unified`）。

## 0. 本地能验证什么 / 不能验证什么

| 能力 | Windows 本地 | 说明 |
|---|---|---|
| `--dry-run`（只看下发的命令） | ✅ | 不需要任何框架依赖 |
| `--list-models` / `--list-devices` | ✅ | 同上 |
| msmodeling 分支真实跑 | ✅ | 需安装 msmodeling whl + torch（CPU 版即可） |
| aic-npu 分支真实跑 | ❌ | aic-npu 从 **CWD 下的 `./model_configs`** 解析模型配置，本地没有 `/workspace/model_configs`，必然找不到模型。请在容器内验证 |

结论：**Windows 本地主要用来打通 msmodeling 分支 + 路由/参数/落盘逻辑**；aic-npu 分支用 `--dry-run` 校验收参即可。

---

## 1. 准备环境

仓库根目录已有一个 `.venv`（Python 3.12，aic-npu 以 editable 方式装过）。两种方式任选：

### 方式 A：复用仓库自带的 `.venv`（推荐）

```powershell
cd D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\aiconfigurator-npu

# 若 PowerShell 禁止执行脚本，先放开当前进程的策略
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

.\.venv\Scripts\Activate.ps1
python -V          # 应输出 Python 3.12.x
```

> ⚠️ **`.venv` 是 `uv` 建的（`pyvenv.cfg` 里 `uv = 0.7.19`），`uv venv` 默认不装 pip。**
> 若 `python -m pip install ...` 报 `No module named pip`，先补装一次即可：
>
> ```powershell
> python -m ensurepip --upgrade     # 从 ensurepip 的 bundled wheel 装 pip
> python -m pip --version
> ```
>
> 之后正常的 `python -m pip install ...` 就都能用了（也可以直接用 `uv pip install -e .`）。

### 方式 B：用 uv 新建虚拟环境

```powershell
cd D:\Projects\ExpProjects\GITProject\aic-npu-cn-ascend\aiconfigurator-npu
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1
```

---

## 2. 安装 aic-npu（含新的 `infer-strategy-npu` 命令）

```powershell
# 重新 editable 安装，注册 infer-strategy-npu 控制台脚本
python -m pip install -e .

# 验证
python -c "import aiconfigurator_npu, infer_strategy_npu; print(aiconfigurator_npu.__file__); print(infer_strategy_npu.__version__)"
infer-strategy-npu --version
```

> 若 `.venv\Scripts\infer-strategy-npu.exe` 不存在，说明你用的是旧安装，重跑一次 `python -m pip install -e .` 即可。
>
> 不想安装也可以直接跑：`$env:PYTHONPATH="src"; python -m infer_strategy_npu.cli ...`

---

## 3. 安装 msmodeling whl

whl 已随本仓库提交（**由 msmodeling 源码构建**，约 9.4MB），直接装本地文件：

```
<repo>\msmodeling_offline_installer\msmodeling-0.2.0-py3-none-any.whl
```

```powershell
$WHL = ".\msmodeling_offline_installer\msmodeling-0.2.0-py3-none-any.whl"

python -m pip install "$WHL"

# 验证（能 import 即安装成功）
python -c "import cli.inference.throughput_optimizer as t; import tensor_cast.device as d; import serving_cast.parallel_runner as p; print('msmodeling ok', len(d.DeviceProfile.all_device_profiles))"
```

### msmodeling 的运行时依赖

whl 只带 `tensor_cast / serving_cast / optix / cli / web_ui` 自身代码，但 **METADATA 里声明了
`Requires-Dist`**（`torch<=2.10,>=2.8`、`transformers<5.8.0,>=5.6.0`、`modelscope`、`salabim`、
`blinker`、`greenlet`、`optree`、`compressed-tensors`、`scikit-learn`、`pyswarms`、`loguru`、
`strenum`、`aenum`、`overrides`、`colorama`、`openpyxl`、`psutil`、`pillow`、`matplotlib`、
`plotext`、`prettytable`、`pyyaml`、`requests`、`filelock`、`pydantic-settings` 等），
所以 `pip install <whl>` 会**自动解析并拉取**，不需要手工列清单。

唯一例外是 **torch**：Windows 上没有 GPU/CANN 版本，必须显式走 CPU 索引，否则会拉到装不上的版本。

```powershell
# 1) 先装 CPU 版 torch（体积大，约 200MB+，只需装一次）
python -m pip install "torch>=2.8,<=2.10" --index-url https://download.pytorch.org/whl/cpu

# 2) 再装 msmodeling，其余依赖交给 pip 自动解析
python -m pip install .\msmodeling_offline_installer\msmodeling-0.2.0-py3-none-any.whl
```

> 现在 whl 是从源码构建的，METADATA 与源码 `pyproject.toml` 一致
> （`transformers>=5.16.0,<5.17.0`），交给 pip 自动解析即可，不要手工 pin 版本。
> （官方旧的 `msmodeling-26.1.0` whl 声明的是 `<5.8.0,>=5.6.0`，与源码不一致，**不要再用**。）

离线/网络受限时的替代方案（`--no-deps` + 只装吞吐优化器真正需要的）：

```powershell
python -m pip install --no-deps .\msmodeling_offline_installer\msmodeling-0.2.0-py3-none-any.whl
python -m pip install modelscope salabim blinker "greenlet>=3.5.1" "optree>=0.13.0" ^
  "compressed-tensors>=0.16.0" "scikit-learn>=1.7.2" pyswarms loguru strenum aenum ^
  "overrides>=7.7.0" "colorama>=0.4.6" "openpyxl>=3.1.5" psutil "pillow>=12.2.0" ^
  "plotext>=5.3.2" "prettytable>=3.17.0" "pyyaml>=6.0.3" "matplotlib>=3.10.9" ^
  filelock requests pydantic-settings
```

（`transformers` 会随 torch/numpy 一起被上面第 2 步解析；这里省略是因为它通常已由其它包带进来，
若报 `No module named transformers`，单独装一个满足 `>=5.16.0,<5.17.0` 的版本。）

> 只想验证 dry-run / 路由，可以**跳过整节**（dry-run 不会 import msmodeling）。

---

## 4. 准备本地模型（避免联网超时）

msmodeling 默认从 HuggingFace 拉 `config.json`，容易超时。仓库内已建好二级模型目录，
**默认就用相对路径**（相对 CWD，所以要在仓库根目录运行）：

```
model_configs\msmodeling\
├── DeepSeek-R1\config.json        # 复用 aic-npu 的真实配置
├── DeepSeek-V4-Pro\config.json
├── Qwen3.8-27B\config.json
├── GLM5.3-Flash\config.json       # 占位，待填真实内容
├── Kimi-K3\config.json            # 占位，待填真实内容
└── MiniMax-M3\config.json         # 占位，待填真实内容
```

默认情况下分发命令里会是：

```
python -m cli.inference.throughput_optimizer model_configs/msmodeling/GLM5.3-Flash --device ...
```

**填占位配置**：把 ModelScope / HF 上对应模型仓库里的 `config.json` 内容覆盖进去即可
（也可以从 `D:\Projects\ExpProjects\GITProject\msmodeling-self\msmodeling\models\GLM5.3-Flash\config.json` 拷）。
详见 `model_configs\msmodeling\README.md`。

需要指向别处时再指定根目录（二选一）：

```powershell
# 命令行
--local-model-root "D:\Projects\ExpProjects\GITProject\msmodeling-self\msmodeling\models"

# 环境变量
$env:INFER_STRATEGY_LOCAL_MODEL_ROOT = "D:\Projects\ExpProjects\GITProject\msmodeling-self\msmodeling\models"
```

命中规则：`<root>/<local_dir>` 存在就用它，例如 `GLM5.3-Flash`；不存在则回退到 ModelScope ID（会联网）。
注册表映射见 `src/infer_strategy_npu/models.py`。

也可以直接把本地目录路径当 `--model` 传（只要目录存在）。

> 注意：从**仓库根目录**运行 msmodeling 分支，否则相对路径 `model_configs/msmodeling/...` 解析不到。
> 在容器里 CWD 就是 `/workspace`，不存在这个问题。

---

## 5. 跑测试

### 5.1 零依赖冒烟：dry-run + 列表

```powershell
# 看支持的模型 / 设备
infer-strategy-npu --list-models
infer-strategy-npu --list-devices

# 走 aic-npu 的路由（不下发，只打印命令）
infer-strategy-npu -recommend --model Qwen/Qwen3.8-27B `
  --device ATLAS_800_A2_376T_64G --num-devices 128 `
  --input-length 1024 --output-length 1024 --tpot-limit 50 --ttft-limit 1000 `
  --quantize DISABLE --dry-run --out-dir .\_debug_out

# 走 msmodeling 的路由
infer-strategy-npu -recommend --model ZhipuAI/GLM-5.3-Flash `
  --device ATLAS_800_A2_376T_64G --num-devices 128 --quantize W8A8 --dry-run `
  --out-dir .\_debug_out
```

### 5.2 真正跑一次（msmodeling 分支）

用 `TEST_DEVICE` + 小卡数，秒级返回：

```powershell
infer-strategy-npu -recommend --model Qwen/Qwen3.8-27B `
  --device TEST_DEVICE --num-devices 8 `
  --input-length 1024 --output-length 512 `
  --tpot-limit 50 --ttft-limit 2000 `
  --quantize W8A8 `
  --jobs 1 --out-dir .\_debug_out
```

模型配置默认走相对路径 `model_configs/msmodeling/GLM5.3-Flash`（需先把占位 config.json 填好）。

Windows 注意事项：

- `ParallelRunner` 用 `ProcessPoolExecutor`，Windows 是 **spawn**，首次启动较慢；用 `--jobs 1` 降低并发最稳。
- 如果模型加载/编译报错，加 `--no-compile` 再试。
- 想关掉投机解码：`--speculative-method none`。

### 5.3 强制指定框架（排障用）

```powershell
# 强制走 msmodeling（跳过路由）
infer-strategy-npu -recommend --model Qwen/Qwen3.8-27B `
  --device ATLAS_800_A2_376T_64G --num-devices 8 --framework msmodeling --dry-run
```

### 5.4 查参数是否按预期下发

```powershell
infer-strategy-npu -recommend --model deepseek-ai/DeepSeek-V4-Pro `
  --device ATLAS_800_A2_376T_64G --num-devices 128 `
  --input-length 4096 --output-length 2048 --quantize w8a8 `
  --dry-run --out-dir .\_debug_out --format json
```

输出的 `routing.commands` 就是将要喂给 msmodeling 的原始命令，可直接复制去单独执行对比。

---

## 6. 查看结果

每次运行会在 `--out-dir` 下建一个带时间戳的目录：

```
_debug_out\<YYYYmmdd-HHMMSS>-<model>-<device>-<num_devices>\
├── run.json          # 统一结构化结果（主产物，HTTP 将来直接返回这个）
├── request.json      # 归一化后的入参
├── summary.txt       # 终端同款表格
├── aic_npu\          # aic-npu 原始产物（agg/ + disagg/ 的 csv、yaml、png）
├── msmodeling\
│   ├── agg\{summary.csv,candidates.json,command.txt}
│   ├── disagg_prefill\{...}
│   └── disagg_decode\{...}
└── logs\             # 两套框架的 stdout（终端输出已重定向到这里）
```

```powershell
# 列出最近一次运行
Get-ChildItem .\_debug_out | Sort-Object Name | Select-Object -Last 1

# 看统一结果
$run = (Get-ChildItem .\_debug_out | Sort-Object Name | Select-Object -Last 1).FullName
code "$run\run.json"        # 或用记事本 / ConvertFrom-Json
python -c "import json,sys; d=json.load(open(r'$run\run.json',encoding='utf-8')); print(d['status']); print(json.dumps(d['recommendation'],ensure_ascii=False,indent=2)[:2000])"

# 看人类可读摘要（PowerShell 控制台中文可能乱码，建议直接看文件）
Get-Content "$run\summary.txt" -Encoding UTF8
```

`run.json` 的关键结构：

- `request`：归一化入参
- `routing.framework` / `routing.reason` / `routing.commands`：路由决策与等价命令
- `frameworks[0].agg` / `frameworks[0].disagg`：PD 混部 / PD 分离结果
  - `disagg.prefill` / `disagg.decode`：PD分离-Prefill 与 PD分离-Decode 两个视角
  - `*.best`：该模式的最优配置
  - `*.candidates`：Top-N 候选
- `recommendation`：跨 agg/disagg 的最终推荐

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `No module named pip` | venv 由 `uv` 创建且未 seed pip → 先跑 `python -m ensurepip --upgrade` |
| `infer-strategy-npu` 不是可识别的命令 | 未重新 `pip install -e .`，或 venv 未激活；也可 `$env:PYTHONPATH="src"` 后用 `python -m infer_strategy_npu.cli` |
| 终端中文乱码 | PowerShell 控制台编码问题，文件本身是 UTF-8。用 `--format json` 或直接看 `run.json` / `summary.txt`；或先执行 `chcp 65001` |
| msmodeling 分支报 `no module named torch` | 按第 3 节装 CPU 版 torch；仅 dry-run 不需要 |
| msmodeling 分支报 `Unknown --device name` | 设备名必须是 26 个之一，用 `--list-devices` 核对 |
| msmodeling 分支报 `No communication grid found for N devices` | 该设备 profile 的通信网格小于 `--num-devices`，调小卡数 |
| 卡在模型下载 | 默认已走相对路径 `model_configs/msmodeling/<目录>`，确认从**仓库根目录**运行且该目录存在；否则用 `--local-model-root` 指向别的本地模型目录 |
| 报 `--model-path` / 模型 ID 非法字符 | 相对路径 `model_configs/msmodeling/GLM5.3-Flash` 是合法的；若用了含空格或中文的自定义根目录，换到无空格路径 |
| aic-npu 分支报找不到模型配置 | 预期行为：Windows 没有 `/workspace/model_configs`，请在容器内验证 |
| `parallel combination` 相关报错 | `--num-devices` 与 tp/ep 候选不整除，换一个卡数 |
| 想看框架原始日志 | 看 `logs\aic_npu.log` / `logs\msmodeling_<task>.log` |
