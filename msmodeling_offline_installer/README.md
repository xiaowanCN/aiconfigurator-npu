# msmodeling 离线安装包

本目录存放 **由 msmodeling 源码构建** 的 wheel，供 `docker/Dockerfile.unified` 与本地调试安装。

> ⚠️ 不要使用官方发布的 `msmodeling-26.1.0-py3-none-any.whl`：
> 它比源码落后较多，**`cli/registry/`（声明式参数注册表）与 `cli/spec_cli.py` 完全缺失**，
> 装了之后 `import cli.registry` 直接失败，`throughput_optimizer` 跑不起来。

## 当前文件

| 文件 | 来源 | 说明 |
|---|---|---|
| `msmodeling-0.2.0-py3-none-any.whl` | 源码 `feature/analyze-throughput-optimizer`（HEAD `44c1350c`）构建 | 版本号取自 `pyproject.toml` 的 `version = "0.2.0"` |

## 重新构建

msmodeling 源码有更新时，重新构建并覆盖本目录的 whl：

```bash
cd D:/Projects/ExpProjects/GITCODEProject/msmodeling

# 构建 wheel，直接输出到本目录（-o 指定输出目录，不会在源码仓库里留 dist/）
uv build --wheel -o "D:/Projects/ExpProjects/GITProject/aic-npu-cn-ascend/aiconfigurator-npu/msmodeling_offline_installer"
```

等价的纯 pip 方式（需先装 `build`）：

```bash
python -m build --wheel --outdir <同上>
```

构建后端是 **hatchling**（见 msmodeling 的 `pyproject.toml`），
打包范围为 `tensor_cast / serving_cast / cli / optix / web_ui`，
并强制包含 `optix/config.toml` 与 `tensor_cast/runtime_configs/draft_configs/dflash_draft_builtin.json`。

## 构建后自检

```powershell
# 确认 cli/registry 已打包进去
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z = [System.IO.Compression.ZipFile]::OpenRead(".\msmodeling-0.2.0-py3-none-any.whl")
($z.Entries | Where-Object { $_.FullName -like 'cli/registry/*' }).Count   # 应 > 0
$z.Dispose()
```

## 注意

- 目录里**只保留一个 `.whl`**：Dockerfile 用 `*.whl` 通配安装，多个会报错。
- 换版本后无需改 Dockerfile / workflow（都是通配），但需同步更新本文件里的版本号说明。
- whl 的 `Requires-Dist` 来自源码 `pyproject.toml`（`torch<=2.10,>=2.8`、
  `transformers>=5.16.0,<5.17.0`、`modelscope`、`salabim` 等），pip 安装时会解析这些依赖。
