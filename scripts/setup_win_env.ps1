<#
.SYNOPSIS
    Windows 本地测试环境一键恢复脚本（infer-strategy-npu + msmodeling）。
.DESCRIPTION
    在仓库自带的 .venv 中安装 msmodeling 真实跑所需的依赖：
      - CPU 版 torch（Windows 无 CANN/GPU，必须走 CPU 索引，否则装不上）
      - 离线 msmodeling whl（自动拉取 transformers / modelscope / optree 等）
    幂等：已安装则跳过对应步骤，可反复执行。
    典型用法：执行了 `uv sync` / 重建 `.venv` 导致环境被清空后，重跑本脚本即可恢复。
.NOTES
    若 PowerShell 禁止执行脚本，先放开当前进程策略：
      Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
#>

$ErrorActionPreference = 'Stop'
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$Py = Join-Path $RepoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Py)) {
    Write-Error "未找到 $Py。请先创建虚拟环境：uv venv .venv --python 3.12"
    exit 1
}

function Test-PyImport($mod) {
    & $Py -c "import $mod" 2>$null
    return $LASTEXITCODE -eq 0
}

# 0) 确保 pip 可用（uv 建的 venv 默认不带 pip）
if (-not (Test-PyImport 'ensurepip')) {
    Write-Host "==> bootstrap pip (ensurepip)"
    & $Py -m ensurepip --upgrade
} elseif (-not (Test-PyImport 'pip')) {
    Write-Host "==> bootstrap pip (ensurepip)"
    & $Py -m ensurepip --upgrade
}

# 1) CPU 版 torch（仅需一次，体积约 200MB+）
if (-not (Test-PyImport 'torch')) {
    Write-Host "==> 安装 CPU 版 torch"
    & $Py -m pip install "torch>=2.8,<=2.10" --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "==> torch 已安装，跳过"
}

# 2) 离线 msmodeling whl
$Whl = Join-Path $RepoRoot 'msmodeling_offline_installer\msmodeling-0.2.0-py3-none-any.whl'
if (-not (Test-Path $Whl)) {
    Write-Error "未找到离线 whl：$Whl"
    exit 1
}
if (-not (Test-PyImport 'tensor_cast')) {
    Write-Host "==> 安装 msmodeling whl + 依赖"
    & $Py -m pip install $Whl
} else {
    Write-Host "==> msmodeling 已安装，跳过"
}

# 3) 自检
Write-Host "==> 自检"
& $Py -c "import cli.inference.throughput_optimizer as t; import tensor_cast.device as d; import serving_cast.parallel_runner as p; print('msmodeling ok; device_profiles =', len(d.DeviceProfile.all_device_profiles))"
if ($LASTEXITCODE -ne 0) {
    Write-Error "自检失败，环境未就绪"
    exit 1
}

Write-Host ""
Write-Host "环境就绪。冒烟测试："
Write-Host "  .\.venv\Scripts\python.exe -m infer_strategy_npu.cli -recommend --model ZhipuAI/GLM-5.3 --device ATLAS_950_486T_96G --num-devices 16 --quantize ENABLE --dry-run"
