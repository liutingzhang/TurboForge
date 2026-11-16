$ErrorActionPreference = 'Stop'
# Use modules matching this PowerShell process, even when launched from pwsh 7.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$env:PSModulePath = Join-Path $PSHOME 'Modules'
$projectRoot = Split-Path -Parent $PSScriptRoot
$managedRoot = Join-Path $projectRoot '.turboforge'
New-Item -ItemType Directory -Path $managedRoot -Force | Out-Null
$logRoot = Join-Path $managedRoot 'logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$startupLog = Join-Path $logRoot ('startup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -Path $startupLog -Force | Out-Null
try {
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    throw '一键部署需要 Windows x64；不支持 32 位系统或 ARM64。'
}

# A pinned standalone interpreter avoids old host Python and registry changes.
. (Join-Path $projectRoot 'src/turboforge/data/ensure_python.ps1')
$bootstrapPython = Get-TurboForgePython -ProjectRoot $projectRoot -DownloadRoot (Join-Path $managedRoot 'downloads')

$venvRoot = Join-Path $managedRoot 'venv-py31318'
$managedPython = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $managedPython)) {
    Write-Host '正在准备 TurboForge 专用 Python 环境...'
    & $bootstrapPython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Python 环境创建失败。' }
}
& $managedPython -c "import sys;sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>$null
if ($LASTEXITCODE -ne 0) { throw '专用 Python 环境已损坏或移动后失效。请删除项目 .turboforge\venv-py31318 后重新启动；模型与其他缓存可保留。' }
$env:PYTHONIOENCODING = 'utf-8'
$env:TURBOFORGE_HOME = $managedRoot
& $managedPython (Join-Path $PSScriptRoot 'start_web.py') @args
$startupExit = $LASTEXITCODE
} catch {
    Write-Host ('启动失败：' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('完整日志：' + $startupLog)
    $startupExit = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $startupExit
