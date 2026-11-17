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
    throw 'One-click deployment requires Windows x64; 32-bit systems and ARM64 are not supported.'
}

# A pinned standalone interpreter avoids old host Python and registry changes.
. (Join-Path $projectRoot 'src/turboforge/data/ensure_python.ps1')
$bootstrapPython = Get-TurboForgePython -ProjectRoot $projectRoot -DownloadRoot (Join-Path $managedRoot 'downloads')

$venvRoot = Join-Path $managedRoot 'venv-py31318'
$managedPython = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $managedPython)) {
    Write-Host 'Preparing the dedicated TurboForge Python environment...'
    & $bootstrapPython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the Python environment.' }
}
& $managedPython -c "import sys;sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>$null
if ($LASTEXITCODE -ne 0) { throw 'The dedicated Python environment is corrupted or invalid after being moved. Delete .turboforge\venv-py31318 in the project and start again; models and other caches can be kept.' }
$env:PYTHONIOENCODING = 'utf-8'
$env:TURBOFORGE_HOME = $managedRoot
& $managedPython (Join-Path $PSScriptRoot 'start_web.py') @args
$startupExit = $LASTEXITCODE
} catch {
    Write-Host ('Startup failed: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Full log: ' + $startupLog)
    $startupExit = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $startupExit
