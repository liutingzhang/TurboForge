function Get-TurboForgePython {
    param([Parameter(Mandatory=$true)][string]$ProjectRoot, [Parameter(Mandatory=$true)][string]$DownloadRoot)
    $spec = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'python-runtime.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $pythonParent = Join-Path $env:LOCALAPPDATA 'NInferTurboForge'
    New-Item -ItemType Directory -Path $pythonParent -Force | Out-Null
    $pythonParent = [IO.Path]::GetFullPath($pythonParent)
    $target = Join-Path $pythonParent $spec.folder
    $python = Join-Path $target 'python.exe'
    if (Test-Path -LiteralPath $python) {
        & $python -c "import sys,struct,sysconfig;sys.exit(0 if (3,13,18) <= sys.version_info < (3,14) and struct.calcsize('P') == 8 and not sysconfig.get_config_var('Py_GIL_DISABLED') else 1)" 2>$null
        if ($LASTEXITCODE -ne 0) { throw '专用 Python 版本或文件异常，请重新准备环境。' }
        return $python
    }
    if (Test-Path -LiteralPath $target) { throw ('Python 目录已存在但不完整，请检查后移除该目录再启动：' + $target) }
    $archive = Join-Path (Join-Path $ProjectRoot 'runtime/python') $spec.filename
    if (-not (Test-Path -LiteralPath $archive)) {
        New-Item -ItemType Directory -Path $DownloadRoot -Force | Out-Null
        $archive = Join-Path $DownloadRoot $spec.filename
        if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $spec.sha256) {
            Write-Host '正在下载带安全补丁的 Python 3.13.18（约 22 MB，不修改系统 PATH）...'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $spec.url -OutFile $archive -UseBasicParsing
        }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $spec.sha256) {
        throw 'Python 归档 SHA-256 不匹配，拒绝解压或运行。请重新下载完整发布包。'
    }
    $tar = Join-Path $env:SystemRoot 'System32/tar.exe'
    if (-not (Test-Path -LiteralPath $tar)) { throw '需要 Windows 10/11 自带的 tar.exe，请更新 Windows 后重试。' }
    $members = & $tar -tzf $archive
    if ($LASTEXITCODE -ne 0) { throw '无法读取 Python 归档。' }
    foreach ($member in $members) {
        $name = $member.Replace('\','/')
        if (-not $name.StartsWith('python/') -or $name -match '(^|/)\.\.(/|$)|:') { throw 'Python 归档路径无效，拒绝解压。' }
    }
    $staging = Join-Path $pythonParent ('python-staging-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        Write-Host '正在准备 Python 3.13.18 专用运行时...'
        & $tar -xzf $archive -C $staging
        if ($LASTEXITCODE -ne 0) { throw 'Python 解压失败。' }
        $candidate = Join-Path $staging 'python/python.exe'
        & $candidate -c "import sys,venv,ensurepip,ssl,struct;assert sys.version_info[:3] == (3,13,18);assert struct.calcsize('P') == 8"
        if ($LASTEXITCODE -ne 0) { throw 'Python 自检失败。' }
        $stagingFull = [IO.Path]::GetFullPath($staging)
        $targetFull = [IO.Path]::GetFullPath($target)
        if (-not $stagingFull.StartsWith($pythonParent + '\') -or -not $targetFull.StartsWith($pythonParent + '\')) { throw 'Python 安装路径越界。' }
        Move-Item -LiteralPath (Join-Path $staging 'python') -Destination $target
    } finally {
        $stagingFull = [IO.Path]::GetFullPath($staging)
        if ($stagingFull.StartsWith($pythonParent + '\') -and (Split-Path -Leaf $stagingFull).StartsWith('python-staging-') -and (Test-Path -LiteralPath $stagingFull)) {
            Remove-Item -LiteralPath $stagingFull -Recurse -Force
        }
    }
    return $python
}
