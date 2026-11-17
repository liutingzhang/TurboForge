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
        if ($LASTEXITCODE -ne 0) { throw 'The dedicated Python version or files are abnormal; prepare the environment again.' }
        return $python
    }
    if (Test-Path -LiteralPath $target) { throw ('The Python directory already exists but is incomplete; inspect and remove it before starting: ' + $target) }
    $archive = Join-Path (Join-Path $ProjectRoot 'runtime/python') $spec.filename
    if (-not (Test-Path -LiteralPath $archive)) {
        New-Item -ItemType Directory -Path $DownloadRoot -Force | Out-Null
        $archive = Join-Path $DownloadRoot $spec.filename
        if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $spec.sha256) {
            Write-Host 'Downloading security-patched Python 3.13.18 (about 22 MB; the system PATH is not modified)...'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $spec.url -OutFile $archive -UseBasicParsing
        }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $spec.sha256) {
        throw 'The Python archive SHA-256 does not match; extraction and execution are refused. Re-download the complete release package.'
    }
    $tar = Join-Path $env:SystemRoot 'System32/tar.exe'
    if (-not (Test-Path -LiteralPath $tar)) { throw 'The tar.exe bundled with Windows 10/11 is required; update Windows and retry.' }
    $members = & $tar -tzf $archive
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the Python archive.' }
    foreach ($member in $members) {
        $name = $member.Replace('\','/')
        if (-not $name.StartsWith('python/') -or $name -match '(^|/)\.\.(/|$)|:') { throw 'Invalid Python archive path; extraction refused.' }
    }
    $staging = Join-Path $pythonParent ('python-staging-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging | Out-Null
    try {
        Write-Host 'Preparing the dedicated Python 3.13.18 runtime...'
        & $tar -xzf $archive -C $staging
        if ($LASTEXITCODE -ne 0) { throw 'Python extraction failed.' }
        $candidate = Join-Path $staging 'python/python.exe'
        & $candidate -c "import sys,venv,ensurepip,ssl,struct;assert sys.version_info[:3] == (3,13,18);assert struct.calcsize('P') == 8"
        if ($LASTEXITCODE -ne 0) { throw 'Python self-check failed.' }
        $stagingFull = [IO.Path]::GetFullPath($staging)
        $targetFull = [IO.Path]::GetFullPath($target)
        if (-not $stagingFull.StartsWith($pythonParent + '\') -or -not $targetFull.StartsWith($pythonParent + '\')) { throw 'The Python install path is out of bounds.' }
        Move-Item -LiteralPath (Join-Path $staging 'python') -Destination $target
    } finally {
        $stagingFull = [IO.Path]::GetFullPath($staging)
        if ($stagingFull.StartsWith($pythonParent + '\') -and (Split-Path -Leaf $stagingFull).StartsWith('python-staging-') -and (Test-Path -LiteralPath $stagingFull)) {
            Remove-Item -LiteralPath $stagingFull -Recurse -Force
        }
    }
    return $python
}
