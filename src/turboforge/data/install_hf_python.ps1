param([Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$DownloadRoot)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'ensure_python.ps1')
Get-TurboForgePython -ProjectRoot $ProjectRoot -DownloadRoot $DownloadRoot
