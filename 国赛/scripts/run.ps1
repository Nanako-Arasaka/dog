param(
    [string]$Python = "python",
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 9800,
    [string]$Mode = "mock",
    [string]$Source = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONPATH = Join-Path $root "src"
& $Python .\vision_server.py --host $HostAddress --port $Port --mode $Mode --source $Source
