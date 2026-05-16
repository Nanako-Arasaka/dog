param(
    [Parameter(Mandatory = $true)][string]$TargetUser,
    [Parameter(Mandatory = $true)][string]$TargetHost,
    [Parameter(Mandatory = $true)][string]$TargetDir
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "Deploying to $TargetUser@$TargetHost:$TargetDir ..."
ssh "$TargetUser@$TargetHost" "mkdir -p $TargetDir"
scp -r "$root\config" "$root\src" "$TargetUser@$TargetHost`:$TargetDir"

Write-Host "Done. Remote run command:"
Write-Host "cd $TargetDir && PYTHONPATH=./src python3 ./src/main.py"
