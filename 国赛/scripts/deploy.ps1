param(
    [Parameter(Mandatory = $true)][string]$TargetUser,
    [Parameter(Mandatory = $true)][string]$TargetHost,
    [Parameter(Mandatory = $true)][string]$TargetDir
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "Deploying inspection vision server to $TargetUser@$TargetHost:$TargetDir ..."
ssh "$TargetUser@$TargetHost" "mkdir -p $TargetDir"
scp -r "$root\config" "$root\src" "$root\assets" "$root\camera_input.py" "$root\vision_server.py" "$TargetUser@$TargetHost`:$TargetDir"

Write-Host "Done. Remote run command:"
Write-Host "cd $TargetDir && PYTHONPATH=./src python3 ./vision_server.py --host 0.0.0.0 --port 9800 --mode camera --source 0"
