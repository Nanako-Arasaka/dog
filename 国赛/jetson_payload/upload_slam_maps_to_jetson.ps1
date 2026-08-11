$ErrorActionPreference = "Stop"

$JetsonHost = "192.168.31.174"
$JetsonUser = "jetson"
$LocalRoot = Split-Path -Parent $PSScriptRoot
$LocalSlamMaps = Join-Path $LocalRoot "jetson_payload\slam_maps"
$RemoteSlamMaps = "/home/jetson/Desktop/guosai/slam_maps"

Write-Host "[INFO] Local slam maps: $LocalSlamMaps"
Write-Host "[INFO] Jetson target: ${JetsonUser}@${JetsonHost}:$RemoteSlamMaps"

ssh "${JetsonUser}@${JetsonHost}" "mkdir -p $RemoteSlamMaps"
scp (Join-Path $LocalSlamMaps "guosai_rgbd_map_FINAL.osa") "${JetsonUser}@${JetsonHost}:$RemoteSlamMaps/"
scp (Join-Path $LocalSlamMaps "guosai_realsense_rgbd_FINAL.yaml") "${JetsonUser}@${JetsonHost}:$RemoteSlamMaps/"
scp (Join-Path $LocalSlamMaps "waypoints_FINAL.yaml") "${JetsonUser}@${JetsonHost}:$RemoteSlamMaps/"

ssh "${JetsonUser}@${JetsonHost}" "ls -lh $RemoteSlamMaps && cd /home/jetson/Desktop/guosai/dog_repo/国赛 && python3 scripts/preflight_guosai_final.py --config config/guosai_final.yaml --root /home/jetson/Desktop/guosai/dog_repo/国赛"

