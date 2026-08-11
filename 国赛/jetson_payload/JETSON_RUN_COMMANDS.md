# Jetson Upload And Run

## Upload SLAM Files From Windows

Run in Windows PowerShell:

```powershell
cd C:\Users\29765\Desktop\TheDog2\dog_repo
.\jetson_payload\upload_slam_maps_to_jetson.ps1
```

If SSH is not reachable, first make sure the Jetson IP is correct and SSH is running:

```bash
sudo systemctl status ssh
hostname -I
```

## Test Run On Jetson

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
bash scripts/run_guosai_final.sh --dry-run
```

Expected result:

```text
FINAL TASK DONE
dry-run complete, shutting down task_manager_node
```

## Formal Preflight On Jetson

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
python3 scripts/preflight_guosai_final.py \
  --config config/guosai_final.yaml \
  --root /home/jetson/Desktop/guosai/dog_repo/国赛
```

## Formal Run On Jetson

Only run this after preflight passes:

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
bash scripts/run_guosai_final.sh
```

## One-Key Field Flow

Use this when you want one entry point for mapping waypoint capture and the
formal run:

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
bash scripts/guosai_onekey.sh all
```

For waypoint capture only:

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
bash scripts/guosai_onekey.sh collect
```

For the formal run only:

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
bash scripts/guosai_onekey.sh final
```

## Important

`waypoints_FINAL.yaml` in this payload is a template with zero coordinates.
Replace the `x`, `y`, and `yaw` values with the real map waypoints before
formal robot motion.
