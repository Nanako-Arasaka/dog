#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/guosai_final.yaml"
DRY_RUN="false"
START_REALSENSE="true"
START_ORBSLAM3="true"
START_PERCEPTION="true"
START_ARM="true"
SKIP_PREFLIGHT="false"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_guosai_final.sh [options]

Options:
  --dry-run          Print/trace the flow without controlling the dog.
  --config PATH      Use another guosai_final.yaml.
  --no-realsense     Do not launch RealSense.
  --no-orbslam3      Do not launch ORB-SLAM3.
  --no-perception    Do not launch inspection/cone perception.
  --no-arm           Do not launch arm nodes.
  --skip-preflight   Launch directly without dependency checks.
  -h, --help         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      START_REALSENSE="false"
      START_ORBSLAM3="false"
      START_PERCEPTION="false"
      START_ARM="false"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --no-realsense)
      START_REALSENSE="false"
      shift
      ;;
    --no-orbslam3)
      START_ORBSLAM3="false"
      shift
      ;;
    --no-perception)
      START_PERCEPTION="false"
      shift
      ;;
    --no-arm)
      START_ARM="false"
      shift
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/final_$STAMP"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run_guosai_final.log") 2>&1

export ROS_LOG_DIR="$LOG_DIR"
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export PYTHONUNBUFFERED=1
export GUOSAI_ROOT="$ROOT_DIR"

stop_sent="false"
send_stop() {
  if [[ "$stop_sent" == "true" ]]; then
    return
  fi
  stop_sent="true"
  echo "[STOP] Sending final stop..."
  if command -v ros2 >/dev/null 2>&1; then
    timeout 2s ros2 topic pub --once /motion/stop std_msgs/msg/Bool "{data: true}" >/dev/null 2>&1 || true
  fi
  if [[ "$DRY_RUN" != "true" ]]; then
    python3 - "$CONFIG_PATH" <<'PY' || true
import json
import socket
import sys

host = "127.0.0.1"
port = 5005
try:
    import yaml
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    motion = cfg.get("motion", {})
    host = motion.get("receiver_host", host)
    port = int(motion.get("receiver_port", port))
except Exception:
    pass

payload = {"source": "run_guosai_final_trap", "vx": 0.0, "vy": 0.0, "wz": 0.0, "selected": "shutdown_stop"}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), (host, port))
sock.close()
PY
  fi
}
trap send_stop EXIT INT TERM

echo "[INFO] Guosai final one-terminal runner"
echo "[INFO] root=$ROOT_DIR"
echo "[INFO] config=$CONFIG_PATH"
echo "[INFO] log_dir=$LOG_DIR"
echo "[INFO] dry_run=$DRY_RUN"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
  set +u
  source /opt/ros/humble/setup.bash
  set -u
else
  echo "[WARN] /opt/ros/humble/setup.bash not found"
fi

for setup_file in \
  "$ROOT_DIR/install/setup.bash" \
  "$ROOT_DIR/arm_grasp/install/setup.bash" \
  "$ROOT_DIR/controller/colcon_ws/install/setup.bash" \
  "/home/jetson/yahboom_ws/install/setup.bash" \
  "/home/jetson/colcon_ws/install/setup.bash"; do
  if [[ -f "$setup_file" ]]; then
    echo "[INFO] source $setup_file"
    # shellcheck disable=SC1090
    export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES:-}"
    set +u
    source "$setup_file"
    set -u
  fi
done

# Workaround: colcon nested-package path bug leaves ros_robot_controller_msgs
# out of AMENT_PREFIX_PATH / PYTHONPATH despite install/setup.bash being sourced.
# The msgs package was built manually into arm_grasp/install/{share,local/lib}
# so we register both paths explicitly for downstream python launch nodes.
ARM_GRASP_INSTALL="$ROOT_DIR/arm_grasp/install"
if [[ -d "$ARM_GRASP_INSTALL/share/ros_robot_controller_msgs" ]]; then
  export AMENT_PREFIX_PATH="$ARM_GRASP_INSTALL:${AMENT_PREFIX_PATH:-}"
  export PYTHONPATH="$ARM_GRASP_INSTALL/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
  echo "[INFO] arm_grasp msgs paths registered (manual cmake install workaround)"
fi

if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
  echo "[INFO] Running preflight checks..."
  python3 "$ROOT_DIR/scripts/preflight_guosai_final.py" \
    --config "$CONFIG_PATH" \
    --root "$ROOT_DIR" \
    --dry-run "$DRY_RUN" \
    --start-realsense "$START_REALSENSE" \
    --start-orbslam3 "$START_ORBSLAM3" \
    --start-perception "$START_PERCEPTION" \
    --start-arm "$START_ARM"
fi

echo "[INFO] Launching final flow..."
ros2 launch "$ROOT_DIR/launch/guosai_final.launch.py" \
  config_path:="$CONFIG_PATH" \
  dry_run:="$DRY_RUN" \
  log_dir:="$LOG_DIR" \
  start_realsense:="$START_REALSENSE" \
  start_orbslam3:="$START_ORBSLAM3" \
  start_perception:="$START_PERCEPTION" \
  start_arm:="$START_ARM"
