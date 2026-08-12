#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/guosai_final.yaml"
MODE="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

YES="false"
FRESH_MAP="true"
START_REALSENSE="true"
START_ORBSLAM3="true"
OUTPUT_WAYPOINTS=""
COLLECT_PIDS=()

cleanup_collect() {
  if [[ "${#COLLECT_PIDS[@]}" -eq 0 ]]; then
    return
  fi
  echo "[INFO] stopping collection background processes..."
  for pid in "${COLLECT_PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/guosai_onekey.sh collect [options]
  bash scripts/guosai_onekey.sh final [options]
  bash scripts/guosai_onekey.sh dry-run [options]
  bash scripts/guosai_onekey.sh preflight [options]
  bash scripts/guosai_onekey.sh all [options]

Modes:
  collect    One-terminal RealSense + ORB-SLAM3 + interactive waypoint capture.
  final      Run the formal national-competition flow.
  dry-run    Run the formal flow without hardware motion.
  preflight  Check dependencies and required files only.
  all        Capture waypoints, run preflight, then ask before formal run.

Options:
  --config PATH          Use another guosai_final.yaml.
  --output PATH          Waypoint YAML output path for collect/all.
  --load-existing-map    Use orbslam3.command from config instead of fresh initialization.
  --no-realsense         Do not start RealSense during collect/all.
  --no-orbslam3          Do not start ORB-SLAM3 during collect/all.
  --yes                  In all mode, start final run after collection without typing RUN.
  -h, --help             Show this help.
EOF
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_WAYPOINTS="$2"
      shift 2
      ;;
    --load-existing-map)
      FRESH_MAP="false"
      shift
      ;;
    --no-realsense)
      START_REALSENSE="false"
      shift
      ;;
    --no-orbslam3)
      START_ORBSLAM3="false"
      shift
      ;;
    --yes)
      YES="true"
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

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 2
fi

export GUOSAI_ROOT="$ROOT_DIR"

source_ros() {
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u
  fi
  for setup_file in \
    "$ROOT_DIR/install/setup.bash" \
    "$ROOT_DIR/arm_grasp/install/setup.bash" \
    "$ROOT_DIR/controller/colcon_ws/install/setup.bash" \
    "/home/jetson/yahboom_ws/install/setup.bash" \
    "/home/jetson/colcon_ws/install/setup.bash"; do
    if [[ -f "$setup_file" ]]; then
      echo "[INFO] source $setup_file"
      set +u
      # shellcheck disable=SC1090
      source "$setup_file"
      set -u
    fi
  done

  # Workaround: colcon nested-package path bug leaves ros_robot_controller_msgs
  # out of AMENT_PREFIX_PATH / PYTHONPATH despite install/setup.bash being sourced.
  # The msgs package was built manually into arm_grasp/install/{share,local/lib}.
  ARM_GRASP_INSTALL="$ROOT_DIR/arm_grasp/install"
  if [[ -d "$ARM_GRASP_INSTALL/share/ros_robot_controller_msgs" ]]; then
    export AMENT_PREFIX_PATH="$ARM_GRASP_INSTALL:${AMENT_PREFIX_PATH:-}"
    export PYTHONPATH="$ARM_GRASP_INSTALL/local/lib/python3.10/dist-packages:${PYTHONPATH:-}"
    echo "[INFO] arm_grasp msgs paths registered (manual cmake install workaround)"
  fi
}

cfg_value() {
  local dotted_key="$1"
  local fallback="${2:-}"
  python3 - "$CONFIG_PATH" "$ROOT_DIR" "$dotted_key" "$fallback" <<'PY'
import os
import sys
from pathlib import Path

import yaml

config_path, root, dotted_key, fallback = sys.argv[1:5]
os.environ["GUOSAI_ROOT"] = root
try:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    value = data
    for part in dotted_key.split("."):
        value = value[part]
    if isinstance(value, str):
        value = os.path.expandvars(value.replace("${GUOSAI_ROOT}", root))
    print(value)
except Exception:
    print(fallback)
PY
}

wait_for_topic() {
  local topic="$1"
  local deadline=$((SECONDS + 45))
  echo "[INFO] waiting for topic: $topic"
  while (( SECONDS < deadline )); do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      echo "[OK] topic ready: $topic"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] topic not found: $topic" >&2
  return 1
}

wait_for_pose_once() {
  local topic="$1"
  echo "[INFO] waiting for pose messages on $topic"
  if timeout 30s ros2 topic echo --once "$topic" >/dev/null 2>&1; then
    echo "[OK] pose is publishing"
    return 0
  fi
  echo "[WARN] $topic did not publish within 30s. Move/rotate the camera slowly until ORB-SLAM3 initializes."
  return 0
}

collect_waypoints() {
  source_ros
  local stamp log_dir color_topic depth_topic pose_topic pose_type output settings_yaml voc_path orb_command realsense_command
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_dir="$ROOT_DIR/logs/collect_$stamp"
  mkdir -p "$log_dir"
  export ROS_LOG_DIR="$log_dir"
  export RCUTILS_LOGGING_BUFFERED_STREAM=1
  export PYTHONUNBUFFERED=1

  color_topic="$(cfg_value realsense.color_topic /camera/camera/color/image_raw)"
  depth_topic="$(cfg_value realsense.depth_topic /camera/camera/aligned_depth_to_color/image_raw)"
  pose_topic="$(cfg_value slam.pose_topic /camera_pose)"
  pose_type="$(cfg_value slam.pose_type pose_stamped)"
  output="${OUTPUT_WAYPOINTS:-$(cfg_value slam.waypoints_yaml /home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml)}"
  settings_yaml="$(cfg_value slam.settings_yaml /home/jetson/Desktop/guosai/slam_maps/guosai_realsense_rgbd_FINAL.yaml)"
  voc_path="$(cfg_value slam.vocabulary_path "$ROOT_DIR/controller/ORB_SLAM3/Vocabulary/ORBvoc.txt")"
  if [[ ! -f "$voc_path" && -f /home/jetson/ORB_SLAM3/Vocabulary/ORBvoc.txt ]]; then
    voc_path="/home/jetson/ORB_SLAM3/Vocabulary/ORBvoc.txt"
  fi

  echo "[INFO] collect log_dir=$log_dir"
  echo "[INFO] waypoint output=$output"

  trap cleanup_collect EXIT INT TERM

  if [[ "$START_REALSENSE" == "true" ]]; then
    realsense_command="$(cfg_value realsense.command "ros2 launch realsense2_camera rs_launch.py camera_name:=camera enable_color:=true enable_depth:=true align_depth.enable:=true pointcloud.enable:=false")"
    echo "[RUN] RealSense"
    bash -lc "$realsense_command" >"$log_dir/realsense.log" 2>&1 &
    COLLECT_PIDS+=("$!")
    wait_for_topic "$color_topic"
    wait_for_topic "$depth_topic"
  fi

  if [[ "$START_ORBSLAM3" == "true" ]]; then
    if [[ "$FRESH_MAP" == "true" ]]; then
      orb_command="cd $(printf "%q" "$(dirname "$settings_yaml")") && ros2 run orbslam3 rgbd $(printf "%q" "$voc_path") $(printf "%q" "$settings_yaml") --ros-args -p use_viewer:=false -r /camera/color/image_raw:=$color_topic -r /camera/aligned_depth_to_color/image_raw:=$depth_topic"
    else
      orb_command="$(cfg_value orbslam3.command "")"
    fi
    echo "[RUN] ORB-SLAM3"
    bash -lc "$orb_command" >"$log_dir/orbslam3.log" 2>&1 &
    COLLECT_PIDS+=("$!")
    wait_for_pose_once "$pose_topic"
  fi

  python3 "$ROOT_DIR/scripts/waypoint_capture_tool.py" \
    --output "$output" \
    --pose-topic "$pose_topic" \
    --pose-type "$pose_type" \
    --stable-samples 10 \
    --stable-max-position-step 0.04 \
    --stable-max-yaw-step 0.18 \
    --timeout-sec 20
}

case "$MODE" in
  collect)
    collect_waypoints
    ;;
  preflight)
    source_ros
    python3 "$ROOT_DIR/scripts/preflight_guosai_final.py" --config "$CONFIG_PATH" --root "$ROOT_DIR"
    ;;
  dry-run)
    bash "$ROOT_DIR/scripts/run_guosai_final.sh" --config "$CONFIG_PATH" --dry-run
    ;;
  final)
    bash "$ROOT_DIR/scripts/run_guosai_final.sh" --config "$CONFIG_PATH"
    ;;
  all)
    collect_waypoints
    source_ros
    python3 "$ROOT_DIR/scripts/preflight_guosai_final.py" --config "$CONFIG_PATH" --root "$ROOT_DIR"
    if [[ "$YES" != "true" ]]; then
      echo ""
      read -r -p "Type RUN to start the formal national-competition flow: " answer
      if [[ "$answer" != "RUN" ]]; then
        echo "[INFO] final run skipped"
        exit 0
      fi
    fi
    bash "$ROOT_DIR/scripts/run_guosai_final.sh" \
      --config "$CONFIG_PATH" \
      --no-realsense \
      --no-orbslam3 \
      --skip-preflight
    ;;
  *)
    echo "[ERROR] Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac
