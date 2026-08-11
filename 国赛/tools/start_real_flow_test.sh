#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-}"
ALLOW_ARM="${2:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/real_flow_test"
PID_FILE="$LOG_DIR/pids.txt"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ARM_SETUP="${ARM_SETUP:-$ROOT_DIR/arm_grasp/install/setup.bash}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/arm_grasp/config/grasp_config.yaml}"
SERIAL_DEVICE="${SERIAL_DEVICE:-/dev/ttyUSB0}"

usage() {
  cat <<USAGE
Usage:
  bash tools/start_real_flow_test.sh inspection
  bash tools/start_real_flow_test.sh perception
  bash tools/start_real_flow_test.sh arm --allow-arm-control

Modes:
  inspection  real inspection camera script -> bridge -> memory -> task_manager
  perception  inspection mode + red-bar vision_node, no arm_control_node
  arm         perception mode + serial_bridge_node + arm_control_node

Environment:
  FREEZE_INSPECTION=0  forward inspection immediately instead of freezing A/B/C/D
  START_ASTRA=1        also start arm_grasp/astra_camera_node.py for /rgbd_cam topics
  SERIAL_DEVICE=...    serial device for arm mode, default /dev/ttyUSB0
USAGE
}

if [[ "$MODE" != "inspection" && "$MODE" != "perception" && "$MODE" != "arm" ]]; then
  usage
  exit 2
fi

if [[ "$MODE" == "arm" && "$ALLOW_ARM" != "--allow-arm-control" && "${ALLOW_ARM_CONTROL:-0}" != "1" ]]; then
  echo "ERROR: arm mode starts real arm control."
  echo "Rerun with: bash tools/start_real_flow_test.sh arm --allow-arm-control"
  exit 2
fi

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  alive=0
  while read -r pid name; do
    [[ -z "${pid:-}" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name is already running with pid $pid"
      alive=1
    fi
  done < "$PID_FILE"
  if [[ "$alive" -eq 1 ]]; then
    echo "Run this first: bash $ROOT_DIR/tools/stop_real_flow_test.sh"
    exit 1
  fi
fi

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ERROR: ROS setup not found: $ROS_SETUP"
  exit 1
fi

if [[ ! -f "$ARM_SETUP" ]]; then
  echo "ERROR: arm_grasp is not built: $ARM_SETUP"
  echo "Build it first:"
  echo "  cd $ROOT_DIR/arm_grasp"
  echo "  source $ROS_SETUP"
  echo "  colcon build"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi

set +u
source "$ROS_SETUP"
source "$ARM_SETUP"
set -u

: > "$PID_FILE"

start_node() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/$name.log"
  echo "Starting $name -> $log_file"
  (cd "$ROOT_DIR" && "$@" > "$log_file" 2>&1) &
  local pid=$!
  echo "$pid $name" >> "$PID_FILE"
  sleep 0.8
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $name exited immediately. Log: $log_file"
    tail -120 "$log_file" || true
    exit 1
  fi
}

BRIDGE_ARGS=()
if [[ "${FREEZE_INSPECTION:-1}" == "0" ]]; then
  BRIDGE_ARGS+=(--no-freeze-inspection)
fi

start_node integration_bridge_node \
  python3 "$ROOT_DIR/integration_bridge/bridge_node.py" "${BRIDGE_ARGS[@]}"

start_node inspection_memory_node \
  ros2 run arm_grasp inspection_memory_node

start_node task_manager_node \
  ros2 run arm_grasp task_manager_node --ros-args -p "config_path:=$CONFIG_PATH"

start_node live_inspection \
  python3 "$ROOT_DIR/live_detect_yolo_opencv.py"

if [[ "$MODE" == "perception" || "$MODE" == "arm" ]]; then
  if [[ "${START_ASTRA:-0}" == "1" ]]; then
    start_node astra_camera_node \
      python3 "$ROOT_DIR/arm_grasp/astra_camera_node.py"
  fi

  start_node vision_node \
    ros2 run arm_grasp vision_node --ros-args -p "config_path:=$CONFIG_PATH" -p target_color:=red
fi

if [[ "$MODE" == "arm" ]]; then
  start_node serial_bridge_node \
    ros2 run ros_robot_controller serial_bridge_node --ros-args -p "device:=$SERIAL_DEVICE"

  start_node arm_control_node \
    ros2 run arm_grasp arm_control_node --ros-args -p "config_path:=$CONFIG_PATH"
fi

echo
echo "Real flow test nodes started."
echo "Mode: $MODE"
echo "PID file: $PID_FILE"
echo "Logs: $LOG_DIR"
echo
echo "Monitor in this terminal:"
echo "  cd $ROOT_DIR"
echo "  source $ROS_SETUP"
echo "  source $ARM_SETUP"
echo "  python3 tools/monitor_real_flow.py"
echo
echo "Stop:"
echo "  bash $ROOT_DIR/tools/stop_real_flow_test.sh"
