#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/ros_weak_test"
PID_FILE="$LOG_DIR/pids.txt"
ROS_SETUP="/opt/ros/humble/setup.bash"
ARM_SETUP="$ROOT_DIR/arm_grasp/install/setup.bash"
CONFIG_PATH="$ROOT_DIR/arm_grasp/config/grasp_config.yaml"

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
    echo "Run this first: bash $ROOT_DIR/tools/stop_weak_ros_test_nodes.sh"
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
  sleep 0.5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $name exited immediately. Log: $log_file"
    tail -80 "$log_file" || true
    exit 1
  fi
}

start_node integration_bridge_node \
  python3 "$ROOT_DIR/integration_bridge/bridge_node.py" --no-freeze-inspection

start_node inspection_memory_node \
  ros2 run arm_grasp inspection_memory_node

start_node task_manager_node \
  ros2 run arm_grasp task_manager_node --ros-args -p "config_path:=$CONFIG_PATH"

echo
echo "Weak ROS test nodes started. PID file: $PID_FILE"
echo "Logs: $LOG_DIR"
echo
echo "Check nodes:"
echo "  source $ROS_SETUP && source $ARM_SETUP && ros2 node list"
echo
echo "Run smoke test:"
echo "  cd $ROOT_DIR"
echo "  source $ROS_SETUP"
echo "  source $ARM_SETUP"
echo "  python3 tools/ros_flow_smoke_test.py --inspection-repeat 1 --no-reset"
