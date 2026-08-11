#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/real_flow_test"
PID_FILE="$LOG_DIR/pids.txt"

stop_known_processes() {
  pkill -f "integration_bridge/bridge_node.py" 2>/dev/null || true
  pkill -f "live_detect_yolo_opencv.py" 2>/dev/null || true
  pkill -f "arm_grasp/astra_camera_node.py" 2>/dev/null || true
  pkill -f "ros2 run arm_grasp inspection_memory_node" 2>/dev/null || true
  pkill -f "ros2 run arm_grasp task_manager_node" 2>/dev/null || true
  pkill -f "ros2 run arm_grasp vision_node" 2>/dev/null || true
  pkill -f "ros2 run arm_grasp arm_control_node" 2>/dev/null || true
  pkill -f "ros2 run ros_robot_controller serial_bridge_node" 2>/dev/null || true
}

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found: $PID_FILE"
  echo "Trying to stop known real-flow test processes by command name."
  stop_known_processes
  echo "Done. If ROS graph still shows stale nodes, wait a few seconds and run: ros2 node list"
  exit 0
fi

echo "Stopping real flow test nodes from $PID_FILE"

while read -r pid name; do
  [[ -z "${pid:-}" ]] && continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name pid=$pid"
    kill "$pid" 2>/dev/null || true
  else
    echo "$name pid=$pid is not running"
  fi
done < "$PID_FILE"

sleep 1

while read -r pid name; do
  [[ -z "${pid:-}" ]] && continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "Force stopping $name pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
done < "$PID_FILE"

stop_known_processes
rm -f "$PID_FILE"
echo "Stopped. Logs remain in: $LOG_DIR"
