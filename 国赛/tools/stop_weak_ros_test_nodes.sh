#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/output/ros_weak_test"
PID_FILE="$LOG_DIR/pids.txt"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found: $PID_FILE"
  echo "Trying to stop known weak-test processes by command name."
  pkill -f "integration_bridge/bridge_node.py" 2>/dev/null || true
  pkill -f "ros2 run arm_grasp inspection_memory_node" 2>/dev/null || true
  pkill -f "ros2 run arm_grasp task_manager_node" 2>/dev/null || true
  echo "Done. If ROS graph still shows stale nodes, wait a few seconds and run: ros2 node list"
  exit 0
fi

echo "Stopping weak ROS test nodes from $PID_FILE"

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

rm -f "$PID_FILE"
echo "Stopped. Logs remain in: $LOG_DIR"
