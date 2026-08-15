#!/usr/bin/env bash
# =============================================================================
# 一键走航点 (run_waypoints_only.sh)
#
# 链路: RealSense + ORB-SLAM3 → watchdog → waypoint_navigator → motion_mux
#       → lite2_motion_receiver(先站立) → 狗 → 顺序走完 13 航点
#
# 不含巡检/抓取 —— 只验证"从站起到走完全程"。
#
# 用法:
#   bash scripts/run_waypoints_only.sh
# =============================================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/guosai_final.yaml"
ROBOT_IP="192.168.1.120"
ROBOT_PORT="43893"
LISTEN_PORT="5005"
GOAL_TIMEOUT=60.0

# 环境
source /opt/ros/humble/setup.bash
for setup_file in \
  "$ROOT_DIR/install/setup.bash" \
  "$ROOT_DIR/arm_grasp/install/setup.bash" \
  "$ROOT_DIR/controller/colcon_ws/install/setup.bash" \
  "/home/jetson/yahboom_ws/install/setup.bash" \
  "/home/jetson/colcon_ws/install/setup.bash"; do
  if [[ -f "$setup_file" ]]; then
    set +u; source "$setup_file"; set -u
  fi
done
export GUOSAI_ROOT="$ROOT_DIR"
export PATH="/home/jetson/colcon_ws/build/orbslam3:$PATH"

mkdir -p /tmp/slam_logs

echo "================================================"
echo " 一键走航点 (start → obstacle → 4×inspection → pick → place×4 → finish)"
echo " robot=$ROBOT_IP:$ROBOT_PORT"
echo "================================================"

# 预检: 狗端网络
echo ""
echo "=== [0] 狗端网络 ==="
if ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
  echo "[OK] 狗运动主机 $ROBOT_IP 可达"
else
  echo "[FAIL] $ROBOT_IP 不可达 — 确认连狗热点(192.168.1.x)后重试"
  exit 1
fi

# 起 RealSense(带 profiles)
echo ""
echo "=== [1] 起 RealSense ==="
if timeout 3 ros2 topic hz /camera/camera/color/image_raw >/dev/null 2>&1; then
  echo "[OK] 已在发布, 跳过"
else
  nohup ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=true \
    enable_infra1:=false enable_infra2:=false \
    align_depth.enable:=true pointcloud.enable:=false \
    rgb_camera.color_profile:=640x480x15 depth_module.depth_profile:=640x480x15 \
    > /tmp/slam_logs/realsense.log 2>&1 &
  echo $! > /tmp/slam_logs/realsense.pid
  disown
  echo "[OK] 启动中 (PID $(cat /tmp/slam_logs/realsense.pid))"
fi

# 起 ORB-SLAM3
echo ""
echo "=== [2] 起 ORB-SLAM3 ==="
if timeout 3 ros2 topic hz /camera_pose >/dev/null 2>&1; then
  echo "[OK] pose 已在发布, 跳过"
else
  cd /home/jetson/Desktop/guosai/slam_maps
  nohup ros2 run orbslam3 rgbd \
    "$GUOSAI_ROOT/controller/ORB_SLAM3/Vocabulary/ORBvoc.txt" \
    /home/jetson/Desktop/guosai/slam_maps/guosai_realsense_rgbd_localization_v4.yaml \
    --ros-args -p use_viewer:=false -p color_encoding:=bgr8 -p sync_queue_size:=20 \
    -r /camera/color/image_raw:=/camera/camera/color/image_raw \
    -r /camera/aligned_depth_to_color/image_raw:=/camera/camera/aligned_depth_to_color/image_raw \
    > /tmp/slam_logs/orbslam3.log 2>&1 &
  echo $! > /tmp/slam_logs/orbslam3.pid
  disown
  cd "$ROOT_DIR"
  echo "[OK] 启动中 (PID $(cat /tmp/slam_logs/orbslam3.pid))"
fi

# 等 SLAM 图像就绪
echo ""
echo "=== [3] 等 SLAM 图像就绪 ==="
for i in $(seq 1 15); do
  if timeout 3 ros2 topic hz /camera/camera/color/image_raw >/dev/null 2>&1; then
    echo "[OK] color 已发布"
    break
  fi
  sleep 2
done

# 起连狗桥(先站立 → 移动模式 → 行走步态)
echo ""
echo "=== [4] 起连狗桥 + 狗站立 ==="
nohup python3 "$ROOT_DIR/controller/lite2_motion_receiver.py" \
  --listen-ip 0.0.0.0 \
  --listen-port "$LISTEN_PORT" \
  --robot-ip "$ROBOT_IP" \
  --robot-port "$ROBOT_PORT" \
  --heartbeat-hz 4.0 \
  --timeout 0.8 \
  --startup-actions stand_sit,move_mode,walk_gait \
  > /tmp/slam_logs/lite2_receiver.log 2>&1 &
echo $! > /tmp/slam_logs/lite2_receiver.pid
disown
sleep 2
echo "[OK] 连狗桥 PID=$(cat /tmp/slam_logs/lite2_receiver.pid) → $ROBOT_IP:$ROBOT_PORT"
echo "      启动动作: 站立 → 移动模式 → 行走步态"

# 起 motion_mux + waypoint_navigator + localization_watchdog
echo ""
echo "=== [5] 起导航节点 (watchdog + navigator + mux) ==="
nohup python3 "$ROOT_DIR/nodes/localization_watchdog.py" \
  --ros-args -p pose_topic:=/camera_pose -p fused_pose_topic:=/camera_pose_fused \
  -p ok_topic:=/localization/ok -p stop_topic:=/motion/stop \
  > /tmp/slam_logs/watchdog.log 2>&1 &
echo $! > /tmp/slam_logs/watchdog.pid
disown

nohup python3 "$ROOT_DIR/nodes/waypoint_navigator.py" \
  --ros-args \
  -p waypoints_yaml:=/home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml \
  -p pose_topic:=/camera_pose_fused -p goal_topic:=/waypoint/goal \
  -p status_topic:=/waypoint/status -p cmd_topic:=/motion/nav_cmd \
  -p localization_ok_topic:=/localization/ok \
  > /tmp/slam_logs/navigator.log 2>&1 &
echo $! > /tmp/slam_logs/navigator.pid
disown

nohup python3 "$ROOT_DIR/nodes/motion_mux.py" \
  --ros-args \
  > /tmp/slam_logs/motion_mux.log 2>&1 &
echo $! > /tmp/slam_logs/motion_mux.pid
disown
sleep 3

echo ""
echo "=== [6] 等待 SLAM 重定位 ==="
echo "  ⚠️ 如果 30s 内 pose 未跳变: 推狗 0.5m + 转 15° 触发重定位"
for i in $(seq 1 20); do
  if timeout 3 ros2 topic echo --once /localization/ok 2>/dev/null | grep -q "data: true"; then
    echo "[OK] 定位建立! 开始走航点"
    break
  fi
  sleep 2
done

echo ""
echo "=== [7] 开始走航点 ==="
python3 "$ROOT_DIR/scripts/waypoint_walker.py" \
  --waypoints /home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml \
  --goal-timeout "$GOAL_TIMEOUT"

echo ""
echo "=== 完成 ==="
