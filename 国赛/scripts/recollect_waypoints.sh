#!/usr/bin/env bash
# =============================================================================
# 全自动重采航点 (recollect_waypoints.sh)
#
# 流程(用户只需按 Enter):
#   1. 起 RealSense + ORB-SLAM3(带 profiles + bgr8)
#   2. 等图像就绪
#   3. 提示推狗触发重定位(pose 从 0 跳几米)
#   4. 跑 waypoint_capture_tool.py 交互模式 —— 每个航点提示后按 Enter
#      (工具自动采稳定 pose → 存 z → 保存, 失败自动重试)
#
# 用法:
#   bash scripts/recollect_waypoints.sh
#
# ⚠️ 前置:
#   - 狗有电 + 在地图区域(建图场地)
#   - 网络通: ping 192.168.1.120
#   - 每步: 推狗到目标位置 → 等稳定 → 按 Enter
# =============================================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WP_OUTPUT="/home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml"
ROBOT_IP="192.168.1.120"

echo "================================================"
echo " 全自动重采 13 航点"
echo " 输出: $WP_OUTPUT"
echo "================================================"

# ── 环境 ──
set +u
source /opt/ros/humble/setup.bash
source ~/setenv_arm.sh 2>/dev/null || true
for setup_file in \
  "$ROOT_DIR/install/setup.bash" \
  "$ROOT_DIR/arm_grasp/install/setup.bash" \
  "$ROOT_DIR/controller/colcon_ws/install/setup.bash" \
  "/home/jetson/yahboom_ws/install/setup.bash" \
  "/home/jetson/colcon_ws/install/setup.bash"; do
  if [[ -f "$setup_file" ]]; then
    source "$setup_file"
  fi
done
set -u
export GUOSAI_ROOT="$ROOT_DIR"
export PATH="/home/jetson/colcon_ws/build/orbslam3:$PATH"

mkdir -p /tmp/slam_logs

# ── 预检 ──
echo ""
echo "=== [0] 预检 ==="
if ! ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
  echo "[WARN] $ROBOT_IP 不可达 — 采集不需要狗网络, 继续 (推狗定位需 SLAM 即可)"
fi
lsusb 2>/dev/null | grep -q "8086:0b3a" && echo "[OK] RealSense 在线" || { echo "[FAIL] RealSense 不在"; exit 1; }

# ── 起栈 ──
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

# ── 等图像就绪 ──
echo ""
echo "=== [3] 等 SLAM 图像就绪 ==="
for i in $(seq 1 20); do
  if timeout 3 ros2 topic hz /camera/camera/color/image_raw >/dev/null 2>&1; then
    echo "[OK] color 已发布"
    break
  fi
  sleep 2
done

# ── 等重定位 ──
echo ""
echo "=== [4] 等待 SLAM 重定位 ==="
echo "  ⚠️ 狗必须在建图场地内!"
echo "  若 pose 未跳变, 推狗 0.5m + 转 15° 触发重定位"
for i in $(seq 1 30); do
  pose=$(timeout 3 ros2 topic echo --once /camera_pose 2>/dev/null | grep -A 1 "position:" | grep -oE "-?[0-9]+\.[0-9]+" | head -3 | tr '\n' ' ')
  if [[ -n "$pose" ]] && ! echo "$pose" | grep -qE "0\.000|0\.001|0\.002"; then
    echo "[OK] pose 已建立: ($pose) m"
    break
  fi
  sleep 2
done

# ── 采集 ──
echo ""
echo "=== [5] 开始采集 13 航点 ==="
echo "  每次: 推狗到目标位置 → 等稳定(画面不动 3 秒) → 按 Enter"
echo "  输入 s 跳过当前点, q 退出保存"
echo ""
python3 "$ROOT_DIR/scripts/waypoint_capture_tool.py" \
  --output "$WP_OUTPUT" \
  --pose-topic /camera_pose --pose-type pose_stamped \
  --timeout-sec 60 \
  --waypoints \
  start_exit obstacle_entry obstacle_exit \
  inspection_box_1_side_1 inspection_box_1_side_2 \
  inspection_box_2_side_1 inspection_box_2_side_2 \
  pick_area place_A place_B place_C place_D finish

echo ""
echo "=== 完成 ==="
