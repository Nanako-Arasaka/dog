#!/usr/bin/env bash
# =============================================================================
# 一键走航点 (run_waypoints_only.sh)
#
# 链路: RealSense + ORB-SLAM3 → watchdog → [自动标定 turn_sign/forward_axis]
#       → waypoint_navigator → motion_mux → lite2_motion_receiver → 狗
#
# 2026-08-16(2) 修复:
#   1. 就绪检测改 `ros2 topic echo --once`: 原用 `timeout ros2 topic hz`,
#      但 topic hz 永不自己退出 → timeout 必返 124 → 检查恒假 → 带"图像未
#      就绪"也放行。现在图像 60s 不就绪直接报错退出。
#   2. stand_sit(0x21010202) 是 站立↔趴下 切换指令: 狗已站立时再发 = 趴下!
#      改为询问操作员, 只有趴着才发 stand_sit(默认只发 move_mode,walk_gait)。
#   3. 原地打转不前进 = turn_sign 反了(转向闭环发散, heading 永不收敛, vx 恒 0)。
#      新增 scripts/calibrate_turn_sign.py 自动标定, 不再靠猜。
#   4. waypoint_walker 改边沿触发日志(原来每条 ok=True 10Hz 刷屏"定位OK"),
#      退出时清 goal 停狗; 本脚本启动前清理上次遗留节点防双实例互踩。
#
# 用法:
#   bash scripts/run_waypoints_only.sh                    # 正常(含自动标定)
#   SKIP_TURN_CAL=1 bash scripts/run_waypoints_only.sh    # 跳过标定用默认值
# =============================================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/guosai_final.yaml"
ROBOT_IP="192.168.1.120"
ROBOT_PORT="43893"
LISTEN_PORT="5005"
GOAL_TIMEOUT=60.0
# 默认值(标定失败或 SKIP_TURN_CAL=1 时兜底)。现场证据(2026-08-16):
# TURN_SIGN=+1 时狗原地打转 → 闭环发散 → 默认翻成 -1。
TURN_SIGN=-1.0
FORWARD_AXIS="z"

# 收到 1 条消息即退出(码 0); 无消息 5s 超时(码 124) —— 修复原来 topic hz 恒假
topic_once() { timeout 5 ros2 topic echo --once "$1" >/dev/null 2>&1; }

cleanup_stale() {
  local pidfile="$1" pattern="$2" name="$3"
  [[ -f "$pidfile" ]] || return 0
  local pid args=""
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$args" == *"$pattern"* ]]; then
      kill "$pid" 2>/dev/null || true
      echo "[cleanup] 停掉旧 $name (PID $pid)"
    fi
  fi
  rm -f "$pidfile"
}

# 环境
set +u
source /opt/ros/humble/setup.bash
set -u
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

# 清理上次遗留节点(双 mux/双 navigator 会互踩: 一个发速度一个发停止)
echo ""
echo "=== [0.5] 清理上次遗留节点 ==="
cleanup_stale "/tmp/slam_logs/lite2_receiver.pid" "lite2_motion_receiver.py" "连狗桥"
cleanup_stale "/tmp/slam_logs/watchdog.pid" "localization_watchdog.py" "watchdog"
cleanup_stale "/tmp/slam_logs/navigator.pid" "waypoint_navigator.py" "navigator"
cleanup_stale "/tmp/slam_logs/motion_mux.pid" "motion_mux.py" "motion_mux"
pkill -f "scripts/waypoint_walker.py" 2>/dev/null || true

# 起 RealSense(带 profiles)
echo ""
echo "=== [1] 起 RealSense ==="
if topic_once /camera/camera/color/image_raw; then
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
if topic_once /camera_pose; then
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

# 等 RealSense 图像就绪(不就绪硬退出, 不带病放行)
echo ""
echo "=== [3] 等 RealSense 图像就绪 ==="
IMAGES_READY=0
for i in $(seq 1 15); do
  if topic_once /camera/camera/color/image_raw; then
    IMAGES_READY=1
    echo "[OK] color 已发布"
    break
  fi
  sleep 2
done
if [[ "$IMAGES_READY" != "1" ]]; then
  echo "[FAIL] 30s 内图像未就绪 — RealSense 日志末尾:"
  tail -n 20 /tmp/slam_logs/realsense.log || true
  exit 1
fi

# 起连狗桥(狗姿态确认 + 移动模式 + 行走步态)
echo ""
echo "=== [4] 起连狗桥 (狗姿态确认) ==="
echo "  ⚠️ 0x21010202(stand_sit) 是 站立↔趴下 切换指令: 狗已站立时发它 = 趴下!"
read -r -t 15 -p "  狗当前已站立? 回车=已站立(不发站立指令) / 输入 l=狗趴着(发一次站立): " POSE_ANSWER || true
POSE_ANSWER="${POSE_ANSWER:-}"
if [[ "${POSE_ANSWER,,}" == "l" ]]; then
  STARTUP_ACTIONS="stand_sit,move_mode,walk_gait"
  echo "  → 狗趴着: 启动动作 = 站立 → 移动模式 → 行走步态"
else
  STARTUP_ACTIONS="move_mode,walk_gait"
  echo "  → 狗已站立: 启动动作 = 移动模式 → 行走步态 (不发 stand_sit)"
fi

nohup python3 "$ROOT_DIR/controller/lite2_motion_receiver.py" \
  --listen-ip 0.0.0.0 \
  --listen-port "$LISTEN_PORT" \
  --robot-ip "$ROBOT_IP" \
  --robot-port "$ROBOT_PORT" \
  --heartbeat-hz 4.0 \
  --timeout 0.8 \
  --startup-actions "$STARTUP_ACTIONS" \
  > /tmp/slam_logs/lite2_receiver.log 2>&1 &
echo $! > /tmp/slam_logs/lite2_receiver.pid
disown
sleep 2
echo "[OK] 连狗桥 PID=$(cat /tmp/slam_logs/lite2_receiver.pid) → $ROBOT_IP:$ROBOT_PORT"

# 起 watchdog + motion_mux (navigator 挪到标定之后启动)
echo ""
echo "=== [5] 起导航底层节点 (watchdog + mux) ==="
nohup python3 "$ROOT_DIR/nodes/localization_watchdog.py" \
  --ros-args -p pose_topic:=/camera_pose -p fused_pose_topic:=/camera_pose_fused \
  -p ok_topic:=/localization/ok -p stop_topic:=/motion/stop \
  -p ground_plane:=xz -p forward_axis:=$FORWARD_AXIS \
  > /tmp/slam_logs/watchdog.log 2>&1 &
echo $! > /tmp/slam_logs/watchdog.pid
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
LOC_OK=0
for i in $(seq 1 20); do
  if timeout 3 ros2 topic echo --once /localization/ok 2>/dev/null | grep -q "data: true"; then
    LOC_OK=1
    echo "[OK] 定位建立!"
    break
  fi
  sleep 2
done
if [[ "$LOC_OK" != "1" ]]; then
  echo "[WARN] 40s 内定位未 OK — watchdog 日志末尾(继续启动, walker 会自行等待):"
  tail -n 10 /tmp/slam_logs/watchdog.log || true
fi

# 自动标定 turn_sign / forward_axis (原地打转根因的根治手段)
echo ""
echo "=== [6.5] 自动标定 turn_sign / forward_axis ==="
echo "  ⚠️ 狗将原地慢转 ~30° 并向前走 ~0.4m — 确保周围 0.5m 无障碍!"
if [[ "${SKIP_TURN_CAL:-0}" == "1" ]]; then
  echo "  [SKIP] SKIP_TURN_CAL=1, 用默认 TURN_SIGN=$TURN_SIGN FORWARD_AXIS=$FORWARD_AXIS"
else
  CAL_OUT="$(timeout 120 python3 "$ROOT_DIR/scripts/calibrate_turn_sign.py" \
    --forward-axis "$FORWARD_AXIS" 2>&1)" || true
  echo "$CAL_OUT"
  CAL_LINE="$(echo "$CAL_OUT" | grep '^CALIB ' | tail -n1 || true)"
  CAL_TS="$(echo "$CAL_LINE" | grep -o 'turn_sign=[+-]1\.0' | cut -d= -f2 || true)"
  CAL_FA="$(echo "$CAL_LINE" | grep -o 'forward_axis=[zx]' | cut -d= -f2 || true)"
  if [[ -n "$CAL_TS" ]]; then TURN_SIGN="$CAL_TS"; fi
  if [[ -n "$CAL_FA" ]]; then FORWARD_AXIS="$CAL_FA"; fi
  echo "  → 使用 TURN_SIGN=$TURN_SIGN FORWARD_AXIS=$FORWARD_AXIS"
fi

# 起 navigator (带标定出的符号/轴)
echo ""
echo "=== [7] 起 waypoint_navigator ==="
nohup python3 "$ROOT_DIR/nodes/waypoint_navigator.py" \
  --ros-args \
  -p waypoints_yaml:=/home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml \
  -p pose_topic:=/camera_pose_fused -p goal_topic:=/waypoint/goal \
  -p status_topic:=/waypoint/status -p cmd_topic:=/motion/nav_cmd \
  -p localization_ok_topic:=/localization/ok \
  -p ground_plane:=xz -p forward_axis:=$FORWARD_AXIS -p turn_sign:=$TURN_SIGN \
  > /tmp/slam_logs/navigator.log 2>&1 &
echo $! > /tmp/slam_logs/navigator.pid
disown
sleep 2
echo "[OK] navigator PID=$(cat /tmp/slam_logs/navigator.pid) (turn_sign=$TURN_SIGN, forward_axis=$FORWARD_AXIS)"

echo ""
echo "=== [8] 开始走航点(从 obstacle_exit 开始, 避开 start_exit 距原点太近)==="
python3 "$ROOT_DIR/scripts/waypoint_walker.py" \
  --waypoints /home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml \
  --goal-timeout "$GOAL_TIMEOUT" \
  --start-from obstacle_exit

echo ""
echo "=== 完成 ==="
