#!/usr/bin/env bash
# =============================================================================
# 一键测试固定运动 (run_fixed_motion.sh)
#
# 链路: RealSense + ORB-SLAM3 → watchdog → motion_mux → UDP 5005
#       → lite2_motion_receiver → 狗; 最后跑 scripts/fixed_motion.py
#
# 用法:
#   bash scripts/run_fixed_motion.sh                              # 默认序列(4m→右转90°→2.5m→右转90°→0.5m→右转90°)
#   bash scripts/run_fixed_motion.sh --sequence "2,90r,1.5,90r"   # 自定义序列(其余参数原样透传)
#   DRY_RUN=1 bash scripts/run_fixed_motion.sh                    # 安全演练: receiver dry-run, 不发真指令给狗
#   TURN_CAL=1 bash scripts/run_fixed_motion.sh                   # 先自动标定 turn_sign/forward_axis 再跑
#   SKIP_SLAM=1 bash scripts/run_fixed_motion.sh                  # SLAM 栈已在跑, 跳过起 RealSense/ORB
#
# 默认: 跳过标定(用现场已确认 TURN_SIGN=-1.0 FORWARD_AXIS=z), 只测运动闭环。
# 退出: 只发急停, 底层(receiver/watchdog/mux)保持运行, 便于反复测试。
# =============================================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="192.168.1.120"
ROBOT_PORT="43893"
LISTEN_PORT="5005"

# 现场已确认默认值 (与 run_waypoints_only.sh 一致; 2026-08-16 实测 +wz 使 heading 减小)
TURN_SIGN="${TURN_SIGN:--1.0}"
FORWARD_AXIS="${FORWARD_AXIS:-z}"
TURN_CAL="${TURN_CAL:-0}"
SKIP_SLAM="${SKIP_SLAM:-0}"
DRY_RUN="${DRY_RUN:-0}"

# 收到 1 条消息即退出(码 0); 无消息 5s 超时(码 124)
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

# 急停钩子: 脚本退出时向 receiver 发零速 (receiver 本身保持运行便于复测)
emergency_stop() {
  python3 - "$LISTEN_PORT" <<'PY' || true
import json, socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(json.dumps({"vx": 0.0, "vy": 0.0, "wz": 0.0}).encode(), ("127.0.0.1", port))
s.close()
PY
}
trap emergency_stop EXIT INT TERM

# ---------------------------------------------------------------- 环境
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
# orbslam3 install 缺可执行文件 (build 在 build/) → PATH hack
export PATH="/home/jetson/colcon_ws/build/orbslam3:$PATH"
mkdir -p /tmp/slam_logs

echo "================================================"
echo " 一键测试固定运动 (fixed_motion)"
echo " robot=$ROBOT_IP:$ROBOT_PORT  dry_run=$DRY_RUN  turn_sign=$TURN_SIGN"
echo " 参数: $*"
echo "================================================"

# ---------------------------------------------------------------- [0] 网络
echo ""
echo "=== [0] 狗端网络 ==="
if ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
  echo "[OK] 狗运动主机 $ROBOT_IP 可达"
else
  echo "[FAIL] $ROBOT_IP 不可达 — 确认连狗热点(192.168.1.x)后重试"
  exit 1
fi

# ---------------------------------------------------------------- [0.5] 清理遗留
echo ""
echo "=== [0.5] 清理上次遗留节点 ==="
cleanup_stale "/tmp/slam_logs/lite2_receiver.pid" "lite2_motion_receiver.py" "连狗桥"
cleanup_stale "/tmp/slam_logs/watchdog.pid" "localization_watchdog.py" "watchdog"
cleanup_stale "/tmp/slam_logs/navigator.pid" "waypoint_navigator.py" "navigator"
cleanup_stale "/tmp/slam_logs/motion_mux.pid" "motion_mux.py" "motion_mux"
pkill -f "scripts/fixed_motion.py" 2>/dev/null || true

# ---------------------------------------------------------------- [1] RealSense
if [[ "$SKIP_SLAM" != "1" ]]; then
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

  # ---------------------------------------------------------------- [2] ORB-SLAM3
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

  # ---------------------------------------------------------------- [3] 等图像就绪
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
else
  echo ""
  echo "=== [1-3] SKIP_SLAM=1, 跳过 RealSense/ORB 检查 (假设已在跑) ==="
fi

# ---------------------------------------------------------------- [4] 起连狗桥
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

RECEIVER_ARGS=(--listen-ip 0.0.0.0 --listen-port "$LISTEN_PORT" \
  --robot-ip "$ROBOT_IP" --robot-port "$ROBOT_PORT" \
  --heartbeat-hz 4.0 --timeout 0.8 --startup-actions "$STARTUP_ACTIONS")
if [[ "$DRY_RUN" == "1" ]]; then
  RECEIVER_ARGS+=(--dry-run)
  echo "  [DRY-RUN] receiver 只打印命令, 不真的驱动狗"
fi
nohup python3 "$ROOT_DIR/controller/lite2_motion_receiver.py" "${RECEIVER_ARGS[@]}" \
  > /tmp/slam_logs/lite2_receiver.log 2>&1 &
echo $! > /tmp/slam_logs/lite2_receiver.pid
disown
sleep 2
echo "[OK] 连狗桥 PID=$(cat /tmp/slam_logs/lite2_receiver.pid) → $ROBOT_IP:$ROBOT_PORT"

# ---------------------------------------------------------------- [5] watchdog + mux
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

# ---------------------------------------------------------------- [6] 等定位
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
  echo "[WARN] 40s 内定位未 OK — fixed_motion 会自行等待(--wait-loc 30s):"
  tail -n 10 /tmp/slam_logs/watchdog.log || true
fi

# ---------------------------------------------------------------- [6.5] 标定 (可选)
if [[ "$TURN_CAL" == "1" ]]; then
  echo ""
  echo "=== [6.5] 自动标定 turn_sign / forward_axis ==="
  echo "  ⚠️ 狗将原地慢转 ~30° 并向前走 ~0.4m — 确保周围 0.5m 无障碍!"
  CAL_OUT="$(timeout 120 python3 "$ROOT_DIR/scripts/calibrate_turn_sign.py" \
    --forward-axis "$FORWARD_AXIS" 2>&1)" || true
  echo "$CAL_OUT"
  CAL_LINE="$(echo "$CAL_OUT" | grep '^CALIB ' | tail -n1 || true)"
  CAL_TS="$(echo "$CAL_LINE" | grep -o 'turn_sign=[+-]1\.0' | cut -d= -f2 || true)"
  CAL_FA="$(echo "$CAL_LINE" | grep -o 'forward_axis=[zx]' | cut -d= -f2 || true)"
  if [[ -n "$CAL_TS" ]]; then TURN_SIGN="$CAL_TS"; fi
  if [[ -n "$CAL_FA" ]]; then FORWARD_AXIS="$CAL_FA"; fi
  echo "  → 使用 TURN_SIGN=$TURN_SIGN FORWARD_AXIS=$FORWARD_AXIS"
else
  echo ""
  echo "=== [6.5] 跳过标定 (TURN_CAL=1 可开启), 用默认 TURN_SIGN=$TURN_SIGN FORWARD_AXIS=$FORWARD_AXIS ==="
fi

# ---------------------------------------------------------------- [7] 跑固定运动
echo ""
echo "=== [7] 跑固定运动 (fixed_motion.py) ==="
echo "  ⚠️ 狗将按序列执行: 前进→右转→… — 确保周围 1m 无障碍!"
python3 "$ROOT_DIR/scripts/fixed_motion.py" \
  --turn-sign "$TURN_SIGN" \
  --forward-axis "$FORWARD_AXIS" \
  "$@"

echo ""
echo "=== 完成 ==="
