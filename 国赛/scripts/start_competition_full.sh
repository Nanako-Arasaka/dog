#!/usr/bin/env bash
# =============================================================================
# 国赛一键全流程启动 (start_competition_full.sh)
#
# 完整链路:
#   RealSense + ORB-SLAM3 → /camera_pose → localization_watchdog → navigator
#     → motion_mux → UDP 127.0.0.1:5005 → lite2_motion_receiver.py
#     → 有线网口 → 狗端 Lite2 (192.168.1.120:43893, 心跳 4Hz)
#   + cone_avoidance / inspection / arm_grasp / voice_broadcast
#
# 用法:
#   bash scripts/start_competition_full.sh                     # 默认狗端 192.168.1.120
#   bash scripts/start_competition_full.sh --robot-ip 192.168.1.121
#   bash scripts/start_competition_full.sh --dry-run           # 只检查,不起任何东西
#   bash scripts/start_competition_full.sh --no-realsense --no-orbslam3   # 栈已在跑时跳过
#
# 退出时自动: 停 lite2_motion_receiver + 发 /motion/stop 急停指令。
# =============================================================================
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/config/guosai_final.yaml"

ROBOT_IP="192.168.1.120"
ROBOT_PORT="43893"
LISTEN_PORT="5005"
DRY_RUN="false"
START_REALSENSE="true"
START_ORBSLAM3="true"
START_PERCEPTION="true"
START_ARM="true"
START_VOICE="true"
RECEIVER_LOG="/tmp/slam_logs/lite2_receiver.log"
RECEIVER_PID_FILE="/tmp/slam_logs/lite2_receiver.pid"

usage() {
  cat <<'EOF'
用法:
  bash scripts/start_competition_full.sh [options]

选项:
  --robot-ip IP        狗端 Lite2 主机 IP (默认 192.168.1.120)
  --robot-port PORT    狗端命令端口 (默认 43893)
  --listen-port PORT   本机 UDP 监听端口 (默认 5005, 与 motion_mux 一致)
  --dry-run            只做硬件/网络/权限检查,不起任何进程
  --no-realsense       不启动 RealSense (已在跑时用)
  --no-orbslam3        不启动 ORB-SLAM3 (已在跑时用)
  --no-perception      不启动巡检/避障感知
  --no-arm             不启动机械臂节点
  --no-voice           不启动语音播报
  -h, --help           帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip) ROBOT_IP="$2"; shift 2 ;;
    --robot-port) ROBOT_PORT="$2"; shift 2 ;;
    --listen-port) LISTEN_PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --no-realsense) START_REALSENSE="false"; shift ;;
    --no-orbslam3) START_ORBSLAM3="false"; shift ;;
    --no-perception) START_PERCEPTION="false"; shift ;;
    --no-arm) START_ARM="false"; shift ;;
    --no-voice) START_VOICE="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 未知选项: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p /tmp/slam_logs "$ROOT_DIR/logs"

# ---------------------------------------------------------------- 环境
if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
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
    set +u
    source "$setup_file"
    set -u
  fi
done
export GUOSAI_ROOT="$ROOT_DIR"
# orbslam3 install 缺可执行文件 (build 在 build/, install/lib 为空) → PATH hack
export PATH="/home/jetson/colcon_ws/build/orbslam3:$PATH"

# ---------------------------------------------------------------- 硬件检查
echo "==================== [0] 硬件检查 ===================="
FAIL=""

# RealSense
if ! lsusb 2>/dev/null | grep -q "8086:0b3a"; then
  echo "[FAIL] RealSense D435i 未检测到 (lsusb 无 8086:0b3a)"
  FAIL="realsense"
else
  echo "[OK] RealSense D435i 在线"
fi

# CH340 机械臂串口
if [[ -e /dev/ttyUSB0 ]]; then
  echo "[OK] /dev/ttyUSB0 存在 (CH340)"
  if [[ ! -w /dev/ttyUSB0 ]]; then
    echo "[WARN] /dev/ttyUSB0 不可写 (当前: $(stat -c %A /dev/ttyUSB0))"
    echo "      尝试 sudo chmod 666 /dev/ttyUSB0 ..."
    if sudo -n chmod 666 /dev/ttyUSB0 2>/dev/null; then
      echo "[OK] chmod 666 成功"
    else
      echo "[FAIL] 无法修改 /dev/ttyUSB0 权限 — 请手动执行: sudo chmod 666 /dev/ttyUSB0"
      FAIL="${FAIL:+$FAIL }ttyusb"
    fi
  fi
else
  echo "[FAIL] /dev/ttyUSB0 不存在 (CH340 未插/未加载) — modprobe ch34x 或 re-plug"
  FAIL="${FAIL:+$FAIL }ttyusb"
fi

# USB 声卡 (语音播报用, 非致命)
if lsusb 2>/dev/null | grep -qi "0d8c:0012"; then
  echo "[OK] USB 声卡 (C-Media) 在线"
else
  echo "[WARN] USB 声卡未检测到 — 语音播报将无输出 (可继续)"
fi

# 13 个航点
WP_COUNT=$(grep -c "name:" "/home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml" 2>/dev/null || echo 0)
if [[ "$WP_COUNT" == "13" ]]; then
  echo "[OK] 航点 13 个"
else
  echo "[FAIL] 航点数量 = $WP_COUNT (应为 13)"
  FAIL="${FAIL:+$FAIL }waypoints"
fi

# ---------------------------------------------------------------- 网络检查 (连狗)
echo ""
echo "==================== [1] 网络检查 (连狗 $ROBOT_IP:$ROBOT_PORT) ===================="
if command -v ip >/dev/null 2>&1; then
  ETH_IF=$(ip -4 addr show 2>/dev/null | awk '/^[0-9]+: (eth|en|usb)/{iface=$2; sub(":","",iface)} /inet /{print iface}' | head -1)
else
  ETH_IF=""
fi
if [[ -n "$ETH_IF" ]]; then
  ETH_IP=$(ip -4 addr show "$ETH_IF" 2>/dev/null | awk '/inet /{print $2}' | head -1)
  echo "[OK] 有线网口 $ETH_IF 存在, IP=$ETH_IP"
else
  ETH_IP=""
  echo "[WARN] 未找到有线网口 (eth/en/usb) — 连狗需要有线以太网"
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] 跳过 ping,跳过启动。硬件/网络检查完毕。"
  if [[ -n "$FAIL" ]]; then echo "[FAIL] 存在阻塞项: $FAIL"; exit 1; fi
  echo "[OK] 检查全部通过,可正式启动。"
  exit 0
fi

if [[ -n "$FAIL" ]]; then
  echo ""
  echo "======================================================"
  echo "[FAIL] 前置硬件检查未通过,拒绝启动。阻塞项: $FAIL"
  echo "       修复后重跑本脚本。"
  echo "======================================================"
  exit 1
fi

# ping 狗端 (软检查, 狗没开机也能继续但会警告)
if ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; then
  echo "[OK] 狗端 $ROBOT_IP 可达"
else
  echo "[WARN] 狗端 $ROBOT_IP 不可达 (ping 失败) — 确认: 有线网线已插 / 狗端开机 / Jetson 网口 IP 在 192.168.1.x 段"
  echo "      脚本继续 (lite2_receiver 会持续重发心跳, 狗上线即接管)"
fi

# ---------------------------------------------------------------- 起 SLAM 栈 (RealSense + ORB)
# 注意: 不能依赖 launch 里的 orbslam3.command 起 ORB —— 它没有 cd slam_maps,
# LoadAtlasFromFile 是相对 cwd 的 basename, cwd 不对会 "Load file not found" → pose=0。
# 这里手动起 (cd slam_maps + PATH hack), 再让 run_guosai_final.sh 跳过 SLAM。
echo ""
echo "==================== [2] 启动 SLAM 栈 (RealSense + ORB-SLAM3) ===================="
SLAM_READY="true"

# --- RealSense ---
if [[ "$START_REALSENSE" == "true" ]]; then
  if timeout 3 ros2 topic hz /camera/camera/color/image_raw >/dev/null 2>&1; then
    echo "[OK] RealSense 已在发布图像, 跳过启动"
  else
    nohup ros2 launch realsense2_camera rs_launch.py \
      enable_color:=true enable_depth:=true align_depth.enable:=true \
      > /tmp/slam_logs/realsense.log 2>&1 &
    echo $! > /tmp/slam_logs/realsense.pid
    disown
    echo "[OK] RealSense 启动中 (PID $(cat /tmp/slam_logs/realsense.pid))"
  fi
fi

# --- ORB-SLAM3 ---
if [[ "$START_ORBSLAM3" == "true" ]]; then
  if timeout 3 ros2 topic hz /camera_pose >/dev/null 2>&1; then
    echo "[OK] /camera_pose 已在发布, 跳过 ORB 启动"
  else
    cd /home/jetson/Desktop/guosai/slam_maps
    nohup ros2 run orbslam3 rgbd \
      "$GUOSAI_ROOT/controller/ORB_SLAM3/Vocabulary/ORBvoc.txt" \
      /home/jetson/Desktop/guosai/slam_maps/guosai_realsense_rgbd_localization_v4.yaml \
      --ros-args -p use_viewer:=false \
      -r /camera/color/image_raw:=/camera/camera/color/image_raw \
      -r /camera/aligned_depth_to_color/image_raw:=/camera/camera/aligned_depth_to_color/image_raw \
      > /tmp/slam_logs/orbslam3.log 2>&1 &
    echo $! > /tmp/slam_logs/orbslam3.pid
    disown
    cd "$ROOT_DIR"
    echo "[OK] ORB-SLAM3 启动中 (PID $(cat /tmp/slam_logs/orbslam3.pid), cwd=slam_maps)"
  fi
fi

echo "[INFO] 等待 SLAM 就绪 (最多 30s)..."
for i in $(seq 1 15); do
  if timeout 3 ros2 topic hz /camera_pose >/dev/null 2>&1; then
    echo "[OK] /camera_pose 已发布"
    SLAM_READY="true"
    break
  fi
  sleep 2
done
if [[ "$SLAM_READY" != "true" ]]; then
  echo "[WARN] 30s 内 /camera_pose 未发布 — 检查 /tmp/slam_logs/{realsense,orbslam3}.log"
  echo "      继续 (launch 会 blocked 直到定位可用, 这是预期安全行为)"
fi

# ---------------------------------------------------------------- 起 lite2_motion_receiver (连狗桥)
echo ""
echo "==================== [3] 启动 lite2_motion_receiver (连狗桥) ===================="
if [[ -f "$RECEIVER_PID_FILE" ]] && kill -0 "$(cat "$RECEIVER_PID_FILE")" 2>/dev/null; then
  echo "[OK] lite2_motion_receiver 已在运行 (PID $(cat "$RECEIVER_PID_FILE"))"
else
  nohup python3 "$ROOT_DIR/controller/lite2_motion_receiver.py" \
    --listen-ip 0.0.0.0 \
    --listen-port "$LISTEN_PORT" \
    --robot-ip "$ROBOT_IP" \
    --robot-port "$ROBOT_PORT" \
    --heartbeat-hz 4.0 \
    --timeout 0.8 \
    --startup-actions move_mode,walk_gait \
    > "$RECEIVER_LOG" 2>&1 &
  RECEIVER_PID=$!
  echo "$RECEIVER_PID" > "$RECEIVER_PID_FILE"
  disown
  sleep 2
  if kill -0 "$RECEIVER_PID" 2>/dev/null; then
    echo "[OK] lite2_motion_receiver PID=$RECEIVER_PID (监听 0.0.0.0:$LISTEN_PORT → $ROBOT_IP:$ROBOT_PORT)"
    echo "     日志: $RECEIVER_LOG"
  else
    echo "[FAIL] lite2_motion_receiver 启动失败 — 看 $RECEIVER_LOG"
    tail -5 "$RECEIVER_LOG" 2>/dev/null || true
    exit 1
  fi
fi

# ---------------------------------------------------------------- 清理钩子
RECEIVER_KILLED="false"
cleanup() {
  if [[ "$RECEIVER_KILLED" == "true" ]]; then return; fi
  RECEIVER_KILLED="true"
  echo ""
  echo "[STOP] 一键流程退出, 清理..."
  # 急停指令 (直接发给 lite2_receiver 端口)
  python3 - "$LISTEN_PORT" <<'PY' || true
import json, socket, sys
port = int(sys.argv[1])
payload = {"source": "start_competition_full", "vx": 0.0, "vy": 0.0, "wz": 0.0}
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(json.dumps(payload).encode(), ("127.0.0.1", port))
s.close()
PY
  if [[ -f "$RECEIVER_PID_FILE" ]]; then
    kill "$(cat "$RECEIVER_PID_FILE")" 2>/dev/null || true
    rm -f "$RECEIVER_PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- 起完整 launch
echo ""
echo "==================== [4] 启动完整国赛流程 (launch) ===================="
echo "[INFO] robot=$ROBOT_IP:$ROBOT_PORT  感知=$START_PERCEPTION  arm=$START_ARM  voice=$START_VOICE"
echo "[INFO] SLAM 由本脚本管理, launch 传 --no-realsense --no-orbslam3 (避免 cwd bug 重复起)"
echo ""
echo "[INFO] 若 SLAM 定位未建立 (pose≈0), launch 起来后导航会 blocked — 这是预期, 推狗 0.5m 重定位即可"
echo ""

FINAL_ARGS=(--no-realsense --no-orbslam3)
[[ "$START_PERCEPTION" == "false" ]] && FINAL_ARGS+=(--no-perception)
[[ "$START_ARM" == "false" ]] && FINAL_ARGS+=(--no-arm)

# run_guosai_final.sh 自带 preflight + launch + EXIT 急停
bash "$ROOT_DIR/scripts/run_guosai_final.sh" \
  --config "$CONFIG_PATH" \
  "${FINAL_ARGS[@]}" \
  --skip-preflight
