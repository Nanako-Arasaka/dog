#!/usr/bin/env bash
# =============================================================================
# 国赛 · 新算力板一键环境安装脚本
# -----------------------------------------------------------------------------
# 功能: 仓库拉取 -> 系统依赖(apt) -> ROS2 Humble -> Python 依赖(pip)
#       -> 仓库自带 ROS2 包(colcon 编译) -> ORB 词汇表/数据校验 -> preflight
#
# 用法:
#   bash scripts/setup_new_board.sh [选项]
#
# 选项:
#   --repo <URL>      仓库地址   (默认: https://github.com/Nanako-Arasaka/dog.git)
#   --branch <name>   拉取分支   (默认: main)
#   --target <DIR>    clone 目标 (默认: ~/dog_repo; 已在仓库内则忽略)
#   --jetpack <5|6>   Jetson JetPack 版本 (默认自动检测; 非 Jetson 忽略)
#   --build-orbslam3  自动编译 ORB-SLAM3 (默认只检测+解压词汇表, 不编译)
#   --skip-ros        跳过 ROS2 安装   (已装时)
#   --skip-orbslam    跳过 ORB-SLAM3 检测 (不编译也不解压词汇表)
#   --no-arm          跳过机械臂 colcon 编译
#   --yes             apt/pip 全程自动确认
#   -h, --help        显示帮助
#
# 说明:
#   - 幂等: 已安装的步骤自动跳过, 可重复执行。
#   - Jetson aarch64 的 torch 必须走 NVIDIA 官方源(见 install_torch)。
#   - 比赛规则禁用 AprilTag: 不安装 dt-apriltags / libapriltag-dev,
#     config/guosai_final.yaml 的 tag_localizer 段已全部禁用。
# =============================================================================
set -Eeuo pipefail

# ----------------------------- 默认参数 ------------------------------------
REPO_URL="https://github.com/Nanako-Arasaka/dog.git"
BRANCH="main"
TARGET_DIR="$HOME/dog_repo"
JETPACK=""                 # 自动检测
BUILD_ORBSLAM="false"
SKIP_ROS="false"
SKIP_ORBSLAM="false"
NO_ARM="false"
ASSUME_YES="false"

# ----------------------------- 参数解析 ------------------------------------
usage() {
  cat <<'EOF'
用法: bash scripts/setup_new_board.sh [选项]

选项:
  --repo <URL>      仓库地址   (默认: https://github.com/Nanako-Arasaka/dog.git)
  --branch <name>   拉取分支   (默认: main)
  --target <DIR>    clone 目标 (默认: ~/dog_repo; 已在仓库内则忽略)
  --jetpack <5|6>   Jetson JetPack 版本 (默认自动检测; 非 Jetson 忽略)
  --build-orbslam3  自动编译 ORB-SLAM3 (默认只检测+解压词汇表, 不编译)
  --skip-ros        跳过 ROS2 安装   (已装时)
  --skip-orbslam    跳过 ORB-SLAM3 检测 (不编译也不解压词汇表)
  --no-arm          跳过机械臂 colcon 编译
  --yes             apt/pip 全程自动确认
  -h, --help        显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       REPO_URL="$2"; shift 2 ;;
    --branch)     BRANCH="$2";   shift 2 ;;
    --target)     TARGET_DIR="$2"; shift 2 ;;
    --jetpack)    JETPACK="$2";  shift 2 ;;
    --build-orbslam3) BUILD_ORBSLAM="true"; shift ;;
    --skip-ros)   SKIP_ROS="true";   shift ;;
    --skip-orbslam) SKIP_ORBSLAM="true"; shift ;;
    --no-arm)     NO_ARM="true"; shift ;;
    --yes)        ASSUME_YES="true"; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "[ERROR] 未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

# 有 sudo 的机器上 --yes 自动确认; 无 sudo 权限的容器环境忽略
SUDO=(sudo)
if [[ "$ASSUME_YES" == "true" ]]; then
  SUDO=(sudo -n)
  PIP_EXTRA="--no-input"
else
  PIP_EXTRA=""
fi

log()  { echo -e "\033[1;36m[SETUP]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
die()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

# ----------------------------- 环境检测 ------------------------------------
ARCH="$(uname -m)"
if [[ -f /etc/nv_tegra_release ]]; then
  if grep -q "R36" /etc/nv_tegra_release 2>/dev/null; then
    JETPACK="${JETPACK:-6}"
  else
    JETPACK="${JETPACK:-5}"
  fi
  log "检测到 Jetson (JetPack ${JETPACK}, ${ARCH})"
elif [[ "$ARCH" == "aarch64" ]]; then
  warn "ARM64 但未检测到 JetPack, torch 将按 CPU 版本安装"
  JETPACK="${JETPACK:-}"
else
  JETPACK=""
fi

# 需要 sudo 权限(apt 步骤)
if ! "${SUDO[@]}" true 2>/dev/null; then
  die "需要 sudo 权限执行 apt 安装。请以有 sudo 权限的用户运行, 或用 --yes 自动确认。"
fi

# ----------------------------- 1. 拉取仓库 --------------------------------
fetch_repo() {
  local here
  here="$(pwd)"
  if [[ -f "$here/config/guosai_final.yaml" && -d "$here/.git" ]]; then
    ROOT_DIR="$here"
    log "已在仓库内: $ROOT_DIR"
    git pull --rebase || warn "git pull 失败(可能无网络/未配置 remote), 继续使用现有代码"
  else
    mkdir -p "$(dirname "$TARGET_DIR")"
    if [[ -d "$TARGET_DIR/.git" ]]; then
      ROOT_DIR="$TARGET_DIR"
      log "目标目录已是仓库: $TARGET_DIR"
      git -C "$ROOT_DIR" pull --rebase || warn "git pull 失败, 继续使用现有代码"
    else
      log "克隆仓库 $REPO_URL (分支 $BRANCH) -> $TARGET_DIR"
      git clone -b "$BRANCH" "$REPO_URL" "$TARGET_DIR" \
        || die "git clone 失败: $REPO_URL"
      ROOT_DIR="$TARGET_DIR"
    fi
    cd "$ROOT_DIR"
  fi
  export GUOSAI_ROOT="$ROOT_DIR"
  # LFS 大文件(SLAM 地图 .osa)
  if command -v git-lfs >/dev/null 2>&1; then
    git lfs pull || warn "git lfs pull 失败, 地图文件可能缺失"
  else
    warn "git-lfs 未安装, 跳过 LFS 拉取(稍后 apt 会装, 可重跑本脚本)"
  fi
  ok "仓库就绪: $ROOT_DIR"
}

# ----------------------------- 2. 系统依赖(apt) ---------------------------
APT_BASE=(
  build-essential cmake git git-lfs curl wget
  python3-pip python3-venv
  libeigen3-dev libboost-all-dev
  alsa-utils ffmpeg
)
APT_ROS=(
  ros-humble-desktop
  ros-humble-cv-bridge
  ros-humble-realsense2-camera
  ros-humble-launch ros-humble-launch-ros
  ros-humble-ament-index-python
  python3-colcon-common-extensions
)

install_apt() {
  log "安装系统依赖..."
  "${SUDO[@]}" apt-get update -y
  "${SUDO[@]}" apt-get install -y "${APT_BASE[@]}"

  if [[ "$SKIP_ROS" != "true" ]]; then
    if [[ -d /opt/ros/humble ]]; then
      ok "ROS2 Humble 已存在, 跳过"
    else
      install_ros2
    fi
  else
    warn "--skip-ros: 跳过 ROS2 相关包安装(仅装基础包)"
  fi
}

install_ros2() {
  log "安装 ROS2 Humble (耗时较长)..."
  "${SUDO[@]}" apt-get install -y gnupg lsb-release software-properties-common
  local key=/usr/share/keyrings/ros-archive-keyring.gpg
  if [[ ! -f "$key" ]]; then
    "${SUDO[@]}" curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o "$key"
  fi
  local codename
  codename="$(lsb_release -cs)"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=$key] http://packages.ros.org/ros2/ubuntu $codename main" \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/ros2.list >/dev/null
  "${SUDO[@]}" apt-get update -y
  for pkg in "${APT_ROS[@]}"; do
    if "${SUDO[@]}" apt-get install -y "$pkg"; then
      ok "apt 安装 $pkg"
    else
      warn "apt 安装 $pkg 失败(可能无该架构包), 记下继续"
    fi
  done
}

# ----------------------------- 3. Python 依赖(pip) ------------------------
PIP_BASE=(
  numpy opencv-python PyYAML pyserial Pillow pytest
  PyMuPDF pyrealsense2 openni pyorbbecsdk
)

pip_install() {
  # 参数1: 描述; 其余: 包名
  local desc="$1"; shift
  if [[ $# -eq 0 ]]; then return 0; fi
  log "pip 安装: $desc"
  python3 -m pip install --no-cache-dir $PIP_EXTRA "$@" \
    || warn "pip 安装失败: $desc (请手动重试: pip install $*)"
}

install_torch() {
  if python3 -c "import torch" 2>/dev/null; then
    ok "torch 已存在: $(python3 -c 'import torch; print(torch.__version__)')"
    return 0
  fi
  if [[ "$ARCH" == "aarch64" && -n "$JETPACK" ]]; then
    # Jetson 必须用 NVIDIA 官方 wheel(带 CUDA), pip 默认源装不上
    local idx
    if [[ "$JETPACK" == "6" ]]; then
      idx="https://developer.download.nvidia.com/compute/redist/jp/v61x/pytorch"
    else
      idx="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch"
    fi
    log "Jetson: 尝试 NVIDIA 官方 torch (JetPack $JETPACK)..."
    if python3 -m pip install --no-cache-dir $PIP_EXTRA torch torchvision --index-url "$idx"; then
      ok "NVIDIA torch 安装成功"
    else
      warn "NVIDIA index 安装失败。请按官方文档手动装 torch 后重跑本脚本:"
      warn "  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html"
      warn "  (装好 torch 后执行: pip install ultralytics)"
      return 1
    fi
  else
    python3 -m pip install --no-cache-dir $PIP_EXTRA torch torchvision \
      || warn "torch 安装失败, 请手动安装后重跑"
  fi
}

install_python_deps() {
  log "安装 Python 依赖..."
  pip_install "基础库" "${PIP_BASE[@]}"
  install_torch || true   # torch 失败不中断脚本, ultralytics 会按 torch 就绪状态跳过
  # ultralytics 依赖 torch, torch 装好才装
  if python3 -c "import torch" 2>/dev/null; then
    pip_install "ultralytics" ultralytics
  else
    warn "torch 未就绪, 跳过 ultralytics(YOLO 依赖)"
  fi
  # 若系统已有 OpenCV(JetPack 自带), 卸掉 pip 版避免覆盖 CUDA/GStreamer 能力
  if python3 -c "import cv2" 2>/dev/null; then
    ok "系统 OpenCV 可用: $(python3 -c 'import cv2; print(cv2.__version__)')"
    if python3 -m pip show opencv-python >/dev/null 2>&1; then
      warn "检测到 pip 版 opencv-python 覆盖了系统版, 建议卸载: pip uninstall -y opencv-python opencv-python-headless"
    fi
  fi
}

# ----------------------------- 4. 仓库 ROS2 包(colcon) --------------------
source_ros() {
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    set +u; source /opt/ros/humble/setup.bash; set -u
  fi
}

build_colcon() {
  source_ros
  if ! command -v colcon >/dev/null 2>&1; then
    warn "colcon 不可用, 跳过仓库包编译(先安装 ROS2 / colcon 后重跑)"
    return 0
  fi

  # ros_robot_controller_msgs: 必须先编(机械臂自定义消息), 装到 arm_grasp/install
  if [[ -d "$ROOT_DIR/arm_grasp/ros_robot_controller_msgs" ]]; then
    log "编译 ros_robot_controller_msgs -> arm_grasp/install"
    ( cd "$ROOT_DIR/arm_grasp/ros_robot_controller_msgs" \
      && colcon build --install-base ../install ) \
      || warn "msgs 编译失败, 参考 arm_grasp/readme.md workaround"
  fi

  # ros_robot_controller: 串口桥接(嵌套包, 单独编)
  if [[ -d "$ROOT_DIR/arm_grasp/ros_robot_controller" ]]; then
    log "编译 ros_robot_controller"
    ( cd "$ROOT_DIR/arm_grasp/ros_robot_controller" && colcon build ) \
      || warn "ros_robot_controller 编译失败"
  fi

  # arm_grasp: 机械臂主包
  if [[ -d "$ROOT_DIR/arm_grasp" ]]; then
    log "编译 arm_grasp"
    ( cd "$ROOT_DIR/arm_grasp" && colcon build ) \
      || warn "arm_grasp 编译失败"
  fi

  # lite2_navigation_bridge: 导航 goal_controller
  if [[ -d "$ROOT_DIR/controller/colcon_ws/src/lite2_navigation_bridge" ]]; then
    log "编译 lite2_navigation_bridge"
    ( cd "$ROOT_DIR/controller/colcon_ws" \
      && colcon build --packages-select lite2_navigation_bridge ) \
      || warn "lite2_navigation_bridge 编译失败"
  fi

  # SDK 软链: serial_bridge_node 顶层 import ros_robot_controller_sdk
  # 运行时靠 PYTHONPATH=$HOME + 本软链 (见 scripts/run_guosai_final.sh:170)
  ln -sf "$ROOT_DIR/arm_grasp/ros_robot_controller_sdk.py" "$HOME/ros_robot_controller_sdk.py"
  ok "SDK 软链: $HOME/ros_robot_controller_sdk.py"

  # 手动 cmake install msgs(README 要求的嵌套包 workaround, 若 colcon 未产出 .so)
  local msgs_install="$ROOT_DIR/arm_grasp/install/share/ros_robot_controller_msgs"
  if [[ ! -d "$msgs_install" ]]; then
    warn "未找到 $msgs_install, 请按 arm_grasp/readme.md 手动 cmake install msgs"
  fi
}

# ----------------------------- 5. ORB-SLAM3 --------------------------------
setup_orbslam3() {
  if [[ "$SKIP_ORBSLAM" == "true" ]]; then
    warn "--skip-orbslam: 跳过 ORB-SLAM3"
    return 0
  fi

  local vocab_dir="$ROOT_DIR/controller/ORB_SLAM3/Vocabulary"
  if [[ -f "$vocab_dir/ORBvoc.txt" ]]; then
    ok "ORB 词汇表已就绪"
  elif [[ -f "$vocab_dir/ORBvoc.txt.tar.gz" ]]; then
    log "解压 ORB 词汇表(139MB, 需要约 2 分钟)..."
    tar -xzf "$vocab_dir/ORBvoc.txt.tar.gz" -C "$vocab_dir" \
      && ok "ORB 词汇表解压完成" || warn "ORB 词汇表解压失败"
  else
    warn "未找到 ORB 词汇表(仓库可能缺文件)"
  fi

  # 仓库内只有 src, 完整 ORB-SLAM3 需整包拷贝或源码编译
  local orbslam_src="$ROOT_DIR/controller/ORB_SLAM3"
  local system_orbslam=""
  if [[ -d "$HOME/ORB_SLAM3" ]]; then system_orbslam="$HOME/ORB_SLAM3"; fi
  if [[ -n "$system_orbslam" && -f "$system_orbslam/build/libORB_SLAM3.so" ]]; then
    ok "检测到已编译 ORB-SLAM3: $system_orbslam (可直接使用)"
    return 0
  fi

  if [[ "$BUILD_ORBSLAM" == "true" ]]; then
    log "自动编译 ORB-SLAM3 (耗时 30-60 分钟)..."
    if [[ ! -d "$HOME/ORB_SLAM3" ]]; then
      git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$HOME/ORB_SLAM3"
    fi
    ( cd "$HOME/ORB_SLAM3" && chmod +x build.sh && ./build.sh ) \
      && ok "ORB-SLAM3 编译完成" || warn "ORB-SLAM3 编译失败, 请查官方文档"
    warn "还需编译 ROS2 wrapper(rgbd 节点)才能 ros2 run orbslam3, 见 https://github.com/zang09/ORB-SLAM3-ROS2"
  else
    warn "未检测到已编译的 ORB-SLAM3。定位功能依赖它, 二选一:"
    warn "  a) 从旧板拷贝 /home/*/ORB_SLAM3 整个目录到本机"
    warn "  b) 重跑本脚本加 --build-orbslam3 自动编译"
  fi
}

# ----------------------------- 6. 数据与验证 --------------------------------
verify_assets() {
  log "校验数据文件..."
  local missing=0
  for f in best_7class.pt cone_avoidance/scripts/cone_yolo_best.pt; do
    if [[ -f "$ROOT_DIR/$f" ]]; then ok "模型: $f"; else warn "缺失: $f"; missing=1; fi
  done
  local wav_count
  wav_count="$(ls "$ROOT_DIR"/output/audio/*.wav 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$wav_count" -ge 12 ]]; then ok "语音 wav: $wav_count 个"; else warn "语音 wav 不足(需 12): $wav_count"; missing=1; fi
  return "$missing"
}

run_preflight() {
  log "运行 preflight 自检(只查依赖与路径, 不启动硬件)..."
  source_ros
  export PYTHONPATH="$HOME:${PYTHONPATH:-}"
  python3 -c "import numpy, cv2, yaml, serial, PIL, pyorbbecsdk; print('核心 Python 依赖 OK')" \
    || warn "部分 Python 依赖缺失"
  python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null \
    || warn "torch 未就绪"
  python3 -c "import ultralytics; print('ultralytics', ultralytics.__version__)" 2>/dev/null \
    || warn "ultralytics 未就绪"
  if command -v ros2 >/dev/null 2>&1; then
    python3 "$ROOT_DIR/scripts/preflight_guosai_final.py" \
      --config "$ROOT_DIR/config/guosai_final.yaml" \
      --root "$ROOT_DIR" --dry-run true || warn "preflight dry-run 有告警/错误(多为硬件/地图路径, 属正常)"
  else
    warn "ros2 命令不可用, 跳过 preflight"
  fi
}

check_config_paths() {
  # 检查 config 里 SLAM 相关路径是否指向仓库内的真实文件(新板无 /home/jetson 旧路径)
  log "检查 config/guosai_final.yaml 的路径指向..."
  python3 - "$ROOT_DIR" <<'PY' || true
import os, sys, yaml
root = Path = __import__("pathlib").Path(sys.argv[1])
cfg = yaml.safe_load((root / "config/guosai_final.yaml").read_text(encoding="utf-8")) or {}
slam = cfg.get("slam", {})
checks = {
    "slam.map_path": slam.get("map_path", ""),
    "slam.settings_yaml": slam.get("settings_yaml", ""),
    "slam.waypoints_yaml": slam.get("waypoints_yaml", ""),
}
for label, val in checks.items():
    v = str(val).replace("${GUOSAI_ROOT}", str(root)).replace("$GUOSAI_ROOT", str(root))
    ok = os.path.exists(v)
    print(f"  {'[OK] ' if ok else '[缺] '}{label}: {val}")
    if not ok:
        print(f"        新板需指向仓库内文件: 运行 python3 scripts/repair_guosai_final_config.py --config config/guosai_final.yaml --root {root} --dry-run 查看候选, 确认后去掉 --dry-run 写入")
PY
}

# =============================== 主流程 ====================================
log "===== 国赛新算力板一键环境安装 ====="
log "架构: $ARCH | JetPack: ${JETPACK:-非Jetson} | 仓库: $REPO_URL"

fetch_repo
install_apt
install_python_deps
if [[ "$NO_ARM" != "true" ]]; then
  build_colcon
else
  warn "--no-arm: 跳过机械臂 colcon 编译"
fi
setup_orbslam3
check_config_paths
verify_assets || true
run_preflight

log "===== 安装流程结束 ====="
echo ""
ok "后续现场动作(硬件相关, 脚本不代做):"
echo "  1. 相机编号: ls /dev/video*  确认 RealSense / Astra 设备号, 对 config/guosai_final.yaml"
echo "  2. 语音声卡: bash scripts/check_onboard_audio.sh 后把 device 改为实际 plughw:X,0"
echo "  3. ORB-SLAM3: $([ "$BUILD_ORBSLAM" = "true" ] || [ "$SKIP_ORBSLAM" = "true" ] && echo '已跳过/已编译' || echo '需按上文 a/b 处理')"
echo "  4. 航点: bash scripts/guosai_onekey.sh collect 采集真实航点"
echo "  5. AprilTag: 标定后把 config/guosai_final.yaml 的 tag_localizer.enabled 置 true"
echo ""
echo "完整依赖说明见: $ROOT_DIR/docs/新算力板依赖安装清单.md"
