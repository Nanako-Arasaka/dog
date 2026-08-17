#!/usr/bin/env bash
# =============================================================================
# 国赛 · Jetson ORB-SLAM3 一键编译脚本
# -----------------------------------------------------------------------------
# 覆盖: 环境检查 -> 系统依赖(apt) -> ORB-SLAM3 本体(build.sh)
#       -> ROS2 wrapper(zang09/ORB_SLAM3_ROS2) -> 编译 -> 验证
# 用法:
#   bash scripts/build_orbslam3.sh           # 完整流程(幂等, 已完成的步骤跳过)
#   bash scripts/build_orbslam3.sh --no-apt  # 跳过 apt(依赖已装过时)
#   bash scripts/build_orbslam3.sh --rebuild # 强制重新编译
#
# 建议在 tmux 里跑(防 SSH 断开中断):
#   tmux new -s orbslam
# =============================================================================
set -Eeuo pipefail

SKIP_APT="false"
REBUILD="false"
for arg in "$@"; do
  case "$arg" in
    --no-apt)  SKIP_APT="true" ;;
    --rebuild) REBUILD="true" ;;
    *) echo "[ERROR] 未知参数: $arg (支持 --no-apt / --rebuild)" >&2; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORBSLAM_DIR="$HOME/ORB_SLAM3"
COLCON_WS="$HOME/colcon_ws"
WRAPPER_DIR="$COLCON_WS/src/orbslam3_ros2"
VOCAB="$ORBSLAM_DIR/Vocabulary/ORBvoc.txt"
SETTINGS="$PROJECT_ROOT/jetson_payload/slam_maps/guosai_realsense_rgbd_FINAL.yaml"

log()  { echo -e "\033[1;36m[ORBSLAM]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
die()  { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

# ---------- 0. 环境检查 ----------
log "===== 0/5 环境检查 ====="
if [[ -n "${CONDA_DEFAULT_ENV:-}" || "$(command -v python3)" == *conda* ]]; then
  die "检测到 conda 环境。ORB-SLAM3/ROS2 需要系统 Python 3.10, 请先: conda deactivate 后重跑本脚本"
fi
PYVER="$(python3 --version 2>/dev/null | awk '{print $2}')"
case "$PYVER" in
  3.10*) ok "系统 Python: $PYVER" ;;
  *) die "需要 Python 3.10, 当前 $PYVER。请确认 python3 为 /usr/bin/python3" ;;
esac
command -v cmake git >/dev/null 2>&1 || die "缺少 cmake/git, 请先安装基础工具"

# ---------- 1. 系统依赖 ----------
if [[ "$SKIP_APT" == "true" ]]; then
  warn "跳过 apt 依赖(--no-apt)"
else
  log "===== 1/5 系统依赖(apt) ====="
  sudo apt update
  sudo apt install -y build-essential cmake git \
    libeigen3-dev libglew-dev libboost-dev libboost-thread-dev \
    libboost-filesystem-dev libpython3-dev nlohmann-json3-dev \
    libssl-dev libwayland-dev libxkbcommon-dev libegl1-mesa-dev
fi

# ---------- 2. ORB-SLAM3 本体 ----------
log "===== 2/5 ORB-SLAM3 本体 ====="
if [[ -d "$ORBSLAM_DIR/Thirdparty/DBoW2" && -d "$ORBSLAM_DIR/src" ]]; then
  ok "ORB-SLAM3 源码已就绪(DBoW2/g2o/Sophus 齐全), 跳过 clone"
else
  log "克隆官方 ORB-SLAM3 (网络不稳会自动重试)..."
  n=0
  until git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$ORBSLAM_DIR" 2>/dev/null; do
    n=$((n+1))
    if [[ $n -ge 3 ]]; then
      warn "clone 失败 3 次。建议从 Mac 打包拷贝源码, 更快更稳:"
      warn "  Mac:  tar -czf ~/orbslam3_src.tar.gz --exclude='.git' --exclude='build' ~/ORB_SLAM3 ~/Pangolin"
      warn "  Mac:  scp ~/orbslam3_src.tar.gz jetson@<IP>:/home/jetson/"
      warn "  Jetson: cd ~ && tar -xzf orbslam3_src.tar.gz"
      die "ORB-SLAM3 源码获取失败"
    fi
    warn "clone 失败(第 $n 次), 5 秒后重试..."
    rm -rf "$ORBSLAM_DIR"
    sleep 5
  done
fi
cd "$ORBSLAM_DIR"

# Pangolin: 官方新版 Thirdparty 不带, 需单独编译安装(系统级, ORB-SLAM3 find_package 依赖)
install_pangolin() {
  if find /usr /usr/local -name "PangolinConfig.cmake" 2>/dev/null | grep -q .; then
    ok "Pangolin 已安装"
    return 0
  fi
  local PDIR="$HOME/Pangolin"
  if [[ ! -f "$PDIR/CMakeLists.txt" ]]; then
    log "克隆 Pangolin (stevenlovegrove/Pangolin)..."
    git clone https://github.com/stevenlovegrove/Pangolin.git "$PDIR"
  fi
  log "编译安装 Pangolin (10-20 分钟)..."
  cd "$PDIR"
  # 最小依赖: 不跑 install_prerequisites.sh(会拉 ffmpeg/catch2/libc++ 大件, 慢网下 apt 卡死)
  # 只装 Pangolin 核心所需小包; ORB-SLAM3 Viewer 用不到 ffmpeg 视频/测试
  log "安装 Pangolin 最小依赖..."
  sudo apt install -y --no-install-suggests --no-install-recommends \
    libgl1-mesa-dev libegl1-mesa-dev libepoxy-dev libjpeg-dev libpng-dev \
    libglew-dev libeigen3-dev libwayland-dev libxkbcommon-dev wayland-protocols \
    ninja-build 2>&1 | tail -3 || warn "部分依赖安装失败(编译可能报缺头文件)"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PANGOLIN_FFMPEG=OFF \
    -DBUILD_PANGOLIN_VIDEO=OFF \
    -DBUILD_PANGOLIN_OPENNI=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF 2>&1 | tail -5 || die "Pangolin cmake 配置失败"
  cmake --build build -j2 2>&1 | tail -8 || die "Pangolin 编译失败, 查看上方日志"
  sudo cmake --install build 2>&1 | tail -3
  ok "Pangolin 安装完成"
}
install_pangolin
if [[ "$REBUILD" == "true" ]]; then
  warn "强制重编: 删除 build/"
  rm -rf build
fi
if [[ -f build/libORB_SLAM3.so ]]; then
  ok "libORB_SLAM3.so 已存在, 跳过本体编译"
else
  log "编译本体 (30-60 分钟, 日志 ~/orbslam_build.log)..."
  if [[ -f build.sh ]]; then
    sed -i 's/make -j4/make -j2/g' build.sh 2>/dev/null || true   # Jetson 内存小, 降并行度防 OOM
    chmod +x build.sh
    ./build.sh 2>&1 | tee ~/orbslam_build.log
  else
    warn "未找到 build.sh(打包时可能被排除), 改用纯 CMake 编译 Thirdparty + 本体..."
    (cd Thirdparty/DBoW2 && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j2) || die "DBoW2 编译失败"
    (cd Thirdparty/g2o   && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j2) || die "g2o 编译失败"
    (cd Thirdparty/Sophus && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j2) || die "Sophus 编译失败"
    mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j2 2>&1 | tee -a ~/orbslam_build.log && cd ..
  fi
  [[ -f build/libORB_SLAM3.so ]] || die "本体编译失败, 查看 ~/orbslam_build.log"
  ok "libORB_SLAM3.so 编译完成"
fi
if [[ ! -f "$VOCAB" && -f "$ORBSLAM_DIR/Vocabulary/ORBvoc.txt.tar.gz" ]]; then
  log "解压词袋(139MB)..."
  tar -xzf "$ORBSLAM_DIR/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORBSLAM_DIR/Vocabulary/"
  ok "词袋解压完成"
fi
# 让 config 的 vocabulary_path(仓库内路径)指向真实词袋
mkdir -p "$PROJECT_ROOT/controller/ORB_SLAM3/Vocabulary"
ln -sf "$VOCAB" "$PROJECT_ROOT/controller/ORB_SLAM3/Vocabulary/ORBvoc.txt" 2>/dev/null || true

# ---------- 3. ROS2 wrapper ----------
log "===== 3/5 ROS2 wrapper ====="
if [[ "$SKIP_APT" != "true" ]]; then
  sudo apt install -y ros-humble-vision-opencv ros-humble-message-filters
fi
mkdir -p "$COLCON_WS/src"
if [[ -f "$WRAPPER_DIR/CMakeLists.txt" && -f "$WRAPPER_DIR/package.xml" ]]; then
  ok "wrapper 源码已就绪, 跳过 clone"
elif [[ -d "$WRAPPER_DIR/.git" ]]; then
  log "更新 wrapper..."
  git -C "$WRAPPER_DIR" pull --rebase || true
else
  log "克隆 zang09/ORB_SLAM3_ROS2..."
  git clone https://github.com/zang09/ORB_SLAM3_ROS2.git "$WRAPPER_DIR"
fi
# 自动修改 CMakeLists: ORB_SLAM3_PATH + Python 路径
cd "$WRAPPER_DIR"
if grep -q "ORB_SLAM3_PATH" CMakeLists.txt; then
  sed -i "s|set(ORB_SLAM3_PATH.*|set(ORB_SLAM3_PATH \"$ORBSLAM_DIR\")|" CMakeLists.txt
  ok "CMakeLists ORB_SLAM3_PATH -> $ORBSLAM_DIR"
else
  warn "CMakeLists 未找到 ORB_SLAM3_PATH 行, 请手动 nano 修改"
fi
PYPATH="$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || true)"
if [[ -n "$PYPATH" ]]; then
  sed -i "s|PYTHON_PACKAGES_PATH.*|PYTHON_PACKAGES_PATH \"$PYPATH\"|; s|PYTHON_SITE_PACKAGES.*|PYTHON_SITE_PACKAGES \"$PYPATH\"|; s|/usr/lib/python3/dist-packages|\"$PYPATH\"|g" CMakeLists.txt 2>/dev/null || true
  ok "CMakeLists Python 路径 -> $PYPATH"
fi
cd "$COLCON_WS"
if [[ "$REBUILD" == "true" ]]; then
  warn "强制重编: 删除 build/install/log"
  rm -rf build install log
fi
set +u; source /opt/ros/humble/setup.bash 2>/dev/null || true; set -u
if [[ -d install/orbslam3 ]]; then
  ok "orbslam3 包已编译, 跳过"
else
  log "colcon build orbslam3 (约 20 分钟)..."
  colcon build --symlink-install --packages-select orbslam3 || {
    warn "编译失败。若报 sophus/se3.hpp 找不到, 依次执行:"
    warn "  cd $ORBSLAM_DIR/Thirdparty/Sophus/build && sudo make install"
    warn "  cd $COLCON_WS && colcon build --symlink-install --packages-select orbslam3"
    exit 1
  }
fi

# ---------- 4. 验证 ----------
log "===== 4/5 验证 ====="
set +u; source "$COLCON_WS/install/setup.bash" 2>/dev/null || true; set -u
if ros2 pkg prefix orbslam3 >/dev/null 2>&1; then
  ok "orbslam3 包就绪: $(ros2 pkg prefix orbslam3)"
else
  die "ros2 找不到 orbslam3 包, 请检查编译日志"
fi

# ---------- 5. 完成提示 ----------
log "===== 5/5 完成 ====="
cat <<EOF

全部编译完成!

启动命令(联调用):
  source /opt/ros/humble/setup.bash
  source $COLCON_WS/install/setup.bash
  ros2 run orbslam3 rgbd $VOCAB \\
    $SETTINGS

注意:
1. 地图加载: $SETTINGS 里 LoadAtlasFromFile=guosai_rgbd_map_v4,
   若 SLAM 起不来, 在运行目录软链地图:
     ln -sf $PROJECT_ROOT/jetson_payload/slam_maps/guosai_rgbd_map_FINAL.osa guosai_rgbd_map_v4.osa
2. 正式流程由 config/guosai_final.yaml 的 orbslam3.command 自动启动(路径已指向仓库内)
3. 启动后不崩溃退出、在等图像话题 = 成功
EOF
