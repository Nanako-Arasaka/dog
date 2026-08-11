#!/usr/bin/env bash
# ============================================================================
# check_onboard_audio.sh — 在 Jetson(感知主机, 装在狗背上)上运行
# 目的: 判断「机器狗自带扬声器」是否能被 Jetson 的 Linux 音频系统驱动播放。
#
# 背景: 绝影 Lite2 机身带一个扬声器(产品手册 P5-6 标注)，但官方 UDP 运动协议
#       没有「播放自定义音频」指令码。能跑我们代码的只有 Jetson(Linux 主机)。
#       所以"机器狗本地播报"= 在 Jetson 上跑节点 + 音频存 Jetson 本地 +
#       出声到「狗自带扬声器(若它物理连在 Jetson 音频口)」或「外置 USB 扬声器」。
#       本脚本帮你在现场确认狗自带扬声器到底是不是 Jetson 能认的声卡。
#
# 用法:
#   bash scripts/check_onboard_audio.sh [测试wav路径, 默认 output/audio/A_normal.wav]
# ============================================================================
set -u
WAV="${1:-output/audio/A_normal.wav}"
# 支持从项目根目录或任意目录调用
if [ ! -f "$WAV" ]; then
  # 尝试相对项目根
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  if [ -f "$PROJ_ROOT/$WAV" ]; then WAV="$PROJ_ROOT/$WAV"; fi
fi

echo "============================================================"
echo " [1] 列出所有声卡  (aplay -l)"
echo "============================================================"
if ! command -v aplay >/dev/null 2>&1; then
  echo "!!! aplay 不可用，请确认 alsa-utils 已安装: sudo apt install alsa-utils"
  exit 1
fi
aplay -l
echo

echo "============================================================"
echo " [2] 列出所有 PCM 设备 (aplay -L)"
echo "============================================================"
aplay -L
echo

echo "============================================================"
echo " [3] 当前默认声卡配置"
echo "============================================================"
[ -f "$HOME/.asoundrc" ] && { echo "--- ~/.asoundrc ---"; cat "$HOME/.asoundrc"; } || echo "无 ~/.asoundrc"
[ -f /etc/asound.conf ] && { echo "--- /etc/asound.conf ---"; cat /etc/asound.conf; } || echo "无 /etc/asound.conf"
echo

echo "============================================================"
echo " [4] 测试 wav 文件: $WAV"
echo "============================================================"
if [ ! -f "$WAV" ]; then
  echo "!!! 未找到测试 wav，请传入绝对路径: bash $0 /path/to/test.wav"
else
  echo "--- 4a) 播放到【默认卡】 ---"
  if timeout 8 aplay "$WAV" >/dev/null 2>&1; then
    echo "     >>> 默认卡播放成功(若此时狗/音箱出声，则默认卡就是你要用的)"
  else
    echo "     >>> 默认卡播放失败或无声音(可能被静音或默认卡不是扬声器)"
  fi
  echo
  echo "--- 4b) 逐卡播放(识别哪个是狗自带扬声器 / 哪个是外接) ---"
  # 解析 aplay -l 的 "card N: NAME, [DEV]" 行
  while IFS= read -r line; do
    if [[ "$line" =~ card\ ([0-9]+):\ ([^,]+),\ \[?([^],]+) ]]; then
      c="${BASH_REMATCH[1]}"; name="${BASH_REMATCH[2]}"; dev="${BASH_REMATCH[3]}"
      printf "  测试 card %s : %s (%s) ... " "$c" "$name" "$dev"
      if timeout 6 aplay -D "plughw:$c,0" "$WAV" >/dev/null 2>&1; then
        echo "可播放 ✅  -> 在 voice_broadcast_node 配置 device: plughw:$c,0"
      else
        echo "失败/无此设备 ❌"
      fi
    fi
  done < <(aplay -l 2>/dev/null)
fi
echo
echo "============================================================"
echo " 判读建议:"
echo "  - 若某张卡名含 tegra/rt565x/i2s/jetson/board 等，多半是狗自带扬声器"
echo "    连在 Jetson 板载音频口 → 直接用该卡即可，无需买外置(合规且省事)"
echo "  - 若只有 USB 设备(含 usb/audio) → 说明狗自带扬声器没连 Jetson，"
echo "    需按规则外接 1 个 USB 扬声器(见 config/guosai_final.yaml 的 engine/device)"
echo "============================================================"
