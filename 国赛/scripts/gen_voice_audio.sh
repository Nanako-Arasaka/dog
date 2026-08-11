#!/usr/bin/env bash
# 生成 12 个巡检播报音频（A/B/C/D × low/normal/high）。
#
# 规则依据：巡检识别 40 分 = 4 次播报 × 10 分（字母 5 + 状态 5）。
#   黄针=偏低(low)  绿针=正常(normal)  红针=偏高(high)
# 无声只有终端输出只给 2.5 分/次，故必须真正出声（Jetson 上 engine=aplay）。
#
# 依赖：macOS 自带 say + afconvert（生成 wav）。Jetson 上无需重新生成，直接拷贝本目录 wav 即可。
# 用法：bash scripts/gen_voice_audio.sh   （在仓库根目录执行）
# 重生成会覆盖 output/audio/*.wav。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIO_DIR="${ROOT}/output/audio"
mkdir -p "${AUDIO_DIR}"

# 中文女声 Tingting，发音清晰；mono 22050Hz 16-bit PCM，aplay/ffplay 通用兼容。
VOICE="${VOICE:-Tingting}"

for z in A B C D; do
  say -v "${VOICE}" "${z}区，仪表偏低" -o "${AUDIO_DIR}/${z}_low.aiff"
  say -v "${VOICE}" "${z}区，仪表正常" -o "${AUDIO_DIR}/${z}_normal.aiff"
  say -v "${VOICE}" "${z}区，仪表偏高" -o "${AUDIO_DIR}/${z}_high.aiff"
  afconvert -f WAVE -d LEI16@22050 "${AUDIO_DIR}/${z}_low.aiff"    "${AUDIO_DIR}/${z}_low.wav"
  afconvert -f WAVE -d LEI16@22050 "${AUDIO_DIR}/${z}_normal.aiff" "${AUDIO_DIR}/${z}_normal.wav"
  afconvert -f WAVE -d LEI16@22050 "${AUDIO_DIR}/${z}_high.aiff"   "${AUDIO_DIR}/${z}_high.wav"
  rm -f "${AUDIO_DIR}/${z}_low.aiff" "${AUDIO_DIR}/${z}_normal.aiff" "${AUDIO_DIR}/${z}_high.aiff"
done

echo "[gen_voice_audio] done: $(ls "${AUDIO_DIR}"/*.wav | wc -l | tr -d ' ') wavs in ${AUDIO_DIR}"
