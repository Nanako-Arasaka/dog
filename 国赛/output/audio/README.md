# 巡检语音播报音频

12 个预录播报音频，供 `nodes/voice_broadcast_node.py` 按区播放。

## 命名规范

`{zone}_{state}.wav`，zone ∈ A/B/C/D，state ∈ low/normal/high。

| 文件 | 文本内容 | 含义 |
|---|---|---|
| A_low.wav / B_low.wav / C_low.wav / D_low.wav | "X区，仪表偏低" | 黄针=偏低(异常) |
| A_normal.wav / ... / D_normal.wav | "X区，仪表正常" | 绿针=正常 |
| A_high.wav / ... / D_high.wav | "X区，仪表偏高" | 红针=偏高(异常) |

## 规则依据

巡检识别 40 分 = 4 次播报 × 10 分（字母 5 + 状态 5）。
无声只有终端输出只给 2.5 分/次（直接丢 20 分），故必须真正出声。

## 数据来源

- 主用：`/inspection/all_detailed`（格式 `A:low,B:normal,C:high,D:normal`，保留偏低/偏高）
- 兜底：`/inspection/all`（格式 `A:abnormal,B:normal,...`，abnormal 无法区分偏低/偏高，该区跳过）

## 格式

- 编码：RIFF WAVE，PCM 16-bit，mono，22050 Hz（aplay / ffplay 通用兼容）
- 语音：macOS `say` 中文女声 Tingting

## 重新生成（仅 Mac）

```bash
bash scripts/gen_voice_audio.sh   # 在仓库根目录执行，覆盖本目录 wav
```

环境变量 `VOICE` 可换语音（如 `VOICE=Sinji bash scripts/gen_voice_audio.sh`）。

## Jetson 部署

无需在 Jetson 上重新生成，直接把整个 `output/audio/` 目录拷到 Jetson 对应路径
（`guosai_final.yaml` 的 `voice_broadcast.audio_dir`，默认 `${GUOSAI_ROOT}/output/audio`），
并把 `voice_broadcast.engine` 从 `mock` 改成 `aplay`。
