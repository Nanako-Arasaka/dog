# 国赛巡检识别闭环

当前项目已清理为“巡检识别模块”专用代码，只保留相机取流、区域字母识别、仪表盘识别、巡检融合、TCP JSON 输出、远端接收网关和 speak_key 音频播放接口。

不再包含机器狗运动控制、路径规划、避障、SLAM、机械臂抓取/搬运/掉落处理、红条抓取或锥桶检测逻辑。

## 保留范围

- `camera_input.py`：mock / video / camera 三种输入源取流，输出统一 `VisionFrame`。
- `src/perception/detector/fixed_detector.py`：`detect_zone_letters()`、`detect_gauges()`、`poll_inspection()` 与 inspection fusion。
- `vision_server.py`：算力板端 TCP JSON 服务，只输出结构化视觉结果。
- `src/perception/remote_gateway.py`：机器狗本地接收远端巡检 JSON。
- `src/hardware/speaker/interface.py`：`AudioFileSpeaker.play(key)` 播放 `A_low.wav` 等预生成音频。
- `config/robot_config.json`：只保留相机、远端感知、字母、仪表、巡检融合和音频配置。

## TCP 请求

```json
{"req": "detect_zone_letters"}
{"req": "detect_gauges"}
{"req": "poll_inspection"}
```

`poll_inspection` 返回融合后的巡检结果，字段包括：

- `zone`: `A/B/C/D`
- `gauge_status`: `low/normal/high`
- `abnormal`: `true/false`
- `confidence`
- `letter_bbox`
- `gauge_bbox`
- `speak_key`: 例如 `A_low`、`B_normal`、`C_high`
- `timestamp`

## 启动 vision_server.py

mock 输入：

```powershell
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode mock
```

视频输入：

```powershell
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode video --source .\sample.mp4
```

摄像头输入：

```powershell
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode camera --source 0
```

调试图：

```powershell
python .\vision_server.py --mode camera --source 0 --save-debug-frames --letter-debug-save-roi --gauge-debug-save-roi --inspection-debug-save
```

默认调试目录：

- `output/debug_frames/`
- `output/debug_letters/`
- `output/debug_gauge/`
- `output/debug_inspection/`

## 音频播放

`AudioFileSpeaker.play(key)` 根据 `speak_key` 查找音频文件：

```text
output/audio/A_low.wav
output/audio/A_normal.wav
output/audio/A_high.wav
...
output/audio/D_high.wav
```

`say_async()` 只作为日志 fallback，不做语音合成。播放日志可在 `config/robot_config.json` 中开启：

```json
{
  "speaker": {
    "enabled": false,
    "engine": "mock",
    "audio_dir": "output/audio",
    "save_playback_log": true,
    "playback_log_path": "output/playback_log.jsonl"
  }
}
```

## 测试

```powershell
python -m pytest -q
python .\tools\test_camera_input.py
python .\tools\test_remote_perception_client.py
python .\tools\test_speaker_playback.py --save-playback-log
```

保留的关键测试：

- `tests/unit/test_fixed_detector.py`
- `tests/unit/test_audio_file_speaker.py`
- `tools/test_camera_input.py`
- `tools/test_remote_perception_client.py`

## YOLO 数据与仪表工具

整理拍摄照片为 YOLO 数据集：

```powershell
python .\tools\dataset_builder.py --raw-dir .\data\raw_photos --out-dir .\data\inspection_yolo --workers 8
```

检查 YOLO 标签并画回 debug 图：

```powershell
python .\tools\label_check.py --dataset-root .\data\inspection_yolo --split train --debug-dir .\output\debug_labels --workers 8
```

训练 YOLO：

```powershell
pip install ultralytics
yolo detect train model=yolov8n.pt data=.\data\inspection_yolo\dataset.yaml imgsz=640 epochs=80 batch=16 workers=4
```

使用 YOLO 权重定位区域字母和仪表盘：

```powershell
python .\tools\yolo_locator.py --model .\runs\detect\train\weights\best.pt --source .\data\inspection_yolo\images\test --debug-dir .\output\debug_yolo --output-json .\output\yolo_detections.json
```

读取单个仪表盘 ROI 状态：

```powershell
python .\tools\gauge_reader.py --image .\sample.jpg --bbox 100,80,260,240 --zone A --debug-dir .\output\debug_gauge_roi
```

单张图闭环推理：

```powershell
python .\tools\inspection_pipeline_demo.py --source .\data\inspection_yolo\images\test\sample.jpg --model .\runs\detect\train\weights\best.pt --output-json .\output\inspection_single.json --debug-dir .\output\debug_inspection_pipeline
```

文件夹批量闭环推理：

```powershell
python .\tools\inspection_pipeline_demo.py --source .\data\inspection_yolo\images\test --model .\runs\detect\train\weights\best.pt --output-json .\output\inspection_batch.json --debug-dir .\output\debug_inspection_pipeline
```

完整真实照片流程见 `docs/inspection_workflow.md`。

## 后续优化方向

- 在 `fixed_detector.py` 内继续增强真实图像下的 A/B/C/D 模板匹配。
- 调整仪表盘角度阈值和 ROI 调试输出。
- 优化巡检融合的空间匹配策略。
- 准备 `A_low.wav` 等现场播报音频文件。
