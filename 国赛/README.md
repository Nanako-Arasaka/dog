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

## Jetson 与狗端运行分工

国赛现场建议采用“Jetson 算力板作为主计算节点，狗本体只保留实时执行和安全兜底”的部署方式。狗端算力有限，除底层运动执行、通信接收和安全停机外，不建议在狗端运行视觉识别、YOLO、OpenCV 读表、SLAM 或复杂任务状态机。

### Jetson 算力板负责

- 巡检识别：运行 `live_detect_yolo_opencv.py`、`gauge_reader.py` 和 YOLO 模型，识别 `zone_A/zone_B/zone_C/zone_D/gauge`，读取仪表盘角度并判断 `low/normal/high`。
- 异常区域记录：保存巡检阶段识别出的异常区域，例如 `A`、`C`，后续用于红条放置目标。
- 锥形桶避障感知：读取摄像头画面，检测锥桶位置，判断绕行、减速或停止策略，并输出简单运动指令。
- 放置区识别：检测纸箱上的 `A/B/C/D` 字母，判断当前是否到达目标异常区域。
- 机械臂高层任务控制：控制“抓红条、保持夹紧、等待目标放置区、识别到目标区后松爪”等任务逻辑。
- 建图、定位和路径决策：运行 ORB-SLAM3、目标点控制、高层速度规划和比赛流程状态机。
- 调试和日志：保存识别图、输出 JSON、打印中文巡检结果和现场调试信息。

### 狗本体负责

- 底层运动控制：接收 Jetson 下发的 `vx`、`vy`、`wz` 等速度指令，并执行步态、转向和停止。
- 通信接收桥：运行类似 `lite2_motion_receiver.py` 的轻量接收程序，将 Jetson 发来的 UDP/ROS/TCP 指令转换为狗底层控制命令。
- 安全 watchdog：如果一段时间收不到 Jetson 指令，狗端自动停止；如果指令异常，狗端限速或急停。
- 必要状态回传：回传运动状态、摔倒状态、电量、底层错误状态，必要时回传里程计或 IMU。

### 不建议放在狗端运行

- YOLO 检测
- OpenCV 仪表盘识别
- 锥桶视觉检测
- ORB-SLAM3
- 数据集整理或训练
- debug 图保存
- 复杂 ROS2 状态机
- 机械臂任务决策
- 放置区字母识别

### 推荐数据流

```text
摄像头
  ↓
Jetson：视觉识别 / 避障 / 巡检 / 放置区判断
  ↓
Jetson：任务状态机
  ↓
网络发送简单运动或机械臂指令
  ↓
狗端：底层运动执行 + watchdog
```

红条抓取与放置建议流程：

```text
Jetson 识别到异常区域 A/C
  ↓
Jetson 控制机械臂抓红条并保持夹紧
  ↓
狗移动到放置区
  ↓
Jetson 识别纸箱字母
  ↓
如果当前看到目标字母，例如 zone_A
  ↓
Jetson 发送放置命令
  ↓
机械臂松爪
```

## 状态转发层

`integration_bridge/` 是国赛模块之间的轻量转发层，只负责格式统一、ROS2 topic 转发和事件日志，不做视觉推理、路径规划或机械臂动作。

主要转发关系：

```text
/bridge/inspection_result  -> /inspection/all
/bridge/placement_zone     -> /placement/recognized_zone
```

Jetson 上启动：

```bash
python3 integration_bridge/bridge_node.py
```

本地无 ROS2 格式验证：

```powershell
python .\integration_bridge\bridge_node.py --no-ros --inspection-json "A:abnormal,B:normal,C:unknown,D:normal"
python .\integration_bridge\bridge_node.py --no-ros --placement-zone "zone_A"
```

详细说明见 `integration_bridge/README.md`。

## 后续优化方向

- 在 `fixed_detector.py` 内继续增强真实图像下的 A/B/C/D 模板匹配。
- 调整仪表盘角度阈值和 ROI 调试输出。
- 优化巡检融合的空间匹配策略。
- 准备 `A_low.wav` 等现场播报音频文件。
