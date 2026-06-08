# 国赛代码架构文档

## 概述

本项目是绝影 Lite2 四足机器人参加国赛的分层软件框架。机器狗本地负责任务状态机、运动控制、机械臂控制和语音播放；外接算力板负责相机取流和视觉检测，并通过 TCP JSON 把结构化感知结果发回机器狗本地。

当前架构保持 `Mission -> Gateway -> Hardware` 的分层边界：

- `mission/` 只做任务流程调度和决策。
- `perception/` 只返回结构化视觉结果，不控制机器狗、机械臂或语音。
- `hardware/` 负责相机、机械臂、扬声器等硬件抽象。
- `dog_sdk/` 负责绝影 UDP 协议和遥测。
- `vision_server.py` 是算力板端进程，负责取流、检测、TCP JSON 响应。

## 分层架构

```text
机器狗本地
┌──────────────────────────────────────────────┐
│ src/main.py                                  │
│ app/config.py, app/container.py              │
├──────────────────────────────────────────────┤
│ mission/national_stage.py                    │
│ 11 个执行阶段状态机                           │
├──────────────┬──────────────┬────────────────┤
│ perception/  │ navigation/  │ runtime/       │
│ gateway.py   │ gateway.py   │ controller.py  │
│ remote_gateway.py           │ speaker.py     │
├──────────────┼──────────────┼────────────────┤
│ hardware/arm │ hardware/camera │ dog_sdk/     │
│ hardware/speaker             │ UDP 协议栈     │
└──────────────────────────────────────────────┘

算力板端
┌──────────────────────────────────────────────┐
│ vision_server.py                             │
├──────────────────────────────────────────────┤
│ camera_input.py                              │
│ CameraInput -> VisionFrame                   │
├──────────────────────────────────────────────┤
│ src/perception/detector/fixed_detector.py    │
│ FixedDetectionPipeline                       │
└──────────────────────────────────────────────┘
```

## 任务流程

`NationalStageMission` 当前是 11 个执行阶段加终态：

```text
INIT
  -> OBSTACLE_APPROACH
  -> OBSTACLE_DETECT
  -> OBSTACLE_CROSS
  -> INSPECTION_NAV
  -> INSPECTION_SCAN
  -> INSPECTION_READ
  -> PICKUP_PLAN
  -> PICKUP_NAV
  -> PICKUP_GRAB
  -> PICKUP_TRANSPORT
  -> PICKUP_PLACE
  -> DONE / FAILED / STOPPED
```

## 关键模块职责

| 模块 | 职责 | 关键类/文件 |
|------|------|-------------|
| `app/` | 配置加载和依赖注入 | `AppConfig`, `AppContainer` |
| `core/` | 领域类型、枚举、异常 | `Zone`, `MeterStatus`, `MissionPhase`, `GaugeReading` |
| `mission/` | 国赛状态机和任务调度 | `NationalStageMission` |
| `perception/gateway.py` | 纯感知接口定义 | `PerceptionGateway` |
| `perception/remote_gateway.py` | 机器狗本地 TCP 感知客户端 | `RemotePerceptionGateway` |
| `navigation/` | 导航抽象和 mock 导航 | `NavigationGateway`, `MockNavigator` |
| `hardware/arm/` | 机械臂原子动作抽象 | `ArmGateway`, `MockArm` |
| `hardware/speaker/` | 预录音频播放抽象 | `SpeakerGateway`, `AudioFileSpeaker` |
| `runtime/controller.py` | 机器狗控制循环 | `DogController` |
| `dog_sdk/` | 绝影 UDP 指令与遥测 | `commands.py`, `transport.py` |
| `camera_input.py` | 算力板端取流层 | `CameraInput`, `VisionFrame` |
| `vision_server.py` | 算力板 TCP JSON 服务端 | `VisionServer` |
| `perception/detector/fixed_detector.py` | 固定检测、字母识别和仪表识别 | `FixedDetectionPipeline` |
| `assets/templates/letters/` | A/B/C/D 模板资源 | `A.png`, `B.png`, `C.png`, `D.png` |

## 远程感知协议

协议是 JSON Lines over TCP，每行一条 JSON。

机器狗本地发送：

```json
{"req": "detect_obstacles"}
{"req": "detect_zone_letters"}
{"req": "detect_gauges"}
{"req": "poll_inspection"}
{"req": "detect_red_strips"}
{"req": "estimate_target_pose", "target": "strip"}
```

算力板返回：

```json
{"type": "obstacles", "detections": [], "timestamp": 123.456}
{"type": "zone_letters", "detections": [], "timestamp": 123.456}
{"type": "gauges", "detections": [], "timestamp": 123.456}
{"type": "inspection_results", "results": [], "detections": [], "timestamp": 123.456}
{"type": "red_strips", "detections": [], "timestamp": 123.456}
{"type": "target_pose", "pose": null, "confidence": 0.0, "timestamp": 123.456}
```

`detections` 中的字段按任务不同包含 `zone`、`object_type`、`status`、`bbox`、`pose`、`center_3d`、`confidence`、`timestamp` 等。输出兼容 `RemotePerceptionGateway`，后续 Mission 可直接消费解析后的领域对象。

## 算力板取流

`camera_input.py` 提供统一帧对象：

```text
VisionFrame
├─ frame_id
├─ timestamp
├─ image
├─ width
├─ height
└─ source_type
```

支持三种输入：

```powershell
python .\vision_server.py --mode mock
python .\vision_server.py --mode video --source .\sample.mp4
python .\vision_server.py --mode camera --source 0
```

取流层支持：

- resize 到统一尺寸，默认 `640x480`
- `--flip-horizontal`
- `--roi x,y,w,h`
- `--save-debug-frames`
- `--debug-dir output/debug_frames`
- `--save-every 30`

## 固定检测闭环

当前 `FixedDetectionPipeline` 已提供可运行的固定检测闭环：

- `detect_obstacles()`：返回锥桶结构化结果。
- `detect_zone_letters()`：基于模板匹配识别 A/B/C/D 字母区域。
- `detect_gauges()`：基于图像识别仪表盘状态。
- `poll_inspection()`：融合字母和仪表结果，生成 Mission 可消费的巡检结果。
- `detect_red_strips()`：简单红色阈值检测红色长条，失败时回退固定结果。
- `estimate_target_pose()`：返回目标位姿估计。

这一步仍不是 YOLO/OCR 正式模型，目的是保证取流、检测、TCP、远端解析、测试闭环已经打通。

## 区域字母识别

`detect_zone_letters()` 已从固定模拟结果升级为基础模板匹配流程：

1. 灰度化。
2. 二值化或自适应阈值。
3. 轮廓/连通域检测。
4. 裁剪候选字母 ROI。
5. 与 A/B/C/D 模板匹配。
6. 输出 `zone`、`letter`、`bbox`、`confidence`、`timestamp`。

模板目录：

```text
assets/templates/letters/
├─ A.png
├─ B.png
├─ C.png
└─ D.png
```

如果模板不存在，检测层会自动生成基础模板图。模板当前是基础块状字形，用于打通闭环；实机调试时可以替换为 Arial 200 磅样张模板。

相关参数：

```powershell
--letter-min-confidence 0.55
--letter-template-dir assets/templates/letters
--letter-debug-save-roi
--letter-debug-dir output/debug_letters
```

带字母调试输出：

```powershell
python .\vision_server.py --mode camera --source 0 --letter-debug-save-roi --letter-debug-dir output/debug_letters
```

## 巡检结果融合

`poll_inspection()` 会在算力板端把 `detect_zone_letters()` 和 `detect_gauges()` 的结果融合为巡检结果：

```json
{
  "zone": "A",
  "gauge_status": "low",
  "status": "low",
  "abnormal": true,
  "confidence": 0.82,
  "letter_bbox": {"x1": 10, "y1": 10, "x2": 80, "y2": 120},
  "gauge_bbox": {"x1": 130, "y1": 40, "x2": 260, "y2": 170},
  "speak_key": "A_low",
  "timestamp": 123.456
}
```

融合规则：

- 低置信度字母或仪表不参与融合。
- 优先按字母 bbox 和仪表 bbox 中心点空间距离匹配。
- 空间距离超过阈值时，允许按稳定顺序回退匹配。
- `normal` -> `abnormal=false`。
- `low/high` -> `abnormal=true`。
- `speak_key` 与 `SpeakerGateway.play(key)` 兼容，例如 `A_low`、`B_normal`、`C_high`。

相关参数：

```powershell
--inspection-debug-save
--inspection-debug-dir output/debug_inspection
--inspection-max-match-distance 180
```

融合 debug 图会标注字母 bbox、仪表 bbox，以及 `zone + status + confidence`。

## 仪表盘识别

`detect_gauges()` 已从固定模拟结果升级为基础图像识别流程：

1. 灰度化。
2. 模糊降噪。
3. HoughCircles 或亮色区域定位表盘。
4. 裁剪表盘 ROI。
5. Canny + HoughLinesP 或 numpy fallback 检测指针。
6. 计算指针角度。
7. 按角度阈值输出 `low`、`normal`、`high`。

相关参数：

```powershell
--gauge-low-angle-range 180,250
--gauge-normal-angle-range 250,310
--gauge-high-angle-range 310,30
--gauge-min-confidence 0.55
--gauge-debug-save-roi
--gauge-debug-dir output/debug_gauge
```

当前本地环境无 `opencv-python` 时会走 numpy fallback；算力板安装 OpenCV 后会优先走 OpenCV 流程。

## 配置说明

编辑 `config/robot_config.json`：

- `robot`: 机器狗 IP 和 UDP 端口。
- `timing`: 心跳和主循环频率。
- `camera`: 机器狗本地相机配置，当前默认 mock。
- `arm`: 机械臂配置，当前默认 mock。
- `speaker`: 预录音频播放配置，禁用时使用 mock。
- `mission`: 任务超时、重试、掉落上限、置信度阈值。
- `perception.driver`: `"mock"`、`"local"`、`"remote"`。
- `remote_perception`: 外接算力板 TCP 地址和超时。
- `scenario_file`: mock 感知场景文件。

远程感知模式示例：

```json
{
  "perception": {
    "driver": "remote",
    "model_dir": "models/",
    "confidence_threshold": 0.6
  },
  "remote_perception": {
    "host": "192.168.1.200",
    "port": 9800,
    "timeout_sec": 2.0
  }
}
```

## 运行方式

机器狗本地 mock 联调：

```powershell
cd E:\DOG\Thedog\国赛
.\scripts\run.ps1
```

算力板端视觉服务：

```powershell
cd E:\DOG\Thedog\国赛
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode mock
```

摄像头输入：

```powershell
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode camera --source 0
```

带仪表调试输出：

```powershell
python .\vision_server.py --mode camera --source 0 --gauge-debug-save-roi --gauge-debug-dir output/debug_gauge
```

部署到机器狗上位机：

```powershell
cd E:\DOG\Thedog\国赛
.\scripts\deploy.ps1 -TargetUser ubuntu -TargetHost 192.168.1.103 -TargetDir /home/ubuntu/national_stage
```

## 测试

相机取流测试：

```powershell
python .\tools\test_camera_input.py
```

远程感知 TCP 闭环测试：

```powershell
python .\tools\test_remote_perception_client.py
```

完整 pytest：

```powershell
$env:PYTHONPATH='E:\DOG\Thedog\国赛\src'
python -m pytest -q
```

当前已覆盖：

- mock 连续取帧。
- video/camera 输入优雅处理。
- debug frame 保存。
- TCP 正常连接、JSON 接收、断连重连、超时、空结果。
- 固定检测结构化输出。
- A/B/C/D 字母模板匹配、模板自动生成、debug 图保存。
- 巡检融合结果、`speak_key` 生成、低置信过滤、debug 图保存。
- 仪表盘 LOW/NORMAL/HIGH 角度分类。
- `RemotePerceptionGateway` 对检测 JSON 的解析。

## 后续扩展

1. 在算力板安装 OpenCV 后，用真实 camera/video 输入调字母模板和仪表角度阈值。
2. 用真实 Arial 200 磅样张替换 `assets/templates/letters/` 中的基础模板。
3. 替换 `FixedDetectionPipeline` 中的固定/阈值逻辑为正式 YOLO、OCR、仪表读数模型。
4. 保持 `PerceptionGateway` 的纯检测边界，不把机器狗运动、机械臂动作或语音逻辑放入算力板视觉服务。
5. 机械臂和导航继续通过现有 `ArmGateway`、`NavigationGateway` 接入。
