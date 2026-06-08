# 国赛代码框架（绝影 Lite2 / UDP 协议）

这是一个面向 2026 四足大型组国赛任务的软件框架。当前代码分为两部分：

- **机器狗本地**：任务状态机、运动控制、机械臂控制、语音播放、远程感知客户端。
- **算力板端**：相机取流、固定检测、A/B/C/D 字母识别、仪表盘识别、TCP JSON 结果输出。

算力板只输出视觉结构化结果，不控制机器狗运动、机械臂或语音。

## 当前进度

已完成：

- 任务状态机骨架：避障、巡检、抓取、搬运、放置。
- `RemotePerceptionGateway`：机器狗本地 TCP 感知客户端。
- `vision_server.py`：算力板端 TCP JSON 服务。
- `CameraInput`：支持 mock、video、camera 三种输入。
- `FixedDetectionPipeline`：固定检测闭环。
- A/B/C/D 字母识别增强：模板自动生成、模板匹配、ROI/debug 图保存。
- 仪表盘识别增强：表盘定位、指针角度估计、`low/normal/high` 状态判断。
- 巡检结果融合：生成 `abnormal` 和 `speak_key`，例如 `A_low`、`B_normal`。
- 远程感知、取流、固定检测、字母识别、仪表识别的测试脚本。

未完成或仍需实机接入：

- 正式 YOLO/OCR 模型。
- RealSense 或现场相机实机调参。
- 真实机械臂驱动。
- 真实导航定位。
- 现场音频文件录制与播放验证。

## 目录结构

```text
国赛
├─ camera_input.py                 # 算力板取流层，输出 VisionFrame
├─ vision_server.py                # 算力板 TCP JSON 视觉服务
├─ assets
│  └─ templates
│     └─ letters                  # A/B/C/D 模板图
├─ config
│  ├─ robot_config.json
│  └─ scenario_mock.json
├─ docs
│  └─ architecture.md
├─ scripts
│  ├─ deploy.ps1
│  └─ run.ps1
├─ src
│  ├─ app                         # 配置 + DI 容器
│  ├─ core                        # 领域类型和异常
│  ├─ dog_sdk                     # 绝影 UDP 协议
│  ├─ hardware                    # 相机/机械臂/扬声器抽象
│  ├─ mission                     # 国赛任务状态机
│  ├─ navigation                  # 导航抽象
│  ├─ perception
│  │  ├─ gateway.py               # 纯感知接口
│  │  ├─ remote_gateway.py        # TCP 远程感知客户端
│  │  └─ detector
│  │     ├─ base.py
│  │     └─ fixed_detector.py     # 固定检测 + 字母识别 + 仪表盘识别
│  ├─ runtime
│  └─ utils
├─ tests
└─ tools
   ├─ test_camera_input.py
   └─ test_remote_perception_client.py
```

## 机器狗本地运行

默认使用 mock 配置：

```powershell
cd E:\DOG\Thedog\国赛
.\scripts\run.ps1
```

如需切换远程感知，在 `config\robot_config.json` 中设置：

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

## 算力板端视觉服务

mock 输入：

```powershell
cd E:\DOG\Thedog\国赛
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode mock
```

视频文件输入：

```powershell
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode video --source .\sample.mp4
```

真实摄像头输入：

```powershell
python .\vision_server.py --host 0.0.0.0 --port 9800 --mode camera --source 0
```

保存取流 debug 帧：

```powershell
python .\vision_server.py --mode camera --source 0 --save-debug-frames --debug-dir output/debug_frames --save-every 30
```

保存仪表盘 debug 图：

```powershell
python .\vision_server.py --mode camera --source 0 --gauge-debug-save-roi --gauge-debug-dir output/debug_gauge
```

保存区域字母 debug 图：

```powershell
python .\vision_server.py --mode camera --source 0 --letter-debug-save-roi --letter-debug-dir output/debug_letters
```

## 远程感知协议

TCP JSON Lines，每行一个请求或响应。

请求：

```json
{"req": "detect_obstacles"}
{"req": "detect_zone_letters"}
{"req": "detect_gauges"}
{"req": "poll_inspection"}
{"req": "detect_red_strips"}
{"req": "estimate_target_pose", "target": "strip"}
```

响应类型：

- `obstacles`
- `zone_letters`
- `gauges`
- `inspection_results`
- `red_strips`
- `target_pose`
- `error`

结果字段包含 `zone`、`object_type`、`status`、`bbox`、`pose`、`center_3d`、`confidence`、`timestamp` 等，兼容 `RemotePerceptionGateway`。

## 区域字母识别参数

`detect_zone_letters()` 当前支持基础模板匹配：

```powershell
--letter-min-confidence 0.55
--letter-template-dir assets/templates/letters
--letter-debug-save-roi
--letter-debug-dir output/debug_letters
```

模板目录：

```text
assets/templates/letters/
├─ A.png
├─ B.png
├─ C.png
└─ D.png
```

如果模板不存在，检测层会自动生成基础模板。真实上场前建议用现场规则中的 Arial、200 磅 A/B/C/D 样张替换这些模板，再结合 `output/debug_letters/` 调置信度阈值。

## 巡检融合输出

`poll_inspection` 会融合字母识别和仪表识别结果，输出可直接给 Mission 使用的巡检结果：

```json
{
  "zone": "A",
  "gauge_status": "low",
  "abnormal": true,
  "confidence": 0.82,
  "letter_bbox": {"x1": 10, "y1": 10, "x2": 80, "y2": 120},
  "gauge_bbox": {"x1": 130, "y1": 40, "x2": 260, "y2": 170},
  "speak_key": "A_low",
  "timestamp": 123.456
}
```

规则：

- `normal` 生成 `abnormal=false`
- `low/high` 生成 `abnormal=true`
- 低置信度字母或仪表不参与融合
- 优先按空间距离匹配，失败时按顺序匹配

调试参数：

```powershell
--inspection-debug-save
--inspection-debug-dir output/debug_inspection
--inspection-max-match-distance 180
```

## 仪表盘识别参数

`detect_gauges()` 当前支持基础图像识别和可调角度阈值：

```powershell
--gauge-low-angle-range 180,250
--gauge-normal-angle-range 250,310
--gauge-high-angle-range 310,30
--gauge-min-confidence 0.55
```

默认角度规则：

- `180-250`: 偏低 `low`
- `250-310`: 正常 `normal`
- `310-360` 或 `0-30`: 偏高 `high`

真实仪表盘上场前需要用 `output/debug_gauge/` 中的 ROI 和标注图继续调参。

## 测试

相机输入测试：

```powershell
python .\tools\test_camera_input.py
```

远程感知闭环测试：

```powershell
python .\tools\test_remote_perception_client.py
```

完整测试：

```powershell
$env:PYTHONPATH='E:\DOG\Thedog\国赛\src'
python -m pytest -q
```

当前验证过：

- mock 取流连续 10 帧。
- video 输入在有 OpenCV 时可读取帧。
- camera 无设备时优雅失败。
- debug frame 保存。
- TCP 正常连接、JSON 接收、断连重连、超时和空结果。
- 固定检测结果可被 `RemotePerceptionGateway` 解析。
- A/B/C/D 字母模板匹配、模板缺失自动生成、debug 图保存。
- 巡检融合、`speak_key` 生成、低置信过滤。
- 仪表盘 LOW/NORMAL/HIGH 角度分类。

## 部署到机器狗上位机

```powershell
cd E:\DOG\Thedog\国赛
.\scripts\deploy.ps1 -TargetUser ubuntu -TargetHost 192.168.1.103 -TargetDir /home/ubuntu/national_stage
```

部署前需要确认：

- `config\robot_config.json` 中机器狗 IP/端口正确。
- 如果使用远程感知，算力板 IP 和端口正确。
- 算力板已安装 OpenCV，真实 camera/video 输入可打开。
- 机械臂、导航和音频文件仍需按现场硬件继续接入。
