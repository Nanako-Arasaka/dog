# 国赛联调方案

本目录用于 2026 年中国高校智能机器人创意大赛（四足大型组）国赛任务联调。当前目标是把巡检识别、锥形桶避障、机械臂红条抓取、放置区识别、狗端移动控制通过轻量状态转发层连接起来。

现场部署建议采用“Jetson 算力板负责主要计算，狗本体只保留底层运动执行和安全兜底”的方式。YOLO、OpenCV 读表、放置区识别、任务状态转发、机械臂高层控制和 SLAM/路径决策放在 Jetson 上；狗端只接收速度指令并执行 watchdog 停机。

## 项目架构

```text
国赛/
├── live_detect_yolo_opencv.py       # 主线实时巡检：YOLO 定位 + OpenCV 读表
├── gauge_reader.py                  # 仪表盘 ROI 指针角度读取
├── integration_bridge/              # 状态转发层：格式统一、ROS2 topic 转发、日志
├── arm_grasp/                       # 机械臂红条抓取、保持夹紧、目标区放置
├── obstacle_avoidance/              # 锥形桶 YOLO 检测 + 规则避障 + UDP 速度输出
├── controller/                      # Lite2 移动接收、ORB-SLAM3、目标点控制
├── tools/                           # 数据集整理、标签检查、YOLO 推理和闭环 demo
├── docs/                            # 真实照片标注和训练流程文档
├── config/                          # 巡检识别配置
├── src/                             # 早期巡检服务、远端网关、播报接口
└── tests/                           # 单元测试
```

核心数据流：

```text
摄像头
  -> Jetson：巡检识别 / 锥桶避障 / 放置区字母识别
  -> integration_bridge：统一状态并转发 ROS2 topic
  -> arm_grasp：根据异常区域抓红条，识别到目标箱后松爪
  -> controller：计算或接收移动目标，向狗端发送速度指令
  -> 狗本体：执行 vx/vy/wz，超时自动停机
```

关键 ROS2 topic：

```text
/bridge/inspection_result       # 桥接层输入：单区或整组巡检结果
/inspection/all                 # 桥接层输出：A:abnormal,B:normal,...
/inspection/target_zones        # 机械臂记忆节点输出：异常区域列表，例如 A,C
/bridge/placement_zone          # 桥接层输入：放置区识别结果
/placement/recognized_zone      # 桥接层输出：当前看到的放置区，例如 A
/task/direct_grasp              # 触发机械臂抓取红条
/arm/command                    # 机械臂动作命令
/task/status                    # 任务状态
```

## 实现方法

国赛主流程按以下方式拆分：

1. **锥形桶避障**：Jetson 端读取摄像头，检测锥桶位置，生成简单绕行、减速或停止策略；狗端只接收 `vx/vy/wz` 并执行。
2. **巡检识别**：`live_detect_yolo_opencv.py` 使用 5 类 YOLO 定位 `zone_A/zone_B/zone_C/zone_D/gauge`，裁剪仪表盘 ROI 后调用 `gauge_reader.py` 读取指针角度并判断 `low/normal/high`。
3. **异常区域记忆**：巡检结果通过 `integration_bridge` 发布到 `/inspection/all`，`arm_grasp/inspection_memory_node.py` 记录异常区域并发布 `/inspection/target_zones`。
4. **红条抓取**：机械臂模块在异常区域存在时抓取红色长条，并保持夹爪闭合，不再按固定时间自动松手。
5. **放置区识别**：Jetson 端识别纸箱上的 `A/B/C/D`，通过 `integration_bridge` 发布 `/placement/recognized_zone`。
6. **目标区放置**：`arm_grasp/task_manager_node.py` 只有在当前识别到的放置区与本次异常目标一致时，才执行放置。

## 环境要求

### Jetson 算力板

- Jetson Xavier NX 或同级算力板。
- Ubuntu Linux。
- Python 3.8+。
- OpenCV 可用。
- YOLO 推理需要 `torch` 和 `ultralytics`。
- 机械臂、状态转发和 controller 联调需要 ROS2，推荐 Humble。

Python 基础依赖：

```bash
cd /home/jetson/yolo_deploy
python3 -m pip install -r requirements.txt
python3 -m pip install opencv-python ultralytics pyyaml
```

ROS2 环境：

```bash
source /opt/ros/humble/setup.bash
```

### 狗本体

- 运行 Lite2 底层控制环境。
- 运行轻量运动接收程序，例如 `controller/lite2_motion_receiver.py`。
- 保留 watchdog：如果一段时间收不到 Jetson 指令，自动停止。
- 不建议在狗端运行 YOLO、OpenCV 读表、ORB-SLAM3、数据集处理或复杂任务状态机。

## 启动指令

以下命令按现场多终端方式组织。正式运行前建议先使用 dry-run 验证通信链路。

### 终端 1：状态转发层

```bash
cd /home/jetson/yolo_deploy
source /opt/ros/humble/setup.bash
python3 integration_bridge/bridge_node.py
```

状态转发层默认会先冻结 A/B/C/D 四个稳定巡检结果，全部完成后才发布最终 `/inspection/all` 给机械臂。调试时如果需要逐帧转发：

```bash
python3 integration_bridge/bridge_node.py --no-freeze-inspection
```

查看当前流程状态：

```bash
ros2 topic echo /competition/state
```

重置巡检冻结结果：

```bash
ros2 topic pub --once /inspection/reset std_msgs/msg/Bool "data: true"
```

本地无 ROS2 格式验证：

```powershell
python .\integration_bridge\bridge_node.py --no-ros --inspection-json "A:abnormal,B:normal,C:unknown,D:normal"
python .\integration_bridge\bridge_node.py --no-ros --placement-zone "zone_A"
```

### 终端 2：巡检识别

模型文件放置：

```text
/home/jetson/yolo_deploy/best.pt
```

启动：

```bash
cd /home/jetson/yolo_deploy
python3 live_detect_yolo_opencv.py
```

### 障碍区域：锥形桶避障

该模块只在障碍区域内启用。YOLO 只负责检测 `cone` 框，`obstacle_avoidance/cone_strategy.py` 根据框的位置和面积输出 `vx/vy/wz`，再通过 UDP 发给 `controller/lite2_motion_receiver.py`。

干跑调试：

```bash
cd /home/jetson/yolo_deploy
python3 -m obstacle_avoidance.obstacle_zone_runner \
  --model /home/jetson/yolo_deploy/cone_best.pt \
  --camera /dev/video0 \
  --dry-run
```

正式接入狗端：

```bash
python3 -m obstacle_avoidance.obstacle_zone_runner \
  --model /home/jetson/yolo_deploy/cone_best.pt \
  --camera /dev/video0 \
  --udp-host 127.0.0.1 \
  --udp-port 5005
```

详细拍摄、标注和参数说明见 `obstacle_avoidance/README.md`。

`live_detect_yolo_opencv.py` 会在 ROS2 可用时自动发布：

```text
/bridge/inspection_result
/bridge/placement_zone
```

如果只想保留窗口显示和终端输出，不向桥接层发布：

```bash
INSPECTION_BRIDGE_DISABLE=1 python3 live_detect_yolo_opencv.py
```

如果只做离线图片验证：

```powershell
python .\tools\inspection_pipeline_demo.py --source .\data\inspection_yolo\images\test --model .\runs\detect\inspection_yolo_gauge_location\weights\best.pt --output-json .\output\inspection_batch.json --debug-dir .\output\debug_inspection_pipeline
```

### 终端 3：机械臂抓取

```bash
cd /home/jetson/arm_grasp
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch arm_grasp jetarm_grasp.launch.py
```

如果需要按完整比赛阶段执行，也就是“巡检阶段只记录异常，机器人到长条抓取区后才开始抓红条”，启动机械臂任务管理时关闭自动开抓：

```bash
ros2 launch arm_grasp jetarm_grasp.launch.py auto_start_on_targets:=false
```

然后启动初步协调程序：

```bash
cd /home/jetson/yolo_deploy
source /opt/ros/humble/setup.bash
python3 tools/inspection_pick_place_coordinator.py
```

当导航或人工调试确认机器人已到长条抓取区后，发布：

```bash
ros2 topic pub --once /mission/event std_msgs/msg/String "data: 'pick_area_arrived'"
```

协调程序会读取已经冻结的 `/inspection/all`，记录 A/B/C/D 正常和异常状态，把异常区域队列发布到 `/inspection/target_zones`，并触发 `/task/start`。后续红条抓取、等待放置区字母、匹配目标纸箱和放置动作仍由 `arm_grasp/task_manager_node.py` 完成。

手动发布巡检结果测试：

```bash
ros2 topic pub --once /bridge/inspection_result std_msgs/msg/String \
  "data: 'A:abnormal,B:normal,C:abnormal,D:normal'"
```

手动触发抓红条测试：

```bash
ros2 topic pub --once /task/direct_grasp std_msgs/msg/String "data: 'red'"
```

手动模拟到达 A 放置区：

```bash
ros2 topic pub --once /bridge/placement_zone std_msgs/msg/String "data: 'A'"
```

### 终端 4：狗端运动接收

先 dry-run：

```bash
cd /home/jetson/controller
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

正式控狗：

```bash
cd /home/jetson/controller
python3 lite2_motion_receiver.py \
  --listen-port 5005 \
  --robot-ip 192.168.1.120 \
  --robot-port 43893 \
  --default-speed 9000 \
  --turn-speed 20000
```

### 终端 5：SLAM 和目标点控制

ORB-SLAM3 和目标点控制的详细流程见 `controller/README.md`。典型启动顺序：

```bash
source /opt/ros/humble/setup.bash
cd /home/jetson/colcon_ws
source install/setup.bash
ros2 run lite2_navigation_bridge goal_controller --ros-args \
  -p pose_topic:=/camera_pose \
  -p pose_type:=pose_stamped \
  -p target_x:=0.3 \
  -p target_y:=0.0 \
  -p target_yaw:=0.0 \
  -p receiver_ip:=127.0.0.1 \
  -p receiver_port:=5005
```

## 巡检识别模块

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
