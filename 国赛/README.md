# 国赛 · 最终集成版

2026 年中国高校智能机器人创意大赛（四足大型组）国赛任务代码。本目录实现完整的比赛闭环：**锥形桶避障 → 仪表盘巡检识别（语音播报）→ 红色异常长条抓取 → 放置区字母识别与精确放置**，并集成了 ORB-SLAM3 视觉导航、motion_mux 运动仲裁与一键启动流程。

部署采用「Jetson 算力板负责主要计算，狗本体只保留底层运动执行和安全兜底」的分布式架构：YOLO、OpenCV 读表、放置区识别、任务状态机、机械臂高层控制与 SLAM/路径决策全部放在 Jetson（Xavier NX）；狗端只接收速度指令并执行 watchdog 停机。

## 项目架构

```text
国赛/
├── launch/guosai_final.launch.py     # 一键启动编排（ROS2 launch）
├── nodes/                            # 国赛业务节点
│   ├── voice_broadcast_node.py       #   语音播报（12 路中文音频）
│   ├── waypoint_navigator.py         #   航点顺序导航
│   ├── motion_mux.py                 #   运动指令仲裁（导航/避障/急停）
│   ├── cone_avoidance_node.py        #   锥桶避障节点
│   └── localization_watchdog.py      #   定位丢失看门狗
├── scripts/                          # 现场脚本
│   ├── guosai_onekey.sh              #   一键全流程（采集航点/预检/运行）
│   ├── run_guosai_final.sh           #   正式运行入口
│   ├── preflight_guosai_final.py     #   赛前预检
│   ├── waypoint_capture_tool.py      #   航点采集工具
│   └── gen_voice_audio.sh / check_onboard_audio.sh / repair_guosai_final_config.py
├── config/guosai_final.yaml          # 最终运行配置（SLAM/相机/导航/避障/巡检/机械臂/语音/FSM）
├── jetson_payload/                   # Jetson 部署包（FINAL SLAM 地图 + 上传脚本）
├── live_detect_yolo_opencv.py        # 主线实时巡检：YOLO 定位 + OpenCV 读表
├── gauge_reader.py                   # 仪表盘 ROI 指针角度读取（独立模块，多层降级）
├── camera_input.py                   # 多输入源统一取流封装（mock/video/camera）
├── vision_server.py                  # TCP 视觉服务（远端推理）
├── integration_bridge/               # 状态转发层：格式统一、ROS2 topic 转发、巡检冻结
├── arm_grasp/                        # JetArm 六自由度机械臂 ROS2 包
│   ├── arm_control_node.py           #   IK 求解 + 舵机底层控制
│   ├── vision_node.py                #   HSV 红条检测 + 3D 位姿估计
│   ├── task_manager_node.py          #   抓取验证 + 重试状态机
│   └── inspection_memory_node.py     #   异常区域记忆
├── cone_avoidance/                   # 锥桶 YOLO 检测 + 策略（当前主线）
├── obstacle_avoidance/               # 锥桶规则避障（早期模块，cone_strategy 四级策略）
├── controller/                       # Lite2 移动接收、ORB-SLAM3、goal_controller
├── output/audio/                     # 12 路中文语音播报（A/B/C/D × 偏低/正常/偏高）
├── docs/                             # 技术文档 / HANDOFF / 现场清单 / 视频脚本
├── tools/ tests/ runs/ models/       # 数据集工具、单元测试、训练产物、模型
└── requirements.txt
```

## 比赛主流程（FSM）

```text
start_exit → obstacle_entry → (锥桶避障) → obstacle_exit
  → inspection_box_1_side_1/2 → inspection_box_2_side_1/2  (巡检识别 + 语音播报)
  → pick_area  (红条抓取)
  → place_A/B/C/D  (放置到异常区域)
  → finish
```

状态机配置见 `config/guosai_final.yaml` 的 `fsm:` 段（航点由 `waypoints_FINAL.yaml` 定义，现场采集）。

## 核心数据流

```text
摄像头 (RealSense D435i)
  → Jetson：巡检识别 / 锥桶避障 / 放置区字母识别 / HSV 红条检测
  → integration_bridge：统一状态并转发 ROS2 topic（巡检冻结，3 次一致才发布）
  → waypoint_navigator：按 FSM 航点导航
  → motion_mux：仲裁 导航/避障/急停 速度指令
  → voice_broadcast_node：按巡检结果播报中文语音
  → arm_grasp：根据异常区域抓红条，匹配目标纸箱后松爪
  → controller：ORB-SLAM3 定位 + lite2_motion_receiver 下发 UDP
  → 狗本体：执行 vx/vy/wz，超时自动停机
```

## 关键 ROS2 话题

```text
/bridge/inspection_result       # 桥接层输入：单区或整组巡检结果
/inspection/all                 # 桥接层输出：A:abnormal,B:normal,...
/inspection/all_detailed        # 详细状态：A:low,B:normal,...（语音播报主用）
/inspection/target_zones        # 机械臂记忆节点输出：异常区域列表，例如 A,C
/bridge/placement_zone          # 桥接层输入：放置区识别结果
/placement/recognized_zone      # 桥接层输出：当前看到的放置区，例如 A
/task/direct_grasp              # 触发机械臂抓取红条
/arm/command · /arm/feedback    # 机械臂动作命令与反馈
/competition/state              # 全局任务状态
/motion/nav_cmd · /motion/avoid_cmd · /motion/stop  # 运动指令（motion_mux 仲裁）
```

## 模块说明

### 1. 巡检识别（巡检识别 40 分）

- `live_detect_yolo_opencv.py`：5 类 YOLO 定位 `zone_A/B/C/D/gauge`，裁剪仪表盘 ROI。
- `gauge_reader.py`：经典 CV 管线读指针角度并判断 `low/normal/high`：
  灰度化 → 高斯模糊 → CLAHE 增强 → Canny → 霍夫圆检测 → 霍夫直线 → HSV 色带分类。
  多层降级：霍夫圆失败→ROI 中心；直线失败→暗色尖端；色带失败→角度阈值。
- 巡检结果经 `integration_bridge` 冻结（每区 3 次一致）后发布 `/inspection/all`。

### 2. 语音播报（语音满分 vs 仅终端减半）

`nodes/voice_broadcast_node.py` 订阅巡检结果，按 `A_低/正常/高` 等 12 种组合播放 `output/audio/` 下预生成 wav，支持 mock / aplay / ffplay 引擎（Jetson 现场用 aplay）。

### 3. 锥形桶避障（10 分）

YOLO 检测锥桶（`cone_avoidance/scripts/cone_yolo_best.pt`，conf 0.35）→ `cone_strategy.py` 四级规则策略：
紧急停车（面积>20%）→ 主动避障（中央区域或面积>8%）→ 微调偏航 → 全速前进（vx=0.16）。
转向方向由左右锥桶加权面积比较决定，8 Hz UDP 下发狗端。

### 4. 红条抓取（50 分）

`arm_grasp/` 基于 JetArm 六自由度机械臂：
- HSV 双阈值红条检测 + RealSense 深度 3D 位姿估计（手眼标定）。
- 几何法两连杆 IK（两遍求解处理腕部 18 cm 连杆）。
- 视觉反馈闭环：抓取前后 Δz>3 cm 判定成功；空抓按像素偏移×0.35 定向重试（≤10 次）；视觉丢失回退底座（≤3 次）。
- 仅当放置区识别字母与异常目标一致时才放置，防止误放。

### 5. SLAM 导航与运动仲裁

- `controller/ORB_SLAM3/`：视觉 SLAM（RGB-D），发布 `/camera_pose`。
- `nodes/waypoint_navigator.py`：按 FSM 顺序到达航点。
- `nodes/motion_mux.py`：仲裁导航速度/避障速度/急停，`obstacle_priority: true`。
- `controller/lite2_motion_receiver.py`：UDP 5005 接收 JSON 指令，转为 Lite2 私有协议下发狗体（默认 192.168.1.120:43893），0.8 s 超时停机。

## 环境要求

### Jetson 算力板

- Jetson Xavier NX（算力不得高于此），Ubuntu，Python 3.8+，OpenCV。
- YOLO：PyTorch + Ultralytics。
- ROS2 Humble（机械臂、状态转发、导航联调）。

```bash
cd 国赛
python3 -m pip install -r requirements.txt
source /opt/ros/humble/setup.bash
```

### 狗本体

- 仅运行轻量运动接收程序（`controller/lite2_motion_receiver.py`）与 watchdog。
- 不建议在狗端运行 YOLO / OpenCV 读表 / ORB-SLAM3 / 任务状态机。

### 大文件（Git LFS）

`*.osa`（SLAM 地图）由 Git LFS 管理，clone 后：

```bash
git lfs pull
```

## 启动方式

### 一键启动（现场推荐）

```bash
bash scripts/guosai_onekey.sh          # 采集航点 → 预检 → 正式运行
bash scripts/run_guosai_final.sh       # 直接运行正式流程
```

### ROS2 统一启动

```bash
source /opt/ros/humble/setup.bash
export GUOSAI_ROOT=$(pwd)
ros2 launch launch/guosai_final.launch.py
```

### 分终端启动（调试）

```bash
# 终端 1：状态转发层
python3 integration_bridge/bridge_node.py

# 终端 2：巡检识别
python3 live_detect_yolo_opencv.py --model best.pt --camera-id 4 --no-gui

# 终端 3：机械臂
cd arm_grasp && colcon build && source install/setup.bash
ros2 launch arm_grasp jetarm_grasp.launch.py

# 终端 4：狗端运动接收（先 dry-run）
python3 controller/lite2_motion_receiver.py --listen-port 5005 --dry-run
```

### 关键调试指令

```bash
ros2 topic echo /competition/state                          # 查看任务状态
ros2 topic pub --once /inspection/reset std_msgs/msg/Bool "data: true"   # 重置巡检冻结
ros2 topic pub --once /bridge/inspection_result std_msgs/msg/String \
  "data: 'A:abnormal,B:normal,C:abnormal,D:normal'"         # 手动注入巡检结果
ros2 topic pub --once /task/direct_grasp std_msgs/msg/String "data: 'red'"  # 触发抓红条
```

## 测试

```bash
python -m pytest -q                                     # 单元测试
python tools/test_camera_input.py                       # 多输入源
python tools/test_remote_perception_client.py           # 远端视觉
python tools/test_speaker_playback.py --save-playback-log   # 语音播报
```

## 相关文档

| 文档 | 位置 |
|---|---|
| 国赛技术文档草稿（评审用） | `docs/技术文档_草稿.md` |
| Jetson 现场执行清单 | `docs/Jetson_现场执行清单.md` |
| 接手指南（HANDOFF） | `docs/接手指南_HANDOFF.md` |
| 线上视频拍摄脚本 | `docs/线上视频_拍摄脚本.md` |
| 机械臂跑通流程 | `arm_grasp/JetArm_跑通流程.md` |
| 运动控制运行流程 | `controller/Lite2正式运行流程.txt` |
| Jetson 部署命令 | `jetson_payload/JETSON_RUN_COMMANDS.md` |
