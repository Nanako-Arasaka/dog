# Controller 模块概览

本目录是国赛底盘移动和建图相关代码，主要负责：

```text
USB 摄像头 / ORB-SLAM3
  -> 发布机器人位姿
  -> goal_controller 计算目标点速度
  -> lite2_motion_receiver.py 转成绝影 Lite2 UDP 运动指令
  -> 机器狗执行移动
```

它不是完整国赛任务状态机，目前不直接负责巡检识别、机械臂抓取、红条放置或语音播报。

## 目录结构

```text
controller/
├── ORB_SLAM3/                         # ORB-SLAM3 源码和词袋文件
├── colcon_ws/src/lite2_navigation_bridge/
│   ├── lite2_navigation_bridge/
│   │   └── goal_controller.py         # ROS2 目标点控制节点
│   ├── launch/
│   │   └── goal_controller.launch.py  # goal_controller 启动文件
│   └── README.md                      # 子包说明
├── lite2_motion_receiver.py           # UDP JSON -> Lite2 运动 UDP 指令
└── Lite2正式运行流程.txt              # 当前实测运行流程记录
```

## 已实现功能

### 1. Lite2 运动指令接收器

文件：

```text
controller/lite2_motion_receiver.py
```

功能：

- 监听上游 UDP JSON，默认 `0.0.0.0:5005`。
- 将速度或动作命令转换为绝影 Lite2 的 UDP 控制包。
- 持续发送心跳。
- 超时未收到上游命令时自动停狗。
- 支持 `--dry-run`，只打印命令，不真的发给机器狗。

支持的输入示例：

```json
{"action":"forward","speed":12000,"duration":0.5}
{"cmd":"turn_left","speed":8000}
{"vx":0.2,"vy":0.0,"wz":-0.3}
{"linear":{"x":0.2,"y":0.0},"angular":{"z":-0.3}}
```

### 2. ROS2 目标点控制

文件：

```text
controller/colcon_ws/src/lite2_navigation_bridge/lite2_navigation_bridge/goal_controller.py
```

功能：

- 订阅 ORB-SLAM3 或里程计位姿。
- 根据当前位姿和目标点计算 `vx/wz`。
- 通过 UDP JSON 发给 `lite2_motion_receiver.py`。
- 到达目标点后发送停止命令。

默认输入/输出：

```text
订阅位姿: /orbslam3/pose 或 /camera_pose
发布控制: UDP JSON -> 127.0.0.1:5005
```

当前控制逻辑比较简单：

- 距离目标较远：前进并修正航向。
- 航向误差过大：先原地转向。
- 到达目标距离阈值内：对齐目标 yaw。
- 位姿超时或没有位姿：停止。

### 3. ORB-SLAM3

目录：

```text
controller/ORB_SLAM3/
```

用途：

- 提供视觉 SLAM 定位能力。
- 输出机器人/相机位姿，供 `goal_controller.py` 使用。

## 当前不包含的功能

当前代码还没有完整实现以下内容：

- 锥形桶避障策略。
- 国赛完整任务状态机。
- 巡检识别。
- 机械臂抓取/放置。
- 根据异常区域自动选择放置箱。
- 目标物/放置区视觉识别到地图坐标的转换。

这些功能应通过外部模块和 ROS2/UDP 话题接入，不建议直接塞进 `goal_controller.py`。

## 典型运行流程

### 终端 1：启动控狗接收器

先 dry-run：

```bash
cd ~/Desktop
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

正式控狗：

```bash
cd ~/Desktop
python3 lite2_motion_receiver.py \
  --listen-port 5005 \
  --robot-ip 192.168.1.120 \
  --robot-port 43893 \
  --default-speed 9000 \
  --turn-speed 20000
```

### 终端 2：启动 USB 摄像头

```bash
cd ~/usb_camera
source install/setup.bash

ros2 run usb_cam usb_cam_node_exe --ros-args \
  --params-file /home/jetson/usb_camera/src/usb_cam-ros2/config/params_1.yaml
```

### 终端 3：启动 ORB-SLAM3

```bash
source /opt/ros/humble/setup.bash
cd ~/colcon_ws
source install/setup.bash

ros2 run orbslam3 mono \
  /home/jetson/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  /home/jetson/ORB_SLAM3/Examples/Monocular/TUM1.yaml
```

检查位姿：

```bash
ros2 topic info /camera_pose
ros2 topic echo /camera_pose
```

### 终端 4：启动目标点控制

建议先用相对目标测试：

```bash
source /opt/ros/humble/setup.bash
cd ~/colcon_ws
source install/setup.bash

ros2 run lite2_navigation_bridge goal_controller --ros-args \
  -p pose_topic:=/camera_pose \
  -p pose_type:=pose_stamped \
  -p target_is_relative:=true \
  -p target_x:=0.3 \
  -p target_y:=0.0 \
  -p target_yaw:=0.0 \
  -p max_vx:=0.10 \
  -p max_wz:=0.15 \
  -p goal_tolerance:=0.12 \
  -p run_timeout:=8.0 \
  -p receiver_ip:=127.0.0.1 \
  -p receiver_port:=5005
```

注意：当前 `goal_controller.py` 源码里没有实现 `target_is_relative` 和 `run_timeout` 参数。如果使用这些参数，需要先确认 Jetson 上运行的是不是更新版代码。当前仓库版本主要支持绝对目标点和 `/lite2/goal` 动态目标。

## 和国赛其他模块的衔接建议

建议保持模块边界：

```text
controller       只负责底盘移动、SLAM 位姿、目标点跟踪
inspection       只负责 A/B/C/D 和仪表盘状态识别
arm_grasp        只负责红条抓取和放置动作
obstacle_avoid   只负责锥形桶检测和避障策略
task_manager     只负责比赛阶段切换
```

后续国赛主流程可以设计为：

```text
START
  -> OBSTACLE_AVOID
  -> INSPECTION
  -> GRASP_RED_BAR
  -> WAIT_PLACE_ZONE_RECOGNITION
  -> PLACE_RED_BAR
  -> DONE
```

其中本目录适合承担：

```text
OBSTACLE_AVOID 中的底盘速度执行
GO_TO_INSPECTION_AREA
GO_TO_GRASP_AREA
GO_TO_PLACE_AREA
```

不建议让本目录直接调用机械臂或巡检识别代码。

## 常见问题

### 一直 waiting_for_pose

检查 ORB-SLAM3 是否真的发布位姿：

```bash
ros2 topic info /camera_pose
ros2 topic echo /camera_pose
```

### 狗乱跑

优先确认当前使用的是绝对目标还是相对目标。地图绝对坐标未标定时容易导致目标方向错误。

### 狗不动

先 dry-run 看 `lite2_motion_receiver.py` 是否收到 UDP：

```bash
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

如果 dry-run 有输出但狗不动，检查：

- `--robot-ip`
- `--robot-port`
- 狗端是否允许 UDP 控制
- Lite2 与 Jetson 是否在同一网段

### 前后/左右/转向反了

`lite2_motion_receiver.py` 提供了方向反转参数：

```bash
--invert-forward
--invert-lateral
--invert-turn
```

不要一次性全改，先看 receiver 打印出来的 `vx/vy/wz` 和最终 `MotionCommand`。
