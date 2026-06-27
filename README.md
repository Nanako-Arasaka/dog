# 四足机器狗 · 中国高校智能机器人创意大赛

2026 年中国高校智能机器人创意大赛（四足大型组）参赛项目，分两阶段：预选赛（仪表盘识别）和国赛（三阶段任务）。

---
## 当前成就
已经通过预选赛（省赛部分），已经保底拿到国家级奖项。

## 项目概览

| 赛段 | 目录 | 任务 | 状态 |
|------|------|------|:----:|
| 预选赛 | `yuxuansai_new/` | 仪表盘指示灯识别（High/Normal/Low） | 完成 |
| 国赛 | `国赛/` | 避障 → 巡检识别 → 红条抓取与投放 | 联调中 |

---

## 一、预选赛 — 机器狗仪表盘识别系统

预选赛部分已经完成，用于 2026 年中国高校智能机器人创意大赛（四足大型组）预选赛阶段的仪表盘状态识别。该版本为最终实际成果版本，整体采用本地摄像头采集、本地模型推理和本地 OpenCV 可视化的方式运行，不再使用狗端与 Jetson 之间的 UDP 视频帧转发架构。

### 系统架构

```text
摄像头
  → OpenCV 读取画面
  → 霍夫圆检测定位仪表盘区域
  → 裁剪仪表盘 ROI
  → ResNet18 三分类模型推理
  → 输出 down / normal / over
  → OpenCV 窗口实时显示 FPS、检测圆框和识别结果
```

系统支持两种推理后端：

1. **PyTorch 推理**：用于常规调试、模型验证和开发阶段运行。
2. **TensorRT 推理**：用于 Jetson 上的最终部署加速，输入尺寸为 `224 × 224`。

### 识别类别

| 模型输出 | 中文含义 | 说明 |
|:--:|:--:|------|
| `down` | 偏低 | 仪表盘读数偏低 |
| `normal` | 正常 | 仪表盘读数处于正常范围 |
| `over` | 偏高 | 仪表盘读数偏高 |

标签顺序固定为：

```text
down → normal → over
```

训练集目录和推理端类别顺序需要保持一致，建议训练数据文件夹命名为：

```text
down/
normal/
over/
```

### 目录结构

```text
yuxuansai/
├── start_dog.py              # 本地推理统一入口，支持 PyTorch / TensorRT
├── start_jetson.py           # 兼容入口，实际调用 start_dog.py
├── Dashboard_detec2t.py      # PyTorch 推理脚本，ResNet18 三分类
├── detect_dashboard_trt.py   # TensorRT 推理脚本，最终部署版本
├── dashboard_model.py        # ResNet18 / ResNet34 模型定义
├── dashboard_train.py        # 仪表盘分类模型训练脚本
├── trans_onnx.py             # PyTorch 权重导出 ONNX
├── trans_trt.py              # ONNX 转 TensorRT 引擎
├── round_detect.py           # 仪表盘圆形检测相关实验脚本
├── Opencv_Dashboard_detect.py # OpenCV 仪表盘检测实验脚本
├── checkpoints/              # 模型权重与 TensorRT 引擎目录
└── resnet18_dashboard.trt    # TensorRT 引擎文件示例
```

### 快速启动

#### PyTorch 模式

默认使用 PyTorch 后端进行本地推理：

```bash
python3 start_dog.py
```

也可以显式指定模型权重：

```bash
python3 start_dog.py \
  --mode torch \
  --model-path ./checkpoints/model_best.pth
```

#### TensorRT 模式

在 Jetson 或支持 TensorRT 的环境中，可以使用 TensorRT 引擎运行：

```bash
python3 start_dog.py \
  --mode trt \
  --engine-path ./checkpoints/resnet18_dashboard.trt
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `torch` | 推理后端，可选 `torch` 或 `trt` |
| `--model-path` | `./checkpoints/model_best.pth` | PyTorch 权重路径 |
| `--engine-path` | `./checkpoints/resnet18_dashboard.trt` | TensorRT 引擎路径 |
| `--camera-device` | PyTorch 为空，TensorRT 为 `/dev/video2` | 摄像头设备路径 |
| `--camera-index` | PyTorch 为 `3`，TensorRT 为 `2` | 摄像头索引备用参数 |
| `--width` | `640` | 摄像头画面宽度 |
| `--height` | `480` | 摄像头画面高度 |
| `--no-infer-flip` | 关闭 | PyTorch 模式下不对推理输入做上下翻转 |

### 推理流程说明

1. 使用 OpenCV 从摄像头读取实时画面。
2. 对画面进行灰度化和高斯模糊处理。
3. 使用霍夫圆检测寻找仪表盘圆形区域。
4. 选择半径最大的圆作为仪表盘主体。
5. 裁剪仪表盘 ROI，并放大后送入模型。
6. 将 ROI resize 到 `224 × 224`。
7. 使用 ResNet18 进行三分类推理。
8. 使用滑动结果缓冲区稳定输出，减少单帧误判。
9. 在 OpenCV 窗口中显示 FPS、识别结果和圆形定位框。

### 注意事项

- 当前最终成果不是“狗端采集画面、Jetson 远程推理、UDP 回传结果”的网络架构，而是本地直接推理架构。
- `start_jetson.py` 只是为了兼容旧文件名保留，实际入口仍然复用 `start_dog.py`。
- TensorRT 推理输入尺寸固定为 `224 × 224`，训练、ONNX 导出和 TensorRT 转换时需要保持一致。
- 类别顺序必须固定为 `down, normal, over`，否则会导致中文结果含义错位。
- 摄像头索引在不同设备上可能不同，如果无法打开摄像头，需要调整 `--camera-device` 或 `--camera-index`。
- PyTorch 模式下默认会对推理输入做上下翻转，如果现场画面方向已经正确，可以使用 `--no-infer-flip` 关闭。

## 二、国赛 — Jetson 主计算联调方案

国赛部分按“Jetson 算力板负责主要计算，狗本体只保留底层运动执行和安全兜底”的方式组织。当前 `国赛/` 目录已经包含巡检识别、YOLO + OpenCV 仪表盘读表、机械臂抓取模块、狗端运动/建图控制模块，以及用于模块间解耦的状态转发层。

### 模块分工

1. **巡检识别** — `国赛/live_detect_yolo_opencv.py` 和 `国赛/gauge_reader.py` 负责识别区域字母、仪表盘位置和仪表状态。
2. **状态转发层** — `国赛/integration_bridge/` 只负责格式统一、ROS2 topic 转发和事件日志，例如 `/bridge/inspection_result -> /inspection/all`、`/bridge/placement_zone -> /placement/recognized_zone`。
3. **机械臂抓取** — `国赛/arm_grasp/` 负责红色长条抓取、保持夹紧，并在识别到目标放置区后执行放置。
4. **狗端运动与建图** — `国赛/controller/` 负责 Lite2 运动指令接收、ORB-SLAM3 相关代码和目标点控制。
5. **避障功能** — `国赛/obstacle_avoidance/` 使用 YOLO 检测锥形桶，再用规则层根据 bbox 位置和面积输出 `vx/vy/wz`，只向狗端下发轻量速度指令。

### 国赛主流程

1. **避障通过** — Jetson 识别锥形桶并规划简单绕行，狗端只执行速度指令和 watchdog 停机。
2. **巡检识别** — Jetson 识别 `A/B/C/D` 区域与仪表盘状态，记录异常区域。
3. **抓取红条** — 机械臂抓取红色长条并持续保持夹紧。
4. **目标放置** — Jetson 在放置区识别纸箱字母，状态转发层发布目标区域，机械臂确认匹配后松爪。

### 项目架构

```text
摄像头
  -> Jetson 视觉识别：巡检 / 锥桶 / 放置区字母
  -> integration_bridge：状态格式统一与 ROS2 topic 转发
  -> arm_grasp：红条抓取、保持夹紧、匹配目标区后放置
  -> controller：SLAM/目标点控制/速度指令
  -> 狗本体：底层运动执行与 watchdog 安全停机
```

### 启动环境

- Jetson Xavier NX，Ubuntu Linux，Python 3.8+。
- Python 依赖：`numpy`、`opencv-python`、`torch`、`ultralytics`、`pytest`。
- ROS2 环境：机械臂和状态转发联调需要 `rclpy`、`std_msgs`，推荐 Humble。
- 狗端仅运行轻量运动接收程序和安全停机逻辑，不建议运行 YOLO、OpenCV 读表或 SLAM。

### 常用启动指令

```bash
cd /home/jetson/yolo_deploy
python3 integration_bridge/bridge_node.py
python3 live_detect_yolo_opencv.py
```

```bash
cd /home/jetson/arm_grasp
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch arm_grasp jetarm_grasp.launch.py
```

```bash
cd /home/jetson/controller
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

锥形桶避障模型训练完成后建议部署为 `/home/jetson/yolo_deploy/cone_best.pt`。先用 dry-run 验证检测和速度策略：

```bash
cd /home/jetson/yolo_deploy
python3 -m obstacle_avoidance.obstacle_zone_runner \
  --model /home/jetson/yolo_deploy/cone_best.pt \
  --camera /dev/video0 \
  --dry-run
```

详见 [国赛/README.md](国赛/README.md)

---

## 仓库结构

```
.
├── README.md                  # 本文件
├── yuxuansai_new/             # 预选赛：仪表盘识别系统
├── 国赛/                       # 国赛：巡检、避障/控制、机械臂、状态转发
└── .gitignore
```
