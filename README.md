# 四足机器狗 · 中国高校智能机器人创意大赛

2026 年中国高校智能机器人创意大赛（四足大型组）参赛项目，分两阶段：预选赛（仪表盘识别）和国赛（三阶段任务）。

---

## 项目概览

| 赛段 | 目录 | 任务 | 状态 |
|------|------|------|:----:|
| 预选赛 | `yuxuansai_new/` | 仪表盘指示灯识别（High/Normal/Low） | 完成 |
| 国赛 | `国赛/` | 避障 → 巡检识别 → 红条抓取与投放 | 联调中 |

---

## 一、预选赛 — 仪表盘识别系统

### 架构

```
摄像头(狗端) → UDP JPEG帧 → Jetson推理(TensorRT) → UDP JSON结果 → 狗端可视化
```

- **狗端** (`start_dog.py`)：采集摄像头画面 → UDP 发送 JPEG 帧 → 接收推理结果 → OpenCV 渲染
- **Jetson 端** (`start_jetson.py`)：接收视频帧 → HoughCircleCropper 裁剪仪表盘 → TensorRT 推理 → 回传 JSON
- 支持独立本地模式：直接连摄像头推理，无需网络（见 `start_dog.py` 内 `standalone` 模式）

### 目录结构

```
yuxuansai_new/
├── start_dog.py              # 狗端入口：采集 + 推流 + 结果可视化
├── start_jetson.py           # Jetson 端入口：收帧 → 推理 → 回传
├── start_dog.sh / start_jetson.sh   # Linux 一键启动脚本
├── perception/               # 感知核心
│   ├── cropper.py            # HoughCircleCropper 霍夫圆检测 + 裁剪
│   ├── detector.py           # DashboardCameraDetector 推理 + 分类
│   ├── model.py              # ResNet18 / ResNet34 分类模型
│   ├── inference.py          # TensorRT 推理封装
│   └── visualize.py          # PIL 中文渲染 / SwitchConfirm 防抖
├── scripts/
│   ├── build_trt.py          # 构建 TensorRT 引擎
│   ├── export_onnx.py        # 导出 ONNX 模型
│   └── video_extractor.py    # 视频抽帧工具
├── checkpoints/              # 模型权重 / TensorRT 引擎（需自行放置）
├── data/                     # 训练/测试数据（需自行放置）
└── requirements.txt
```

### 快速启动

**Jetson 端：**

```bash
python3 start_jetson.py \
  --target-ip <狗端IP> --target-port 5005 \
  --engine-path ./checkpoints/model_fp16_160.engine \
  --input-size 160
```

**狗端：**

```bash
python3 start_dog.py \
  --jetson-ip <Jetson IP> --jetson-frame-port 6006 \
  --listen-port 5005
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--confidence-threshold` | 0.5 | 单帧置信度阈值 |
| `--hough-interval` | 3 | 圆检测间隔帧数 |
| `--cls-confirm-window` | 2 | 分类确认窗口（连续 N 帧一致才切换） |
| `--input-size` | 160 | 推理输入尺寸（需与训练/ONNX 一致） |
| `--send-hz` | 15 | UDP 发送频率 |

### 结果显示

| 类别 | 颜色 | 含义 |
|:----:|:----:|------|
| High | 红色 | 仪表读数偏高 |
| Normal | 绿色 | 正常 |
| Low | 黄色 | 仪表读数偏低 |
| Unknown | 灰色 | 未检测到 / 置信度不足 |

### 注意事项

- **训练 `--input-size` / 导出 ONNX / Jetson 启动 `--input-size` 三者必须一致**
- **训练 `class_order` 与推理 `--class-names` 必须一致（high, normal, low）**
- 先 FP32 保精度，再切 FP16 做性能优化
- 调参顺序：先 `--hough-canny-high` 确保圆检测正确 → 再调 `--confidence-threshold` 控制误分类 → 最后调 `--stabilizer-window` 控制输出抖动

---

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
