# 锥形桶避障主线架构

本模块只负责国赛障碍区域内的两个锥形桶避障。它不接管巡检、机械臂、抓取、放置区识别，也不把锥形桶写进静态地图。

## 推荐方案

```text
摄像头画面
  -> YOLO 检测 cone bbox
  -> cone_strategy.py 根据 bbox 位置和面积输出 vx/vy/wz
  -> obstacle_zone_runner.py 通过 UDP 发给 controller/lite2_motion_receiver.py
  -> 狗端只执行速度指令，watchdog 超时自动停机
```

避障区域外不运行该模块。进入障碍区域后启用，通过障碍区域后停止并切换到巡检流程。

## 文件说明

```text
obstacle_avoidance/
├── cone_detector_yolo.py      # 加载 cone YOLO 模型，输出 ConeDetection
├── cone_strategy.py           # 纯规则避障层，输入 bbox，输出 vx/vy/wz
├── obstacle_zone_runner.py    # 摄像头 + YOLO + 规则 + UDP 运行入口
└── README.md                  # 本说明
```

## 数据集建议

后续拍摄锥形桶图片时建议使用单类别：

```text
0 cone
```

建议目录：

```text
data/cone_yolo/
├── raw_photos/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
├── labels/test
└── data.yaml
```

`data.yaml`：

```yaml
path: data/cone_yolo
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: cone
```

拍摄内容至少覆盖：

- 一个锥桶、两个锥桶。
- 锥桶在画面左侧、中间、右侧。
- 远距离、中距离、近距离。
- 机器狗运动时的模糊画面。
- 障碍区域真实光照、地面、背景。
- 锥桶局部遮挡或被画面边缘截断。

## 训练命令示例

```bash
cd /home/jetson/yolo_deploy
yolo detect train model=yolov8n.pt data=data/cone_yolo/data.yaml imgsz=640 epochs=60 batch=-1 name=cone_obstacle_yolo
```

训练完成后部署模型：

```bash
cp runs/detect/cone_obstacle_yolo/weights/best.pt /home/jetson/yolo_deploy/cone_best.pt
```

## 调试运行

先只看策略输出，不控制狗：

```bash
cd /home/jetson/yolo_deploy
python3 -m obstacle_avoidance.obstacle_zone_runner \
  --model /home/jetson/yolo_deploy/cone_best.pt \
  --camera /dev/video0 \
  --dry-run
```

确认方向正确后，再发给狗端 receiver：

```bash
python3 -m obstacle_avoidance.obstacle_zone_runner \
  --model /home/jetson/yolo_deploy/cone_best.pt \
  --camera /dev/video0 \
  --udp-host 127.0.0.1 \
  --udp-port 5005
```

狗端或 Jetson 另一终端需要先启动：

```bash
python3 controller/lite2_motion_receiver.py --listen-port 5005
```

## 规则层当前行为

- 没看到锥桶：低速直行。
- 锥桶挡住画面中间：向更空的一侧转向绕行。
- 锥桶偏左：向右轻微修正。
- 锥桶偏右：向左轻微修正。
- 锥桶面积过大：停止前进，只转向空侧。

现场需要重点调这些参数：

- `center_left_ratio`
- `center_right_ratio`
- `near_area_ratio`
- `stop_area_ratio`
- `forward_speed`
- `slow_speed`
- `avoid_turn_speed`

这些参数在 `cone_strategy.py` 的 `AvoidanceConfig` 中。

