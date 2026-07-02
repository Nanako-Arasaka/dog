# 锥形桶避障模块初步方案

## 任务背景

根据《2026年中国高校智能机器人创意大赛（四足大型组）》6.4 线下挑战任务，避障区域会随机放置 2 个锥形桶，四足机器人需要全自主避开锥形桶并通过障碍区域。道具清单中四足大型组序号 1 为 PVC 圆锥，规格为 67 x 32 x 32 cm，数量 2。

评分相关约束：

- 成功自主穿越障碍区域：10 分。
- 碰到障碍物或比赛道具：每次扣 5 分，最多扣 30 分。
- 出现失控连续冲撞：比赛立即结束。
- 比赛不允许使用激光雷达。

本模块采用与先前仪表盘识别类似的 YOLO 视觉识别方案，并结合三目结构光 3D 深度相机进行空间定位和避障决策。

## 总体思路

YOLO 只负责回答“锥桶在哪里”，三目结构光深度相机负责回答“锥桶离机器人多远、在机器人左边还是右边、是否挡住安全通道”。这样比单目视觉估距更稳定，也能减少因为锥桶尺寸、光照、拍摄角度变化带来的距离误差。

推荐数据流：

```text
RGB 图像
  -> YOLO 锥桶检测
  -> 输出 bbox

深度图 / 点云
  -> 与 RGB 对齐
  -> 在 bbox 内提取有效深度
  -> 计算锥桶 3D 坐标和占据区域

锥桶 3D 坐标
  -> 安全距离膨胀
  -> 判断左/中/右通道
  -> 输出速度和转向指令
```

## 模块划分

### 1. YOLO 锥桶识别

复用仪表盘识别的 YOLO 流程，新增一个目标类别：

```text
cone
```

训练数据建议包含：

- 单个锥桶、两个锥桶同时出现。
- 远距离、中距离、近距离。
- 锥桶位于画面左侧、右侧、中央、边缘。
- 出发区视角、机器人行进中视角。
- 弱光、强光、反光、阴影。
- 运动模糊、部分遮挡、背景杂物。

推理输出：

```text
class_id
confidence
bbox: x1, y1, x2, y2
```

建议初期参数：

- `conf_thres`: 0.45-0.60
- `iou_thres`: 0.45
- 输入尺寸：640 x 640 起步，根据 Jetson 实测 FPS 再调整。

### 2. 深度定位

三目结构光 3D 深度相机应输出 RGB 图像、深度图，最好还能输出点云。需要先确认 RGB 与深度是否已经硬件或 SDK 对齐。

对每个 YOLO 检测框进行深度处理：

1. 取 bbox 下半部分或中下部区域，避开锥桶尖端和背景。
2. 过滤无效深度、过远深度、突变噪声。
3. 使用中位数深度，而不是平均值，减少离群点影响。
4. 将像素坐标和深度反投影到相机坐标系。
5. 根据相机外参转换到机器人坐标系。

建议输出结构：

```text
ConeDetection3D:
  id
  confidence
  bbox
  x_robot_m      # 左右方向，左正或右正按项目坐标系统一
  y_robot_m      # 垂直方向，可选
  z_robot_m      # 前方距离
  depth_valid
  timestamp
```

如果 SDK 能直接给点云，可以在 bbox 内选取点云点，投影到地面平面后计算锥桶中心和最近点。避障更关心最近点，而不是视觉框中心。

### 3. 安全占据区

PVC 圆锥底部约 32 x 32 cm，高 67 cm。避障时不能只按 32 cm 计算，需要把机器人宽度、步态摆动、定位误差都算进去。

建议初期安全半径：

```text
cone_radius = 0.16 m
robot_half_width = 0.20-0.25 m
walking_margin = 0.15-0.25 m
depth_error_margin = 0.05-0.10 m

safe_radius = 0.55-0.75 m
```

调试早期建议偏保守，宁愿绕得大一点，也不要碰到道具。

### 4. 通道判断

将相机前方区域划分为左、中、右三个通道：

```text
前方 ROI: z_robot_m in [0.3, 2.5]
左通道:   x_robot_m < -lane_width / 3
中通道:   abs(x_robot_m) <= lane_width / 3
右通道:   x_robot_m > lane_width / 3
```

避障策略初版：

- 未检测到锥桶：低速直行。
- 检测到锥桶但不在前方危险区：低速直行，并持续观察。
- 单个锥桶在左前方：向右绕行。
- 单个锥桶在右前方：向左绕行。
- 单个锥桶在正前方：选择更空的一侧绕行。
- 两个锥桶形成中间通道：如果中间宽度足够，走中间。
- 两个锥桶间距不足：选择外侧空间更大的一边绕行。
- 最近障碍距离小于急停阈值：停止，重新判断。

建议阈值：

```text
slow_down_distance = 1.2 m
avoid_distance = 0.9 m
stop_distance = 0.45-0.55 m
```

### 5. 运动控制

避障模块不直接关心底层步态，只输出高层速度指令：

```text
linear_x
linear_y
angular_z
state
```

初期建议只使用前进速度和偏航角速度：

```text
linear_x = 0.10-0.25 m/s
angular_z = -0.4-0.4 rad/s
```

如果机器人控制接口支持横移，可以在绕锥桶时加入小幅 `linear_y`，这样路线会更平滑。但第一版建议先用“前进 + 转向”，降低调试复杂度。

### 6. 状态机

建议将锥桶避障做成一个独立状态机，穿越障碍区域后再交接给巡检任务。

```text
INIT
  -> SEARCH_CONE
  -> APPROACH_OBSTACLE_AREA
  -> AVOID_CONE
  -> RECENTER
  -> PASS_OBSTACLE_AREA
  -> DONE
```

状态说明：

- `INIT`: 加载 YOLO 模型，初始化相机，检查深度流。
- `SEARCH_CONE`: 低速观察，等待稳定检测。
- `APPROACH_OBSTACLE_AREA`: 朝障碍区域前进。
- `AVOID_CONE`: 根据 3D 锥桶位置绕行。
- `RECENTER`: 绕过锥桶后回到通道中心方向。
- `PASS_OBSTACLE_AREA`: 确认前方无危险障碍，继续穿越。
- `DONE`: 输出完成标志，交给巡检识别模块。

### 7. 深度相机异常兜底

结构光深度在强光、黑色/反光表面、距离过近时可能出现空洞或噪声。需要设置兜底逻辑。

推荐策略：

- YOLO 检测到锥桶，但深度无效：减速，不立即高速通过。
- 连续多帧深度无效：使用 bbox 高度做粗略距离估计，仅用于保守避障。
- 深度突变过大：丢弃当前帧，使用上一帧稳定结果。
- 检测框置信度低且深度异常：不参与路径决策。
- 前方近距离点云密集但 YOLO 未检出：触发保守减速或停止。

## 推荐实现文件结构

```text
cone_avoidance/
  README.md
  config/
    cone_avoidance.yaml
  models/
    cone_yolo_best.pt
  scripts/
    train_cone_yolo.py
    run_cone_detection.py
    run_cone_avoidance.py
  src/
    cone_detector.py
    depth_projector.py
    obstacle_tracker.py
    avoidance_planner.py
    state_machine.py
  datasets/
    images/
    labels/
  tools/
    collect_images.py
    visualize_depth_bbox.py
```

## 开发里程碑

### M1: 数据采集与 YOLO 初训

- 采集 300-800 张锥桶图片。
- 完成 `cone` 类别标注。
- 训练第一版 YOLO 模型。
- 离线验证检测效果，重点看漏检和误检。

拍照、标注目录、数据集划分和训练命令见：

```text
cone_avoidance/DATASET_AND_TRAINING.md
```

### M2: 深度融合

- 打通三目结构光相机 RGB + depth 数据流。
- 完成 RGB bbox 到深度图的映射。
- 输出锥桶在机器人坐标系下的 3D 坐标。
- 可视化 bbox、深度值、3D 位置。

### M3: 避障决策

- 实现安全半径膨胀。
- 实现单锥桶和双锥桶通道判断。
- 输出模拟速度指令，不直接控制机器人。
- 用录制数据回放测试决策是否稳定。

### M4: 低速实机闭环

- 接入机器人速度控制接口。
- 单锥桶静态测试。
- 双锥桶随机摆放测试。
- 调整 `stop_distance`、`avoid_distance`、`safe_radius`。

### M5: 赛道流程集成

- 避障完成后输出 `DONE`。
- 与巡检识别状态机衔接。
- 测试完整流程：出发区 -> 障碍区 -> 检测区。

## 第一版验收标准

- 锥桶检测稳定，近中远距离都能识别。
- 深度定位在 0.5-2.5 m 范围内稳定输出。
- 两个随机锥桶摆放时，机器人能低速绕开并通过障碍区。
- 机器人不会主动贴近锥桶，最近距离建议大于 0.45 m。
- 深度异常或识别不稳定时，机器人优先减速或停止。

## 当前实机联调命令

在主工程 `国赛` 目录下，感知脚本可以直接通过 JSONL 管道接入控制模块：

```bash
python3 cone_avoidance/scripts/realsense_aligned_depth_web.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model cone_avoidance/scripts/cone_yolo_best.pt \
  --conf 0.45 \
  --roi 80 \
  --control-jsonl \
  --control-rate-hz 10 | \
python3 -m cone_avoidance.main_avoidance_run \
  --receiver-ip 127.0.0.1 \
  --receiver-port 5005
```

`--control-jsonl` 会让 stdout 只输出控制模块需要的字段：

- `obstacles`: depth 有效的锥桶列表
- `front_depth`: 画面中心安全 ROI 最近有效深度
- `depth_valid_ratio`: 安全 ROI 有效深度比例
- `aligned_depth_ok`: aligned depth 是否可用
- `realsense_ok`: RealSense 是否正常
- `realsense_fps`: RealSense 实测帧率

浏览器预览仍然可用：

```text
http://<jetson-ip>:8080/
```

## 后续优化方向

- 将 YOLO 模型导出 TensorRT，提高 Jetson Xavier NX 推理速度。
- 用多帧跟踪减少检测抖动。
- 使用点云地面分割，得到更可靠的锥桶占据区域。
- 根据场地图尺寸加入虚拟边界，避免机器人绕行过大。
- 记录每次测试的检测结果、深度结果、速度指令，方便技术报告写实测数据。
