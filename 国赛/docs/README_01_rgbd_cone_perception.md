# Intel RealSense D435i RGB-D 相机接入 + 锥桶识别 + 3D 定位

本文档交给 RGB-D 感知负责人。第一阶段目标是把 Intel RealSense D435i 的 RGB、Depth、CameraInfo 跑通，并把锥桶检测结果转换成相对机器狗的局部 3D 位置。该模块只做感知输出，不控制机器狗。

## 我负责什么

- 确认 Intel RealSense D435i 在 ROS2 中实际发布的话题。
- 读取 RGB 图像、Depth 图像、CameraInfo/内参。
- 编写 `rgbd_probe_node.py` 或等价调试脚本，先证明 RGB-D 数据可用。
- 使用 YOLO 检测锥桶，类别建议统一为 `cone`。
- 使用 bbox 内的 depth 估计锥桶位置，输出 `x` 和 `z`。
- 对连续帧检测结果做简单滤波和短时保留。
- 做可视化调试图：bbox、depth ROI、x/z/conf、有效深度比例。
- 把 `ConeObstacle` 以 Python 数据结构、ROS topic 或 JSON 形式提供给避障控制负责人。

## 我不负责什么

- 不负责避障策略。
- 不负责机器狗速度控制。
- 不负责 UDP 发包。
- 不负责最终比赛状态机。
- 不负责建图、SLAM 或全局路径规划。
- 不把巡检仪表盘识别、长条抓取识别混进第一版锥桶感知。

## 项目背景

本项目面向 2026 年中国高校智能机器人创意大赛足式机器人挑战赛四足大型组。国赛线下挑战任务包括避障、巡检、长条抓取。当前第一阶段只做避障任务：障碍区域随机放置 2 个锥形桶，规格约 67 x 32 x 32 cm，四足机器人需要自主避开锥桶并通过障碍区域。

比赛不允许使用激光雷达，所以避障模块必须以相机视觉为核心。本项目当前已有普通 USB 图像到 ORB-SLAM3 mono 的运行思路，但这不等于 Intel RealSense D435i 的 depth 已经接入成功。锥桶避障第一版不要求建图，不要求完整 SLAM，不要求全局路径规划，只需要局部反应式避障：看到前方锥桶，估计锥桶 3D 位置，把障碍列表交给控制模块。

## 当前相机参数

- 型号：Intel RealSense D435i
- 深度技术：Active Stereoscopic，主动红外双目/主动立体视觉
- 官方工作距离：约 0.3m - 3m
- Depth 最高分辨率/帧率：1280×720 @ 30FPS
- 深度视场角：H:87° / V:58°
- RGB Sensor：有
- 6DoF IMU：有，但第一版锥桶避障不强依赖 IMU
- 接口：USB 3
- 软件栈：librealsense2 + realsense-ros
- 工作环境：室内

Jetson 上当前 V4L2 设备映射诊断结果：

```text
Sonix USB 2.0 Camera:
  /dev/video0: 普通外接 USB 摄像头，MJPG/YUYV，不是 RealSense depth
  /dev/video1: metadata，不是图像

Intel RealSense D435i:
  /dev/video2: Z16 深度流，16-bit
  /dev/video3: metadata，不是图像
  /dev/video4: RealSense 深度侧 IR/UYVY 流，可预览但不推荐作为 YOLO 主 RGB
  /dev/video5: metadata，不是图像
  /dev/video6: RealSense 专用 RGB 彩色流，YUYV，推荐作为 YOLO/预览 RGB
  /dev/video7: metadata，不是图像
```

如果使用 OpenCV 直接读取 RealSense 彩色图做 YOLO 调试，默认相机应使用 `/dev/video6` 或 index `6`，FOURCC 优先使用 `YUYV`。RealSense 裸深度流是 `/dev/video2`。不要把普通外接 USB 摄像头 `/dev/video0` 当成 RealSense depth，也不要把普通 USB RGB 和 RealSense depth 混合做定位。

重要约束：

- D435i 官方最小工作距离约 0.3m，但避障控制仍要考虑机器狗惯性、步态摆动和感知延迟。
- RGB 和 depth 默认不一定对齐，必须确认是否启用了 `aligned_depth_to_color`。
- 第一版推荐使用 aligned depth。如果没有对齐，不能直接拿 RGB bbox 去原始 depth 图同坐标取距离。
- 如果没有 aligned depth，需要使用驱动提供的对齐话题、把 RGB resize 到 depth 尺寸，或根据标定关系做坐标映射。
- 当前仓库只看到 `cone_avoidance` 资料，未看到 D435i 接入代码。需要实机确认 RealSense topic、depth encoding、depth 单位和 `depth_scale`。

已知部署包修复点：

```text
live_detect_yolo_opencv.py:
  如果目标是 RealSense RGB，CAMERA_ID 应使用 6
  CAMERA_PATH 应使用 /dev/video6
  FOURCC 从 MJPG 改为 YUYV

obstacle_avoidance/obstacle_zone_runner.py:
  如果目标是 RealSense RGB，--camera 默认值应使用 /dev/video6

vision_server.py / camera_input.py:
  如果目标是 RealSense RGB，OpenCV RGB 输入使用 --source 6
```

这些改动只解决 RGB 图像输入问题。后续 bbox + depth 定位仍建议使用 `pyrealsense2` 或 `realsense-ros` 的 aligned depth，不建议用 `/dev/video6` 和 `/dev/video2` 两个裸 V4L2 设备强行按同坐标融合。

## 与控制负责人的接口

感知模块输出 `ConeObstacle` 列表。第一版建议只输出稳定检测到的锥桶，并按 `z` 从近到远排序；如果控制同学更需要左右顺序，可以再约定改成按 `x` 从左到右排序。

```yaml
ConeObstacle:
  x: float                 # 左右偏移，单位 m，默认 x > 0 为左侧
  z: float                 # 前方距离，单位 m
  conf: float              # 融合 YOLO 置信度和 depth 有效性的置信度
  bbox: [x1, y1, x2, y2]   # RGB 图像上的检测框
  age: int                 # 连续稳定帧数
  last_seen: time          # 最近一次观测时间
```

坐标约定：

- `z` 表示前方距离，单位 m。
- `x` 表示左右偏移，单位 m。
- 默认 `x > 0` 为左侧，`x < 0` 为右侧。
- 如果实际相机坐标相反，必须在本模块里统一修正，并在调试记录中写清楚。

交付给控制模块的最低要求：

- 没有稳定检测时返回空列表，而不是返回假距离。
- depth 无效或有效像素比例太低时，不输出该锥桶或降低 `conf`。
- 双锥桶同时出现时，输出两个 `ConeObstacle`。
- 感知模块不发送 `vx/vy/wz`。

## 需要阅读的已有代码和资料

当前 checkout 中可读到：

- `cone_avoidance/README.md`：已有锥形桶避障初步方案，包含安全半径、通道判断和里程碑。
- `cone_avoidance/DATASET_AND_TRAINING.md`：锥桶数据采集、YOLO 标注和训练说明。
- `cone_avoidance/config/cone_dataset.yaml`：YOLO 数据集配置。
- `cone_avoidance/scripts/train_cone_yolo.py`：训练脚本。
- `cone_avoidance/scripts/split_yolo_dataset.py`：数据集划分脚本。

题述还要求阅读这些文件，但当前 `/Users/silencecf/Documents/DOG` checkout 未出现，需要从主工程补齐或在实机目录确认：

- `Lite2正式运行流程.txt`
- `lite2_motion_receiver.py`
- `colcon_ws/src/lite2_navigation_bridge/lite2_navigation_bridge/goal_controller.py`
- `colcon_ws/src/lite2_navigation_bridge/launch/goal_controller.launch.py`
- `colcon_ws/src/lite2_navigation_bridge/setup.py`

如果后续拿到这些文件，本负责人主要关注相机 topic 是否与现有 `/image_raw`、ORB-SLAM3 mono 流程冲突；锥桶感知第一版不依赖 `/camera_pose`。

## RealSense ROS2 话题检查

先记录实际 topic 名称，不要猜：

- `/camera/camera/color/image_raw`
- `/camera/camera/color/camera_info`
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/depth/camera_info`
- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/aligned_depth_to_color/camera_info`
- `/camera/camera/imu`
- `/camera/camera/depth/color/points`

第一版推荐使用：

```text
RGB:   /camera/camera/color/image_raw
Depth: /camera/camera/aligned_depth_to_color/image_raw
Info:  /camera/camera/color/camera_info
```

IMU 和点云可以记录，但第一版锥桶避障不强依赖。RealSense topic 名可能因 namespace、launch 参数或 realsense-ros 版本不同而变化，必须用 `ros2 topic list` 实测。

推荐启动命令：

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true pointcloud.enable:=false
```

如果需要 IMU 调试，可以使用：

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true enable_accel:=true enable_gyro:=true
```

具体参数名以本机安装的 realsense-ros 版本为准，需实机确认。

调试命令：

```bash
ros2 topic list | grep -Ei "camera|image|depth|point|cloud|rgb|color|ir"
ros2 topic info <rgb_topic>
ros2 topic info <depth_topic>
ros2 topic info <camera_info_topic>
ros2 topic hz <rgb_topic>
ros2 topic hz <depth_topic>
ros2 topic echo <camera_info_topic> --once
ros2 topic info /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

## `rgbd_probe_node.py` 调试脚本要求

请先做 probe，再做 YOLO 和定位。这个脚本只用于确认数据，不需要控制机器狗。

当前仓库已提供一个 `pyrealsense2` 版本的 aligned depth 探针脚本：

```bash
cd /Users/silencecf/Documents/DOG/cone_avoidance
python3 scripts/realsense_aligned_depth_probe.py --width 640 --height 480 --fps 30
```

在 Jetson 上运行时，把锥桶依次放到画面中心约 0.5m、1.0m、1.5m 位置，观察终端输出的 `center_roi.median_m` 是否接近真实距离。脚本会保存 color、aligned depth 可视化和 overlay 到：

```text
cone_avoidance/debug/realsense_probe/
```

这个 probe 跑通后，再把 YOLO bbox 接到 aligned depth ROI。不要跳过这一步直接融合 `/dev/video6` 和 `/dev/video2`。

如果 Jetson 缺少 `pyrealsense2`，可以先走 ROS2 话题版本。先启动 RealSense ROS2 驱动：

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true pointcloud.enable:=false
```

然后运行：

```bash
cd /Users/silencecf/Documents/DOG/cone_avoidance
python3 scripts/ros2_aligned_depth_probe.py
```

如果实际 topic 名不同，先用 `ros2 topic list | grep -Ei "camera|depth|color|image"` 查看，再传参：

```bash
python3 scripts/ros2_aligned_depth_probe.py \
  --rgb-topic /camera/camera/color/image_raw \
  --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
  --info-topic /camera/camera/color/camera_info
```

如果 Jetson 没有显示器，使用浏览器串流版：

```bash
python3 scripts/realsense_aligned_depth_web.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model scripts/cone_yolo_best.pt \
  --conf 0.45
```

在同一局域网电脑浏览器打开：

```text
http://<jetson-ip>:8080/
```

页面会显示 RGB overlay、YOLO 锥桶 bbox、bbox 内 depth ROI、aligned depth 可视化、`center_roi.median_m` 和 `obstacles`。`obstacles` 中 `x > 0` 表示锥桶在机器狗左侧，`z` 表示前方距离，单位 m。如果不知道 Jetson IP，可以在 Jetson 终端执行：

```bash
hostname -I
```

必须具备：

- 可配置 `rgb_topic`、`depth_topic`、`camera_info_topic`，并记录是否使用 `aligned_depth_to_color`。
- 订阅 RGB、Depth、CameraInfo。
- 使用 `cv_bridge` 转 OpenCV。
- 打印 RGB 分辨率、Depth 分辨率、Depth encoding。
- 打印 color、raw depth、aligned depth 的 shape。
- 检查 depth 是否是原始 depth raw，不要误用 colorized depth。
- 检查 depth 单位和 RealSense `depth_scale`。
- 打印画面中心 ROI 的 depth 最小值、最大值、中位数。
- 如果 RGB 和 Depth 尺寸不一致，打印明确警告。
- 如果 depth 超出 D435i 约 0.3m - 3m 的主要工作范围，按无效或低可信处理。
- 保存 color、raw depth 可视化、aligned depth 可视化三类调试图。

需要特别确认 depth 单位：

- `16UC1` 常见单位是 mm，但必须实测确认。
- `32FC1` 常见单位是 m，但必须实测确认。
- RealSense raw depth 通常还涉及 `depth_scale`，不能只看 encoding 猜单位。
- 把锥桶分别放在 0.5m、1.0m、1.5m，检查中位数是否接近真实距离。

## YOLO 识别方案

主方案：YOLO 检测 `cone` 或 `pvc_cone`。项目内已有训练资料建议使用单类别 `cone`，第一版推荐沿用 `cone`，避免类别名混乱。

兜底方案：颜色/形状检测可以作为调试辅助，但不要替代 YOLO 主流程。第一版只识别锥桶，不要把巡检仪表盘识别混进来。

YOLO 输出接口：

```yaml
ConeBBox:
  bbox: [x1, y1, x2, y2]
  conf: float
  class_id: int
  class_name: "cone"
```

数据集必须覆盖：

- 单锥桶。
- 双锥桶。
- 0.5m、1.0m、1.5m、2.0m 距离。
- 左右边缘。
- 部分遮挡。
- 强光和暗光。
- 运动模糊。
- 从机器狗出发区域视角拍摄。
- 两个锥桶同时出现。

## bbox + depth 定位方法

不要用 bbox 高度单目估距作为主方案。Intel RealSense D435i 可以输出深度图，第一版应使用 RGB 图像、aligned depth 图像、CameraInfo/内参完成 RGB-D 局部定位。

建议流程：

1. 对 YOLO bbox 取中下部 ROI，避开锥桶尖端和大量背景。
2. 将 ROI 映射到 depth 图坐标。第一版优先使用 `/camera/camera/aligned_depth_to_color/image_raw`，确认 aligned depth 与 color 图像坐标一致。
3. 过滤 `0`、`NaN`、过近、过远深度。
4. D435i 官方工作距离约 0.3m - 3m，超出主要工作范围的值需要谨慎处理。
5. 使用 median depth，不使用平均值。
6. 统计 ROI 有效像素比例。比例太低时，不输出该锥桶或降低置信度。
7. 使用 CameraInfo 反投影得到局部坐标。

3D 坐标计算：

```text
u = bbox 中心 x
v = bbox 底部中心或中下部 ROI 中心
z = ROI depth 中位数，单位 m
x = (u - cx) * z / fx
```

使用 CameraInfo 中的 `fx`、`fy`、`cx`、`cy`。第一版主要输出 `x` 和 `z`，`y` 可以先不交给控制模块。

## 滤波要求

- 连续 3 帧以上稳定检测才输出给避障模块。
- 丢失 3-5 帧以内可以短暂保留历史，但必须增加 `age` 或降低 `conf`，不能伪装成新检测。
- 抖动过大的检测降低置信度。
- 双锥桶时第一版按 `z` 从近到远排序，并在接口中记录这个约定。
- 不要把 depth 空洞、0 或 NaN 当成真实距离。

## 可视化调试

调试画面至少显示：

- RGB 画面上的 bbox。
- `x/z/conf` 标注。
- bbox 内实际取 depth 的 ROI。
- 当前检测到几个锥桶。
- depth 有效像素比例。
- 保存 debug 图像或短视频片段。

建议每次实机调试保存一组样例：

```text
debug/color_YYYYMMDD_HHMMSS.png
debug/raw_depth_vis_YYYYMMDD_HHMMSS.png
debug/aligned_depth_vis_YYYYMMDD_HHMMSS.png
debug/cone_overlay_YYYYMMDD_HHMMSS.png
```

## 推荐文件结构

只推荐结构，不要求本次一次性实现全部文件：

```text
cone_avoidance/
  rgbd_camera.py
  rgbd_probe_node.py
  cone_detector_yolo.py
  cone_depth_localizer.py
  cone_tracker.py
  perception_debug_viewer.py
  config/
    perception.yaml
```

## 推荐开发步骤

1. 用 topic 命令确认 RGB、Depth、CameraInfo 实际名称和频率。
2. 写 `rgbd_probe_node.py`，保存 color、raw depth、aligned depth 可视化图。
3. 确认 depth encoding、单位和 `depth_scale`，用 0.5m、1.0m、1.5m 实测校验。
4. 确认 RGB 和 Depth 是否对齐。没有 `aligned_depth_to_color` 时先不要接 YOLO 输出。
5. 跑现有 YOLO 训练流程，得到能识别 `cone` 的模型。
6. 把 YOLO bbox 接到 depth ROI，输出单帧 `x/z/conf`。
7. 加连续帧滤波和短时丢失保留。
8. 做双锥桶输出和排序。
9. 与控制负责人对接 `ConeObstacle` 列表，不接机器狗也能独立调试。

## 验收标准

- 能稳定看到 RGB 图。
- 能稳定读到 depth 图。
- depth 单位确认清楚，是 mm 还是 m。
- 0.5m、1.0m、1.5m 放置锥桶时，`z` 误差可接受。
- 锥桶在左边时 `x` 符号正确。
- 锥桶在右边时 `x` 符号正确。
- 双锥桶能同时输出两个 `ConeObstacle`。
- depth 空洞时不会输出离谱距离。
- 能把 `ConeObstacle` 以 Python 数据结构、ROS topic 或 JSON 形式提供给控制模块。
- 不控制机器狗也能独立调试。

## 常见坑

- 必须插 USB 3，不能按普通低速 USB 连接来判断性能。
- Jetson 上 librealsense2 / realsense-ros 版本要确认。
- RealSense topic 名可能因 namespace 不同而变化，必须 `ros2 topic list` 实测。
- RGB 和 depth 默认不一定对齐，优先使用 `aligned_depth_to_color`。
- 不要用 colorized depth 计算距离，要使用 raw depth 或 aligned raw depth。
- depth 图可能是 `16UC1`，单位可能需要结合 `depth_scale` 确认。
- depth 图也可能被转换成 `32FC1`，必须实测单位。
- bbox 包含背景，平均 depth 会被背景污染。
- 锥桶边缘深度可能不稳定。
- 强光、反光、黑色、透明物体可能造成 depth 空洞。
- D435i 有 IMU，但第一版避障不使用 IMU 作为主输入。
- 不要把无效深度 `0`、`NaN` 当成真实距离。
- 当前已有普通 `/image_raw` 跑通，不代表 depth 已经接入成功。

## 最终集成接口约定

第一阶段感知模块必须可单独调试。最终集成时只交付障碍列表，不直接动机器狗：

```yaml
perception_output:
  source: "rgbd_cone_perception"
  timestamp: time
  frame_id: string
  obstacles:
    - x: 0.25
      z: 1.10
      conf: 0.86
      bbox: [320, 180, 430, 410]
      age: 5
      last_seen: time
```

控制负责人只依赖 `obstacles`，不依赖 YOLO 内部细节。感知负责人需要在集成记录里写清楚 topic 名称、depth 单位、`depth_scale`、RGB/Depth 对齐方式、`x` 正方向和排序方式。

## 当前实机管道命令

在主工程 `国赛` 目录下，当前版本的感知脚本可以直接通过 JSONL 管道接入避障控制：

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

`--control-jsonl` 会让 stdout 只输出控制侧需要的 JSON 行，浏览器预览仍然可用：

```text
http://<jetson-ip>:8080/
```

每行控制 JSON 至少包含：

```json
{"obstacles":[{"x":0.25,"z":0.95,"conf":0.84,"bbox":[3,0,333,282],"age":0,"last_seen":1782978792.13}],"front_depth":0.99,"depth_valid_ratio":1.0,"aligned_depth_ok":true,"realsense_ok":true,"realsense_fps":29.8}
```

字段含义：

- `obstacles` 只包含 depth 有效的锥桶，按 `z` 从近到远排序。
- `front_depth` 来自画面中心安全 ROI 的最近有效深度，单位 m。
- `depth_valid_ratio` 是中心安全 ROI 的有效深度比例。
- `aligned_depth_ok` 为 `true` 表示当前使用的是 `rs.align(rs.stream.color)` 后的 depth。
- `realsense_ok` 和 `realsense_fps` 供控制模块做掉线/低帧率安全停止。
