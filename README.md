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

## 一、预选赛 — 视觉识别与 ROS 综合任务

预选赛部分对应 2026 年中国高校智能机器人创意大赛足式机器人挑战赛专项赛预选赛任务。项目以绝影 Lite2 四足机器人为载体，完成了视觉识别、ROS 通信、基础操作、建图与导航等全部考核内容。

预选赛不只是单一的仪表盘识别任务，而是由四个部分组成：

| 模块        | 任务内容                      | 技术路线                                    |  状态 |
| --------- | ------------------------- | --------------------------------------- | :-: |
| 表针识别      | 仪表盘偏低 / 正常 / 偏高三态识别       | 霍夫圆检测 + ResNet18 分类 + TensorRT 加速       |  完成 |
| 颜色识别      | RGB 三色、五类形状物块检测与统计        | HSV 分割 + 形态学处理 + 分水岭分割 + 几何特征分类         |  完成 |
| ROS 程序题   | 邮政编码查询服务                  | ROS1 Service + 自定义 srv + 客户端 / 服务端节点    |  完成 |
| ROS 基础操作  | turtlesim 轨迹录制与复现、工作空间覆盖  | rosbag + ROS_PACKAGE_PATH 配置            |  完成 |
| ROS 建图与导航 | Gazebo 建图、地图保存、小车“马”字轨迹导航 | Gazebo + gmapping + map_server + 多目标点导航 |  完成 |

---

### 1. 表针识别

表针识别任务用于识别工业仪表盘的当前状态，输出结果分为三类：

| 类别 | 含义         |
| -- | ---------- |
| 偏低 | 仪表读数低于正常范围 |
| 正常 | 仪表读数处于正常范围 |
| 偏高 | 仪表读数高于正常范围 |

#### 技术路线

```text
摄像头采集图像
  -> OpenCV 图像预处理
  -> 霍夫圆检测定位仪表盘
  -> 裁剪仪表盘 ROI
  -> ResNet18 三分类模型推理
  -> TensorRT 加速部署
  -> 输出偏低 / 正常 / 偏高状态
```

该模块先利用仪表盘近似圆形的几何特征，通过霍夫圆变换快速定位仪表盘区域；然后裁剪出 ROI 区域，送入训练好的 ResNet18 分类模型进行状态判断。为了适配机器狗边缘端算力有限的部署环境，模型进一步使用 TensorRT 进行 FP16 推理加速，降低延迟、提升帧率。

#### 核心优化

* 使用高斯模糊降低图像噪声，提高霍夫圆检测稳定性。
* 对检测到的多个圆形进行合并，避免重复圆和重叠圆干扰。
* 选择最大圆作为仪表盘区域，提升 ROI 裁剪准确性。
* 使用 ResNet18 完成仪表盘三态分类。
* 使用 TensorRT 进行层融合、FP16 精度优化和计算图优化，提升嵌入式端实时推理性能。
* 加入状态缓冲机制，连续多帧结果一致后再更新显示，减少输出抖动。

#### 相关文件

```text
yuxuansai_new/
├── start_dog.py              # 预选赛视觉识别入口
├── start_jetson.py           # Jetson / TensorRT 推理入口或兼容启动入口
├── Dashboard_detec2t.py      # 仪表盘检测与分类相关逻辑
├── detect_dashboard_trt.py   # TensorRT 推理版本
├── checkpoints/              # 模型权重 / TensorRT 引擎文件
├── data/                     # 训练与测试数据
└── requirements.txt
```

> 注意：当前预选赛仪表盘识别应以最终成果代码为准。如果旧 README 中仍出现“狗端 UDP 推流、Jetson 回传 JSON、input-size=160”等旧架构描述，需要根据最终代码实际情况修正，避免和当前成果不一致。

---

### 2. 颜色与形状识别

颜色识别任务要求识别图像中的红、绿、蓝三种颜色物块，并统计不同形状的数量。物块形状包括：

```text
圆盘 / 小球 / 正方体 / 长方体 / 圆柱
```

其中粉色物块按照赛题要求归入红色统计。

#### 技术路线

```text
输入图像
  -> HSV 颜色空间转换
  -> 红 / 绿 / 蓝 / 粉色颜色掩码生成
  -> 背景区域过滤
  -> 形态学开闭运算
  -> 粘连物体分水岭分割
  -> 轮廓提取与去重
  -> 几何特征计算
  -> 形状分类
  -> 绘制轮廓并输出统计结果
```

该模块没有使用深度学习，而是采用传统图像处理方法完成识别。原因是预选赛物块颜色和形状类别固定，传统视觉方法更轻量、可解释性更强，也更适合在机器人端稳定运行。

#### 核心优化

* **背景去除优化**：在颜色阈值基础上增加低饱和度、低亮度、高亮低饱和区域过滤，减少桌面和反光背景误检。
* **粉色归红处理**：将粉色单独设置 HSV 阈值区间，先独立识别，再归入红色统计，避免红色和粉色互相干扰。
* **粘连物体分割**：对粘连区域使用分水岭算法切分，提升多个物块靠近时的检测稳定性。
* **形状分类增强**：结合圆形度、长宽比、直角数量、凸包密实度、亮度标准差等特征，区分圆盘、小球、正方体、长方体和圆柱。
* **中文结果显示**：在图像上绘制中文标签，并在终端输出颜色与形状统计结果。

#### 输出示例

```text
颜色统计：
红色 7 个，蓝色 2 个，绿色 3 个

形状统计：
圆盘 2 个，小球 5 个，正方体 1 个，长方体 3 个，圆柱 1 个
```

---

### 3. ROS 程序题：邮政编码查询服务

ROS 程序题基于 ROS1 Service 通信机制，实现了一个邮政编码查询系统。系统采用客户端 / 服务端架构：客户端输入城市名称，服务端查询内置数据表，并返回对应邮政编码和位置信息。

#### 系统架构

```text
ROS Master
  -> 注册 /query_postcode 服务
  -> 服务端 postcode_server.py
  -> 自定义 QueryPostcode.srv
  -> 客户端 postcode_client.py
  -> 返回查询结果
```

#### 功能特点

* 使用自定义 `.srv` 文件定义请求和响应格式。
* 请求字段为城市名称。
* 响应字段使用单个 `result` 字符串，便于中文格式化输出。
* 内置 10 个城市的邮政编码和位置信息。
* 支持客户端节点调用，也支持 `rosservice call` 手动测试。
* 对服务未启动、参数错误、城市不存在等情况加入异常处理。

#### 示例城市数据

| 城市  | 邮政编码   | 位置   |
| --- | ------ | ---- |
| 广州  | 510000 | 广东省  |
| 成都  | 610000 | 四川省  |
| 首尔  | 04547  | 韩国   |
| 悉尼  | 2000   | 澳大利亚 |
| 纽约  | 10001  | 美国   |
| 杭州  | 310000 | 浙江省  |
| 卑尔根 | 5003   | 挪威   |
| 普洱  | 665000 | 云南省  |
| 西安  | 710000 | 陕西省  |
| 景德镇 | 333000 | 江西省  |

#### 运行流程

```bash
# 终端 1：启动 ROS Master
roscore
```

```bash
# 终端 2：启动服务端
rosrun postcode_service postcode_server.py
```

```bash
# 终端 3：客户端查询
rosrun postcode_service postcode_client.py 广州
```

也可以使用：

```bash
rosservice call /query_postcode "city: '广州'"
```

---

### 4. ROS 基础操作

ROS 基础操作部分包含两个任务：

```text
任务 1：使用 rosbag 录制并再现 turtlesim 小乌龟运动轨迹
任务 2：配置 ROS 工作空间覆盖机制
```

#### 4.1 turtlesim 轨迹录制与复现

该任务使用 `rosbag` 录制 turtlesim 运行过程中的关键话题，并在之后进行回放，复现小乌龟运动轨迹。

录制的话题包括：

```text
/turtle1/cmd_vel
/turtle1/pose
```

其中：

* `/turtle1/cmd_vel` 记录键盘控制发送的速度指令。
* `/turtle1/pose` 记录小乌龟的实际位置和朝向。

只录制速度指令时，回放轨迹可能存在偏差；加入位姿话题后，轨迹复现更加完整稳定。

典型命令：

```bash
roscore
```

```bash
rosrun turtlesim turtlesim_node
```

```bash
rosrun turtlesim turtle_teleop_key
```

```bash
rosbag record -O run.bag /turtle1/cmd_vel /turtle1/pose
```

回放：

```bash
rosbag play run.bag
```

#### 4.2 ROS 工作空间覆盖

该任务用于验证 ROS 多工作空间环境下的包查找优先级。通过配置 `ROS_PACKAGE_PATH` 和环境变量，使指定工作空间或 `/opt` 路径下的功能包优先生效。

验证方式包括：

```bash
echo $ROS_PACKAGE_PATH
rospack find ros_tutorials
roscd ros_tutorials
pwd
```

该部分主要用于掌握 ROS 包查找机制、工作空间叠加关系和环境配置方法。

---

### 5. ROS 建图与导航

ROS 建图与导航部分包含三个任务：

```text
任务 1：新建四面墙体包围的 Gazebo 仿真环境，并使用 gmapping 建图
任务 2：修改 Gazebo 环境，并实现地图自动保存
任务 3：基于已知地图完成小车“马”字轨迹导航
```

#### 技术路线

```text
Gazebo 仿真环境
  -> 机器人模型与传感器加载
  -> gmapping 建图
  -> map_saver 保存地图
  -> map_server 加载地图
  -> 定位与路径规划
  -> 多目标点顺序导航
  -> 小车轨迹写“马”字
```

#### 核心内容

* 使用 Gazebo 搭建四面墙体包围的仿真环境。
* 使用 gmapping 融合激光雷达与里程计信息，构建二维栅格地图。
* 使用 map_saver 保存地图文件，并规范地图保存路径。
* 修改 Gazebo 环境，验证建图流程的可迁移性。
* 编写 launch 文件，整合仿真、建图、地图保存等流程。
* 编写多目标点导航逻辑，使小车按顺序移动并形成“马”字轨迹。

#### 优化点

* 调整 gmapping 参数，减少地图重影和墙体边缘模糊。
* 统一地图保存目录，避免地图文件散落在不同路径。
* 编写延迟自动保存脚本，减少手动执行 `map_saver` 的步骤。
* 将建图、导航、目标点控制流程模块化，便于复现和验收。

---

### 6. 预选赛成果总结

预选赛阶段完成了从视觉识别到 ROS 操作、建图导航的完整任务闭环：

```text
视觉识别：
  仪表盘三态识别 + RGB 三色五类形状识别

ROS 通信：
  自定义 Service + 邮政编码查询服务

ROS 基础：
  rosbag 轨迹录制与复现 + 工作空间覆盖配置

建图导航：
  Gazebo 环境搭建 + gmapping 建图 + 地图保存 + 小车“马”字导航
```

整体来看，预选赛部分不仅完成了单点算法验证，也覆盖了机器人系统开发中的视觉感知、通信机制、仿真建图和导航控制，为国赛阶段的四足机器人巡检、避障、识别和任务调度打下了基础。

---

### 预选赛目录建议

如果仓库中后续需要继续整理预选赛代码，建议按以下方式组织：

```text
预选赛/
├── dashboard_recognition/       # 表针识别：霍夫圆 + ResNet18 + TensorRT
├── color_shape_recognition/     # 颜色与形状识别：HSV + 分水岭 + 几何特征
├── postcode_service/            # ROS Service 邮政编码查询
├── rosbag_turtlesim/            # turtlesim 轨迹录制与复现
├── workspace_overlay/           # ROS 工作空间覆盖实验
├── gazebo_mapping_navigation/   # Gazebo 建图与导航、小车“马”字轨迹
├── docs/                        # 技术报告、流程图、测试结果
└── README.md
```

当前根目录中如果仍使用 `yuxuansai_new/` 保存预选赛视觉代码，也可以先保留原目录，但 README 中应明确说明：`yuxuansai_new/` 只是预选赛视觉识别部分，不代表预选赛全部内容。

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
