# 四足机器狗 · 2026 中国高校智能机器人创意大赛（四足大型组）

> 参赛队伍：**Miner**（中南民族大学）　|　赛项：足式机器人挑战赛（专项赛）· 四足机器人（大型组）
>
> 平台：绝影 Lite2 四足机器人 + Jetson Xavier NX + Intel RealSense D435i + JetArm 六自由度机械臂

本项目分两阶段：**预选赛**（仪表盘三态识别，线上视频赛）与**国赛**（避障 → 巡检识别 → 红条抓取与放置，线下挑战 + 技术文档）。

- 预选赛代码：`NEW Edition/`、`new/`
- 国赛代码：`国赛/`（当前为最终集成版，含一键启动、语音播报、SLAM 导航与 Jetson 部署包）

---

## 比赛规则速览（依据官方规则 V1.1 与 6.24 答疑）

| 项 | 内容 |
|---|---|
| 任务分值 | 避障 10 分 + 巡检识别 40 分 + 长条抓取 50 分（满分 100） |
| 总分构成 | 线下挑战 60% + 技术报告 40% |
| 巡检播报 | 仪表盘状态 5 分/次 + 区域字母 5 分/次，共 4 次；仅终端输出无语音得分减半 |
| 抓取规则 | 红色长条=异常，绿色=正常；悬空超 3 s 计成功；掉落每次 -5 分，掉落 3 次结束 |
| 设备约束 | 算力 ≤ Jetson Xavier NX；机械臂臂展 ≤ 50 cm、自由度 ≤ 6、重量 ≤ 2 kg；**禁止激光雷达** |
| 赛制 | 4 分钟测试 + 10 分钟正赛，最多两轮取最好成绩 |

> 技术报告评分：技术方案 40% + 文档呈现 40% + 工程代码 20%（详见 `国赛/docs/技术文档_草稿.md`）。

---

## 一、预选赛 — 仪表盘识别（`NEW Edition/`、`new/`）

预选赛采用线上提交视频方式，任务为仪表盘三态识别（指针位于黄/绿/红区 → 偏低/正常/偏高），要求距相机 ≥ 1 m、终端中文输出、每个表盘旋转 ≥ 3 次。

### 技术路线

```text
摄像头采集 → OpenCV 预处理 → 霍夫圆检测定位仪表盘 → 裁剪 ROI
  → ResNet18 三分类（PyTorch / TensorRT FP16） → 输出 偏低/正常/偏高
```

### 核心优化

- 霍夫圆检测参数（`param1=100`、`param2` 收紧、`dp=1.0`、`min_radius`/`min_dist` 自适应放大）减少误检伪圆
- 圆形确认机制 + `miss_hold_frames=15` 抑制输出抖动
- TensorRT 层融合 + FP16 加速，适配边缘端实时推理
- 状态缓冲：连续多帧一致才更新显示

### 目录结构

```text
NEW Edition/          # 预选赛最终版（本地推理 PyTorch/TRT）
new/                  # 预选赛修订版
├── start_dog.py      # 狗端本地推理入口
├── start_jetson.py   # Jetson / TensorRT 推理入口
├── detect_dashboard_trt.py   # TensorRT 推理（最终版）
├── Dashboard_detec2t.py      # PyTorch 推理（最终版）
├── trans_onnx.py / trans_trt.py  # 模型转换脚本
└── checkpoints/      # 模型权重 / TensorRT 引擎
```

---

## 二、国赛 — 最终集成版（`国赛/`）

### 2.1 系统架构

采用「Jetson 算力板负责主要计算，狗本体只保留底层运动执行与安全兜底」的分布式架构，控制层走 UDP（实时速度指令），任务层走 ROS2 Topic（状态与事件）。

```text
摄像头 (RealSense D435i)
  → Jetson 视觉感知：YOLO 巡检识别 / 锥桶检测 / HSV 红条检测 / 放置区字母识别
  → integration_bridge：状态格式统一 + ROS2 Topic 转发 + 巡检结果冻结
  → 任务状态机：waypoint_navigator → motion_mux → 机械臂 task_manager
  → 语音播报：voice_broadcast_node（A_low.wav 等 12 路预生成音频）
  → controller：ORB-SLAM3 定位 + lite2_motion_receiver（UDP 5005）
  → 狗本体：底层步态执行 + watchdog 超时停机
```

### 2.2 四大任务模块

| 模块 | 关键文件 | 说明 |
|---|---|---|
| 巡检识别 | `live_detect_yolo_opencv.py`、`gauge_reader.py` | YOLO 定位 A/B/C/D 与仪表盘 ROI → OpenCV 管线读指针角度 → low/normal/high；多层降级策略（霍夫圆失败→ROI 中心、直线失败→暗色尖端、色带失败→角度阈值） |
| 锥桶避障 | `cone_avoidance/`、`obstacle_avoidance/cone_strategy.py` | YOLO 检测锥桶（`cone_yolo_best.pt`，conf 0.35）→ 四级规则策略（紧急停车→主动避障→微调→全速），8 Hz UDP 下发 |
| 红条抓取 | `arm_grasp/`（arm_control_node / vision_node / task_manager_node / inspection_memory_node） | HSV 双阈值红条检测 + RealSense 深度 3D 位姿 + 几何法两连杆 IK；视觉反馈闭环（Δz>3 cm 判成功、空抓定向重试 ≤10 次、视觉丢失回退 ≤3 次） |
| SLAM 导航 | `controller/`（ORB-SLAM3、goal_controller、lite2_motion_receiver） | ORB-SLAM3 视觉定位 → 航点导航 → motion_mux 优先级仲裁 → UDP 下发狗体 |

### 2.3 语音播报

`nodes/voice_broadcast_node.py` 订阅巡检结果，按 `A_低/正常/高` 等 12 种组合播放 `output/audio/` 下的预生成 wav（中文播报），支持 mock / aplay / ffplay 引擎，满足「语音播报得满分、仅终端减半」的评分要求。

### 2.4 一键启动（现场）

```bash
# 一键全流程（采集航点 / 预检 / 正式运行）
bash scripts/guosai_onekey.sh
bash scripts/run_guosai_final.sh

# 或 ROS2 统一启动
source /opt/ros/humble/setup.bash
ros2 launch launch/guosai_final.launch.py
```

完整现场流程见 `docs/Jetson_现场执行清单.md`，部署包见 `jetson_payload/`。

### 2.5 国赛目录结构

```text
国赛/
├── launch/guosai_final.launch.py     # 一键启动编排
├── nodes/                            # 语音播报 / 航点导航 / 运动仲裁 / 避障节点 / 定位看门狗
│   ├── voice_broadcast_node.py
│   ├── waypoint_navigator.py
│   ├── motion_mux.py
│   ├── cone_avoidance_node.py
│   └── localization_watchdog.py
├── scripts/                          # guosai_onekey.sh / preflight / waypoint_capture_tool 等
├── config/guosai_final.yaml          # 最终运行配置（SLAM/相机/导航/机械臂/语音/FSM）
├── jetson_payload/                   # Jetson 部署包（SLAM 地图 + 上传脚本）
├── integration_bridge/               # 状态转发层：格式统一 + ROS2 转发 + 巡检冻结
├── arm_grasp/                        # JetArm 机械臂 ROS2 包
├── cone_avoidance/                   # 锥桶避障感知与策略
├── obstacle_avoidance/               # 规则避障策略（旧版模块）
├── controller/                       # ORB-SLAM3 / lite2_motion_receiver / goal_controller
├── live_detect_yolo_opencv.py        # 主线实时巡检
├── gauge_reader.py                   # 仪表盘指针读取（独立模块）
├── camera_input.py                   # 多输入源取流封装
├── vision_server.py                  # TCP 视觉服务（远端推理）
├── output/audio/                     # 12 路中文语音播报音频
├── docs/                             # 技术文档 / HANDOFF / 现场清单 / 视频脚本
├── tools/ tests/ runs/ models/       # 数据集工具、单元测试、训练产物、模型
└── requirements.txt
```

### 2.6 环境要求

- **Jetson**：Jetson Xavier NX、Ubuntu、Python 3.8+、OpenCV、PyTorch + Ultralytics、ROS2 Humble
- **狗本体**：仅运行轻量运动接收与 watchdog，不部署视觉/SLAM/状态机
- **大文件（Git LFS）**：`*.osa`（SLAM 地图）由 Git LFS 管理，clone 后执行 `git lfs pull` 获取实体

```bash
pip install -r 国赛/requirements.txt
git lfs pull   # 拉取 SLAM 地图等大文件
```

### 2.7 关键 ROS2 话题

```text
/bridge/inspection_result   → /inspection/all            # 巡检结果归一化
/bridge/placement_zone      → /placement/recognized_zone # 放置区字母
/inspection/all_detailed                                 # 含 偏低/偏高 详细状态（语音播报主用）
/inspection/target_zones                                 # 异常区域列表
/competition/state                                       # 全局任务状态
/motion/nav_cmd · /motion/avoid_cmd · /motion/stop       # 运动指令（motion_mux 仲裁）
```

---

## 测试

```bash
cd 国赛
python -m pytest -q                                    # 单元测试
python tools/test_camera_input.py                      # 多输入源
python tools/test_remote_perception_client.py          # 远端视觉
python tools/test_speaker_playback.py --save-playback-log  # 语音播报
```

---

## 仓库结构

```text
.
├── README.md          # 本文件
├── NEW Edition/       # 预选赛 · 仪表盘识别（最终版）
├── new/               # 预选赛 · 仪表盘识别（修订版）
├── 国赛/               # 国赛 · 最终集成版（巡检/避障/抓取/放置/语音/SLAM）
└── .gitignore
```

---

## 相关文档

| 文档 | 位置 |
|---|---|
| 国赛技术文档草稿（评审用） | `国赛/docs/技术文档_草稿.md` |
| Jetson 现场执行清单 | `国赛/docs/Jetson_现场执行清单.md` |
| 接手指南（HANDOFF） | `国赛/docs/接手指南_HANDOFF.md` |
| 线上视频拍摄脚本 | `国赛/docs/线上视频_拍摄脚本.md` |
| 机械臂跑通流程 | `国赛/arm_grasp/JetArm_跑通流程.md` |
| 运动控制运行流程 | `国赛/controller/Lite2正式运行流程.txt` |
| 锥桶避障详细说明 | `国赛/obstacle_avoidance/README.md`、`国赛/cone_avoidance/` |
| 状态转发层说明 | `国赛/integration_bridge/README.md` |
