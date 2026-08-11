# 国赛四足机器人项目 · 接手提示词 / Handoff 文档

> 用途：给**零基础接手的人（或下一位 AI 助手）**一份能直接开干的交接材料。
> 下面「① 直接复制给 AI 的提示词」即可让对方 0 基础接手；后面的附录是备查细节。

---

## ① 直接复制给 AI 的提示词

```
你将要接手一个「2026 中国高校智能机器人创意大赛 · 足式机器人专项赛 · 四足机器人（大型组）」的比赛项目代码。
这是一个绝影 Lite 四足机器人 + Jetson 算力板跑全部高层算法（视觉/SLAM/机械臂/任务状态机）、狗本体只做底层运动的系统。
比赛任务：10 分钟自主完成「避障(10分) → 巡检识别(40分) → 抓红色长条(50分) → 放到对应字母纸箱」。

【⚠️ 最重要的一件事：项目真正的最终版不在这个工作区】
当前打开的 /Users/silencecf/code/DOG/dog_repo/国赛 是 7月6日的旧版，缺 nodes/ 和 launch/ 两个核心目录。
真正的最新完整代码在：/Users/silencecf/Documents/DOG/tmp/dog_repo_review/dog_repo/国赛
后续所有修改都基于这个目录。三个 git 仓库 HEAD 都是 424cedd，nodes/launch/scripts 是未提交的工作区改动（不要去 git 里找）。

【请先做三件事】
1. 读 /Users/silencecf/Documents/DOG/tmp/dog_repo_review/dog_repo/国赛/docs/接手指南_HANDOFF.md（本文件）的附录，搞清架构、topic 表、状态机、已知缺口。
2. 读 arm_grasp/arm_grasp/task_manager_node.py（575 行 FSM，是整个系统的总指挥）。
3. 读 nodes/voice_broadcast_node.py + config/guosai_final.yaml + launch/guosai_final.launch.py（语音播报，目前只是框架草稿）。

【当前真实进度】
- 避障 / 巡检识别 / 机械臂抓取放置 / SLAM导航 / 状态机 都已完成并可跑。
- 唯一明显缺口是「语音播报」：框架已写好但没完善（见下方缺口清单）。
- 还有两个必须现场做的硬骨头：航点采集（waypoints_FINAL.yaml 全是 0）、Jetson 上的 SLAM 地图路径确认。

【请你接着干的优先项（按价值排序）】
1. ✅【已完成】修 bridge 让 low/high 区分透传：integration_bridge 现在额外发布 /inspection/all_detailed
   （A:low,B:normal,C:high,D:normal），/inspection/all 仍保持 abnormal/normal 给 FSM。语音节点订阅 detailed。
2. ✅【已完成】用 TTS 生成 12 个 wav（A/B/C/D × low/normal/high），在 output/audio/，命名 A_low.wav 等。
   重生成脚本 scripts/gen_voice_audio.sh。
3. 【现场一步】把 config/guosai_final.yaml 的 voice_broadcast.engine 从 mock 改 aplay（真正出声）。
4. 【现场必做】现场采集航点（bash scripts/guosai_onekey.sh collect），跑 scripts/preflight_guosai_final.py 自查。
   waypoints_FINAL.yaml 全 0 是纯现场任务，preflight 会检测全 0 并报错。

【环境提示】
- 这台 Mac 上只能做静态检查（python3 -m py_compile）和改代码；ROS2/rclpy 只在 Jetson（Ubuntu）上能真正运行。
- 规则 PDF：/Users/silencecf/Downloads/2026（6.24）高校智能机器人创意大赛规则说明_副本.pdf（只有地图图）；
  四足大型组详细规则：arm_grasp/2026年中国高校智能机器人创意大赛（四足大型组）.pdf。
- 不要动 .workbuddy 目录。改动后记得更新本接手文档的缺口清单。

请先通读上面提到的文件，给我一份「你理解的任务拆解 + 你建议的下一步」，再动手写代码。
```

---

## ② 附录：项目全貌（备查）

### 2.1 比赛规则速记

| 项 | 内容 |
|---|---|
| 设备 | 绝影 Lite 四足 + Nvidia AI 主机(≤Jetson Xavier NX) + 1 扬声器 + 1 机械臂(臂展≤50cm/自由度≤6/重量≤2kg) |
| 禁用 | **激光雷达**（发现直接取消成绩） |
| 场地 | 5000×6000mm，5 区：出发区 / 障碍区 / 检测区 / 抓取区 / 放置区 |
| 时间 | 测试 4 分钟 + 正赛 10 分钟（可跑 2 轮取最好） |
| 评分 | 线下挑战 60% + 技术文档 30% + 线上视频 10% |

三大任务：

1. **避障（10 分）**：障碍区 2 个 67×32×32cm 锥桶，自主绕开通过。
2. **巡检识别（40 分）**：检测区配电柜+变压器两侧贴 A/B/C/D 字母图 + A4 仪表盘。指针 **黄=偏低(异常) / 绿=正常 / 红=偏高(异常)**。需做 **4 次语音播报**，每次 10 分（字母 5 + 状态 5）。**无声只有终端输出 → 只给 2.5 分/次**（直接丢 20 分）。
3. **长条抓取（50 分）**：抓取区 800×600×500 高台上 4 个 100×50×50mm 长条（红=异常 / 绿=正常）。抓**红色**长条搬到放置区对应字母纸箱。抓 2 次：抓成功悬空>3s 10 + 搬运跨放置区边界 10 + 正确放置 5。

扣分：搬运掉落 −5/次（最多 −10，**掉 3 次结束**）；碰障碍/道具 −5/次（最多 −30）。

### 2.2 目录结构与职责（最终版）

```
国赛/
├── launch/guosai_final.launch.py   # 一键拉起全部节点（总入口）
├── config/guosai_final.yaml        # 全局配置（slam/realsense/orbslam3/motion/navigation/cone/inspection/arm/fsm/voice_broadcast）
├── nodes/
│   ├── localization_watchdog.py     # 定位稳定性判定 + 丢失急停 → /localization/ok
│   ├── waypoint_navigator.py       # 航点导航：订阅 /waypoint/goal → 发 /motion/nav_cmd + /waypoint/status
│   ├── motion_mux.py                # 速度仲裁（避障优先于导航）+ UDP 下发狗端 + 超时保护
│   ├── cone_avoidance_node.py      # 锥桶避障 ROS 节点，受 /motion/enable_cone_avoidance 门控
│   └── voice_broadcast_node.py     # ★ 语音播报（框架草稿，待完善）
├── integration_bridge/             # /bridge/inspection_result → /inspection/all；/bridge/placement_zone → /placement/recognized_zone；默认冻结 A/B/C/D
├── live_detect_yolo_opencv.py      # 5 类 YOLO(zone_A/B/C/D/gauge) + OpenCV 仪表读数 → /bridge/inspection_result
├── obstacle_avoidance/             # YOLO 检 cone → 规则避障 → UDP 狗端（只在障碍区启用）
├── arm_grasp/                      # JetArm 6DOF + Astra 深度 + 手眼标定 + IK + task_manager_node(FSM)
├── controller/                     # Lite2 UDP 运动接收器 + ORB-SLAM3 + goal_controller
├── src/hardware/speaker/interface.py  # AudioFileSpeaker.play(key)（mock/aplay/ffplay/powershell）— 早期 DI 架构，与 ROS2 脱节
├── scripts/guosai_onekey.sh        # collect / final / dry-run / preflight / all
├── scripts/preflight_guosai_final.py  # 起飞前检查（地图/航点是否全0/ROS包/相机）
├── scripts/waypoint_capture_tool.py   # 交互式航点采集
└── jetson_payload/slam_maps/       # guosai_rgbd_map_FINAL.osa(322MB) + yaml + waypoints_FINAL.yaml
```

### 2.3 ROS2 Topic / Service 速查

| Topic | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/camera_pose` | PoseStamped | 发布 | ORB-SLAM3 位姿 |
| `/localization/ok` | Bool | 发布 | localization_watchdog 定位是否稳定 |
| `/waypoint/goal` | String | 发布 | 目标航点名 |
| `/waypoint/status` | String | 发布 | `arrived:<name>` |
| `/motion/nav_cmd` `/motion/avoid_cmd` | String | 发布 | 导航/避障速度指令 |
| `/motion/stop` | Bool | 发布 | 急停 |
| `/motion/enable_cone_avoidance` | Bool | 发布 | 锥桶避障门控 |
| `/motion_mux/state` | String | 发布 | 仲裁状态 |
| `/bridge/inspection_result` | String | 发布 | 原始巡检结果 `A:low,B:normal,...` 或 JSON |
| `/inspection/all` | String | 发布 | 冻结后 `A:abnormal,B:normal,C:abnormal,D:normal`（FSM 消费，abnormal/normal） |
| `/inspection/all_detailed` | String | 发布 | 冻结后 `A:low,B:normal,C:high,D:normal`（语音播报消费，保留偏低/偏高） |
| `/inspection/target_zones` | String | 发布 | 异常字母逗号串 |
| `/competition/state` | String | 发布 | WAITING_INSPECTION / INSPECTION_PROGRESS:... / INSPECTION_FROZEN:... |
| `/placement/recognized_zone` | String | 发布 | 放置区识别字母 |
| `/vision/grasp_pose` `/arm/command` `/arm/feedback` | String | 发布 | 机械臂视觉/指令/反馈 |
| `/task/direct_grasp` `/task/status` `/task/start` `/task/reset` | String | 发布 | 任务层指令 |
| `/arm/grasp_red_bar` | Trigger(svc) | 调用 | 抓取红色长条 |
| `/arm/place_A..D` | Trigger(svc) | 调用 | 放置到对应字母箱 |

### 2.4 任务状态机（task_manager_node.py，13 态）

```
WAIT_LOCALIZATION → GO_START → GO_OBSTACLE_ENTRY
  → OBSTACLE_ZONE(开避障) → GO_INSPECTION ×4 航点
  → WAIT_INSPECTION(等 /inspection/all 四区齐全)
  → GO_PICK → GRASP → GO_PLACE(按异常字母选 place_A/B/C/D) → PLACE
  → (循环第二个异常区) → GO_FINISH → DONE
```
容错：机械臂 3 次重试、巡检 45s 总超时/每航点 5s、定位丢失急停、避障优先仲裁。

### 2.5 已知缺口清单（接手后照着填）

- [x] **语音播报 low/high 透传**：已修。`integration_bridge` 现在同时发布两个 topic：
  `/inspection/all`（`A:abnormal,B:normal,...`，FSM 消费，契约不变）+
  `/inspection/all_detailed`（`A:low,B:normal,C:high,D:normal`，语音播报消费，保留偏低/偏高）。
  改动：`schemas.py`（加 `zone_state_detailed` / `format_inspection_all_detailed`，老接口不动）、
  `inspection_freezer.py`（加 `frozen_text_detailed`）、`bridge_node.py`+`ros_publishers.py`+`bridge_core.py`（发布 detailed）、
  `voice_broadcast_node.py`（主订阅 detailed，`/inspection/all` 降级兜底）。测试 11/11 通过。
- [x] **12 个 wav 生成**：已生成到 `output/audio/`，命名 `A_low.wav` 等，内容「X区，仪表偏低/正常/偏高」，
  mono 22050Hz 16-bit PCM（aplay/ffplay 兼容）。重生成脚本 `scripts/gen_voice_audio.sh`（macOS say+afconvert）。
- [x] **播报节点接 detailed topic**：`voice_broadcast_node.py` 已订阅 `/inspection/all_detailed`，
  low/normal/high → `A_low.wav`/`A_normal.wav`/`A_high.wav` 正确映射（已验证）。**剩现场一步**：
  Jetson 上把 `config/guosai_final.yaml` 的 `voice_broadcast.engine` 从 `mock` 改成 `aplay`（真正出声）。
  仍与 `src/hardware/speaker/interface.py` 的 `AudioFileSpeaker` 保持解耦（MiniSpeaker 内联）。
- [ ] **航点采集**：`waypoints_FINAL.yaml`（在 Jetson `/home/jetson/Desktop/guosai/slam_maps/`）13 个航点 x/y/yaw 全 0，
  必须现场跑 `bash scripts/guosai_onekey.sh collect` 采集。`config/waypoints_FINAL.template.yaml` 是空模板。
  `preflight_guosai_final.py` 会检测全 0 并报错提示修复命令——纯现场任务，Mac 上无法做。
- [ ] **SLAM 路径确认**：`guosai_final.yaml` 硬编码 `/home/jetson/Desktop/guosai/slam_maps/`，确认 Jetson 已上传（有 `upload_slam_maps_to_jetson.ps1`）。
- [ ] **手眼/底盘坐标标定精化**：cam2arm(x=0.255,y=-0.06,z=-0.55) + ORB-SLAM3 与航点对齐。

### 2.6 怎么跑 / 测

- Jetson(Ubuntu+ROS2)：`ros2 launch guosai_final.launch.py`（参数 start_realsense/start_orbslam3/start_perception/start_arm/start_voice，默认全 true；dry_run=true 测 FSM）。
- 起飞前：`python3 scripts/preflight_guosai_review_guosai_final.py`（注：实际文件名 `preflight_guosai_final.py`）。
- 本机(Mac)只能 `python3 -m py_compile <file>` 做语法校验 + 改代码，不能跑 ROS2。
- 语音播报节点已能 mock 自测：收到 `/inspection/all` 即按 A→B→C→D 顺序 log 播放计划，同一文本不重播，`/competition/state` 回 WAITING_INSPECTION 时重新武装。

### 2.7 别踩的坑

- 改完代码记得同步回 `/Users/silencecf/Documents/DOG/tmp/dog_repo_review/dog_repo/国赛`，**别在旧的工作区 `code/DOG/dog_repo/国赛` 改**（那是旧版，缺 nodes/launch）。
- 不要把语音播报接成"只有终端输出"——规则里那样只给 2.5 分/次。
- 锥桶避障只在障碍区开启（`/motion/enable_cone_avoidance`），离开要关，否则会误避。
- `git` 里找不到最终集成的 nodes/launch（未提交工作区改动），别以为丢失了。
