# Jetson 现场执行清单（国赛最终版）

> 更新：2026-08-13 · Jetson：Orin NX · 代码仓库：`/home/jetson/Desktop/guosai/dog_repo`（与 GitHub main 同步）
> 目标：把"已跑通 dry-run/机械臂"变成"正赛 10 分钟拿分"

---

## 第一步：部署与代码同步（✅ 已基本完成，验证即可）

- [x] 国赛项目已部署到 Jetson `/home/jetson/Desktop/guosai/dog_repo`（git 仓库，remote=GitHub，SSH 直推已配好）
- [x] RealSense D435i 已插，`/dev/video0~5` 全部识别（launch camera-id=4 正确）
- [x] Orbbec/Astra 相机已确认（机械臂视觉专用，`/rgbd_cam/*`），2 相机方案已与主办方确认合规
- [x] SLAM 地图已建（`guosai_rgbd_map_v4.osa` 308MB + `localization_v4.yaml`），config 已对齐
- [x] ORB 词典 fallback 已配（`/home/jetson/ORB_SLAM3/Vocabulary/ORBvoc.txt`，139MB 不入库）
- [x] 机械臂控制板 CH340 已接，`/dev/ttyUSB0`，波特率 1000000
- [ ] ⚠️ Jetson 代码保持最新：`cd ~/Desktop/guosai/dog_repo && git pull`（上机前先拉一次）

## 第二步：语音播报配置（20 分，必做）

- [ ] 跑 `bash scripts/check_onboard_audio.sh`：
  - 板载 tegra-dlink 是空 pipe 出不了声 → **必须接外置 USB 扬声器**，记下卡号
- [ ] 编辑 `config/guosai_final.yaml`：
  ```yaml
  voice_broadcast:
    enabled: true
    engine: aplay          # mock -> aplay
    device: "plughw:X,0"   # 填上一步 USB 扬声器卡号
  ```
- [ ] 确认 `output/audio/` 下 12 个 wav 就位（preflight 已验证 12/12）
- [ ] 实测：发一条巡检结果，确认能听到中文语音

## 第三步：航点采集（导航前提）

- [ ] `bash scripts/guosai_onekey.sh collect --load-existing-map`（用已有 v4 地图，不重新建图）
- [ ] 沿场地走一圈，采 13 个航点
- [ ] 验证 `cat waypoints_FINAL.yaml` 不再是全 0

## 第四步：机械臂联调复核（✅ 已跑通，上机快速复验）

- [x] serial_bridge_node 已集成进 launch（arm_control 之前启动）
- [x] SDK 已 symlink 到 `~/ros_robot_controller_sdk.py`，msgs Python 模块已复制
- [x] 5 舵机已确认响应 set_position（踩坑：VIN 4.1V 不代表舵机有电，需外部 12V 供电）
- [ ] 复验：`/usr/bin/python3` 跑 SDK 冒烟（蜂鸣 + 读 5 舵机位置）
- [ ] 复验：launch 启动后 serial_bridge 无报错，`/arm/command` 发 home 能响应
- ⚠️ 注意：单舵机命令 5 秒到位、多舵机同时命令慢/不响应 → 正式抓取前重点验证多舵机时序

## 第五步：机械臂视觉验证（Orbbec，50 分抓取关键）

- [ ] 确认 Orbbec 是 `/dev/video0`（或改 launch 里 `camera_index` 指向真实节点）
- [ ] 启动 astra_camera_node → 确认 `/rgbd_cam/color/image_rect_color` 话题在发
- [ ] ⚠️ **深度真伪验证**：当前 astra 发伪深度 0.5m，抓取 z 会偏：
  - 放红条在机械臂前方 → `/vision/detect_request` → 看 `/vision/grasp_pose` 输出
  - z 偏 → cam2arm.z 补偿（5 分钟）；x 随距离变 → 需真深度（装 pyorbbecsdk 或换方案）
- [ ] 真实抓取冒烟：放红条 → 检测 → 直抓 → 看命中率（cam2arm 校准）

## 第六步：手眼标定（50 分抓取前提）

- [ ] 真实光照下用棋盘格复标 `cam2arm`（当前初值 x=0.255, y=-0.06, z=-0.55）
- [ ] 6 月已知问题复验：视觉 x 偏大 ~10cm → 抓取后根据偏差调 cam2arm.x
- [ ] 更新 `arm_grasp/config/grasp_config.yaml` 的 `camera_to_arm`

## 第七步：preflight 自查

- [ ] `bash scripts/guosai_onekey.sh preflight`（自带 source_ros，别裸跑 python3）
- [ ] 确认：语音 12/12、engine≠mock、device 非空、航点非 0、SLAM 地图存在、机械臂 msgs 包 OK
- [ ] 任一 ERROR/WARN 先消掉再往下

## 第八步：联调（dry-run → final）

- [ ] `bash scripts/guosai_onekey.sh dry-run` —— 空跑验证时序不卡
- [ ] 真机单跑巡检识别（近距 1m + 远距 2m+）：`live_detect_yolo_opencv.py --model best.pt --camera-id 4 --camera-path /dev/video4 --no-gui --no-stream`，看 `src=` 是 color/angle 非 unknown
- [ ] `bash scripts/guosai_onekey.sh final` —— 完整 10 分钟跑避障→巡检→抓红条→放对应箱
- [ ] 重点看：4 次语音是否出声、红条是否抓对箱（A→A 箱）、掉落/超时容错是否触发正确、多舵机命令是否卡顿

## 第九步：合规复核（赛前）

- [x] 扬声器数量：内置+外置是否被算 2 个？赛前与主办方确认
- [ ] 机械臂 ≤6 自由度、无激光雷达、除相机/扬声器/AI 主机外无其他加装
- [ ] 备用机与正式机一起检录（若规则要求）

---

## 现场应急速查

| 症状 | 处理 |
|---|---|
| 语音无声 | 检查 USB 扬声器是否插、`device` 卡号、engine=aplay、aplay -l 复看 |
| 机械臂不动 | `/dev/ttyUSB0` 存在？`/usr/bin/python3`（非 conda）？外部 12V 供电？单舵机先测 |
| 抓取偏 | 看 grasp_pose 的 x/y/z，调 cam2arm（z 偏=补偿，x 随距离变=深度问题） |
| 巡检读表错 | 看 `src=`：color 正常 / angle 兜底 / unknown=失败，查光照与距离 |
| 节点起不来 | 先 `bash scripts/guosai_onekey.sh preflight`，别裸跑 python3 |
| 摄像头错位 | `ls /dev/video*` + v4l2-ctl 确认编号，改 config/launch 的 camera_index |

---

> 已勾选 [x] 项表示 8/12-13 联调已验证通过，上机只做快速复验；未勾选 [ ] 项为必做现场动作。
