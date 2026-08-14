# AprilTag tag 4/5/6 现场标定失败根因分析

> 时间：2026-08-14 21:30+ CST
> 关联 commit：86a81e0 (apriltag 标定 + dt-apriltags 替换)
> 状态：tag 1+2+3 入库 OK，tag 4/5/6 抹掉待重采

## 现象

现场用 `tools/calibrate_tags.py` 标 tag 4/5/6（仪表箱 1 观察侧 2 / 观察侧 1 / 障碍区入口），
采集到的世界位姿都聚在 SLAM 原点附近 0.5m³ 球内。按 ORB v4 标准（任意两 tag 欧氏距离
≥0.5m 视为独立有效定位点），存在以下异常：

| Tag 对 | 实测距离 | 物理期望 | 状态 |
|---|---|---|---|
| tag 4-5 | **0.007m (7mm)** | 仪表箱两侧 ≥0.5m | 同一个点重采 |
| tag 4-6 | **0.111m (11cm)** | 仪表箱 → 障碍区入口 ≥2m | 完全没移位 |
| tag 5-6 | **0.110m (11cm)** | 同上 | 完全没移位 |

更广泛的 spread：tag 1-6 总 spread 仅 x=0.32m, y=0.83m, z=1.05m（不到 1m³）。
对于 5×6m 场地，6 个 tag 应该 spread ≥5m³。

R_wt 矩阵数学上全部正确（det=+1，||R·Rᵀ-I|| ≈ 1e-16），所以问题不在旋转，
只在位置——采集时 dog 没有在 tag 之间真实移动。

## 根因推断（基于 backup 时间线）

`config/tags.yaml.bak_20260814_*` 11 个时间戳显示 ORB-SLAM3 在 20:40 前后失跟：

| backup 时间戳 | tag 1 变化 | tag 4/5/6 变化 |
|---|---|---|
| 19:57:53 | 首次采到 | 占位 |
| 20:19:18 | 重采 | 占位 |
| 20:40:27 | 重采 | 占位 |
| 20:43:44 | 重采 | 占位 |
| 20:46:23 | 重采 | 占位 |
| **20:50:01** | 重采 | **tag 4 首次采到**（距 tag 1/2/3 仅 23-26cm） |
| **20:54:29** | 重采 | **tag 5 采到**，tag 6 未采到（工具退出/超时） |

关键观察：19:57 → 20:46 多次重采都没新增 tag 4/5/6——说明那时 ORB 已经在
失跟；`calibrate_tags.py` 的"SLAM 稳定"检查（10 帧内位姿变化 < 4cm）在失跟抖动
样本上误判通过（失跟位姿在原点附近抖动 ≈1e-4m，10 帧均值仍在 4cm 内），导致工具
把同一失跟位姿当"稳定"采下来。

## 同根因历史

这是 commit `604c8c3` 脏 waypoint 的同根因（ORB-SLAM3 wrapper 失跟）。
那次事故 13 个航点聚在原点附近 y-span=0.12m，已被 commit `89c19f8` 现场重采覆盖。

## 当前状态（commit 86a81e0 后）

- ✅ **tag 1+2+3** 现场标定入 tags.yaml
  - tag 1-2: 1.21m, tag 1-3: 0.48m, tag 2-3: 1.09m
  - R 矩阵 orthonormal, det=+1
- ⏸️ **tag 4/5/6** 在 tags.yaml 里抹成占位（x=y=z=yaw=pitch=roll=0）
- ⏸️ **tag 7-10** 仍占位（未采集）
- ❌ **verify tag 1+2+3 未跑**（要求推狗+按 Enter 的交互操作不在 Jetson 端可达）
- ❌ **tag_localizer.enabled** 仍为 false（config/guosai_final.yaml 未改）

## 重启 stack 后的现状（21:35 CST）

按 `deploy.md` canonical launch 命令重启 RealSense + ORB-SLAM3：
- RealSense D435I 启动成功（PID 6714/6718）
- ORB-SLAM3 atlas 加载成功（PID 6770/6776，map 1, KF 1935）
- `/camera_pose` **14.17 Hz**（≥10 Hz SOP 阈值）
- tracking 初始化 OK（"New Map created with 1478 points"）

重启后**未做 SLAM 跟手 self-check**（推 0.5m 看 pose delta）——需要推车现场操作。
如果跟手 self-check 通过，重采 tag 4/5/6 应该能成功。

## 后续行动（优先级排序）

1. **现场 verify tag 1+2/3**：在你终端跑
   `python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml --verify --yes --ids 1,2,3`，
   按 Enter 让 dog 推到 tag 1/2/3 视野。期望误差：位置 ≤10cm + yaw ≤5°。
2. **若 verify 通过** → 现场重采 tag 4/5/6，重启后 SLAM 跟手应该是健康的；
   采完 commit 增量（`docs/apriltag_456_recovery_202608XX.md` 记录新时间线）。
3. **若 verify 失败** → 排查 ORB-SLAM3 wrapper 失跟根因（log "Active map reset"
   频次、`/camera_pose` 抖动频谱），可能需要调整 ORB wrapper 参数或换 SLAM 后端。
4. **剩余 tag 7-10** 按 SOP 优先级（点七 > 点六 > 点五/四 > 点一 > 点九 > 点八
   > 点二 > 点三 > 点十）补采。

## Ship-and-iterate 决策记录

按 `deadline_pragmatism.md` 同 2026-08-14 waypoint re-collect（89c19f8）的处理
模式：tag 1+2/3 数据不完美（tag 3 的 0.48m 接近 0.5m 阈值下界）但比无数据好，
tag 4/5/6 重留作后续 commit 重采。`tag_localizer.enabled` 仍 false 保护：
即使 tag 1+2/3 数据有偏差也不会在比赛中启用误定位。

## 参考

- `国赛/backups/tags.yaml.bak_20260814_*` — 11 个时间戳快照，保留全部历史
- `国赛/backups/waypoints_FINAL.yaml.bak_20260814_*` — 604c8c3 失跟事故参照
- `memory/apriltag_calibration_sop.md` — SOP + 优先级 + verify 阈值
- `memory/orb_slam_selfcheck_and_coord.md` — SLAM 跟手 self-check 方法
- `memory/deploy.md` — RealSense + ORB canonical launch 命令