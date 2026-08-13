# AprilTag 世界坐标标定 · 现场操作指引

> 目标：把 10 个 tag 的「世界坐标（SLAM 坐标系下的 6DoF）」标定进 `config/tags.yaml`，
> 让 `tag_localizer_node` 在 SLAM 丢失时能靠 tag 兜底定位。
> 工具：`tools/calibrate_tags.py`（交互式，须在 **Jetson 上跑**）。

---

## 0. 一句话流程

```
贴好 10 个 tag → 起 SLAM+相机 → 逐 tag 标定 → --verify 验证 → 置 enabled:true
```

## 1. 前置条件（缺一不可，先自查）

| # | 检查项 | 命令 / 说明 |
|---|---|---|
| 1 | 代码已同步 | Jetson 上 `git pull`（拿到 `calibrate_tags.py`、`config/tags.yaml`） |
| 2 | 官方 apriltag 库已装 | `apt install libapriltag-dev` + `pip install apriltag`。<br>⚠️ JetPack 5 自带 OpenCV 4.5 **不含** AprilTag 字典，降级后端不可用，**必须装官方库** |
| 3 | SLAM + RealSense 已启动 | 跑 `bash scripts/guosai_onekey.sh`（或 launch），`/camera_pose` 在持续发布 |
| 4 | 定位正常 | `ros2 topic hz /camera_pose` ≥ 10Hz，停稳后不漂、无跳变 |
| 5 | 10 个 tag 已贴好 | A4 打印件 18cm、正对狗来向、高度≈相机高度（45cm）、避开强光直射 |

> ⚠️ **坐标系铁律**：标定出的坐标是「**建图那次** ORB-SLAM3 世界系」下的。
> 地图一次建好、永不重建，否则所有 tag 坐标作废（与航点同理）。

## 2. 标定（逐 tag 交互采集）

```bash
cd /home/jetson/Desktop/guosai/dog_repo/国赛
python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml
```

对每个 tag（id 1→10，即点一→点十）：

1. **把狗开到能看到该 tag 的位置**（2~3m 内，画面中 tag 清晰、别太斜）
2. 工具提示 `SLAM 定位正常后按 Enter` → **先等狗停稳、SLAM 稳定**（工具会自动等：连续 10 帧位移 <0.03m、yaw <0.1rad）→ 按 Enter
3. 工具自动采 30 帧（SLAM 相机位姿 + tag 检测配对）→ 离群剔除 + 平均 → 打印结果：
   ```
   结果（28/30 帧有效，最大散布 3.2cm）：
     x=2.5134  y=-1.2078  z=0.4512
     yaw=87.21°  pitch=90.02°  roll=0.11°
   ```
4. 确认：
   - **Enter** = 接受 → 进下一个 tag
   - **r** = 重采（散布 >10cm 时工具会警告，建议重采）
   - **s** = 跳过该 tag（稍后单独补）
   - **q** = 退出（已采的会保留写入）

> 中途想只补某几个 tag：`--ids 7,8`（比如只补障碍区出口和仪表箱2侧）。

## 3. 验证（标定是否可用）

```bash
python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml --verify
```

- 对每个已标定 tag：把狗开到能看到的位置 → 等稳定 → Enter → 采 15 帧
- 用标定坐标反推相机位姿，与 SLAM 位姿对比
- **通过标准：平均位置误差 ≤10cm、平均 yaw 误差 ≤5°**（`--verify-max-pos-err` / `--verify-max-yaw-err-deg` 可调）

```
========== 验证汇总 ==========
  tag 1: 位置 4.2cm / yaw 1.3° (15帧) [PASS]
  ...
全部通过。可将 config/guosai_final.yaml 的 tag_localizer.enabled 置为 true。
```

⚠️ 有 FAIL 的：对该 tag 重新标定（去掉 `--verify`），不要直接启用。

## 4. 启用兜底

全部 PASS 后：

```yaml
# config/guosai_final.yaml
tag_localizer:
  enabled: true   # ← 从 false 改为 true
```

重启 launch（或只重启 tag_localizer 相关节点），然后确认：

```bash
ros2 topic echo /tag_localizer/status    # 应出现 ok:id=N,q=... 而非 none
ros2 topic echo /tag_localizer/seen_tags # 可见 tag ID 列表
ros2 topic echo /localization/status     # watchdog 仲裁状态
```

## 5. 快速故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 一直 `no tag` / 采不到 | tag 太远 / 太斜 / 光照强反光 | 靠近、摆正、换角度重试 |
| 提示 apriltag 库不可用 | 没装官方库（降级到 OpenCV 失败） | `apt install libapriltag-dev` + `pip install apriltag` |
| SLAM 稳定等待超时 | 狗在动 / SLAM 在漂 | 停稳狗、确认 `/camera_pose` 稳定 |
| 验证 FAIL | tag 贴歪了 / 采集中狗移动 | 重新贴正 + 重标定该 tag |
| 散布警告（>10cm） | SLAM 漂移或狗移动 | 按 r 重采 |

## 6. 注意

- 标定全程**狗必须静止**（工具会自动等稳定，但人别碰狗、别挡相机）
- 建议按优先级标定：**点七（障碍出口）> 点五/四 > 点一 > 点九**，时间不够先标这 5 个，其余 `--ids` 补齐
- 标定完成后 `tags.yaml` 会自动备份（`.bak_时间戳`），改坏了能回滚
