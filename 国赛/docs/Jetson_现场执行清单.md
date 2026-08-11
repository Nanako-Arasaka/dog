# Jetson 现场执行清单（拿到机器后一次跑通）

> 前提：代码最终版在 `/Users/silencecf/Documents/DOG/tmp/dog_repo_review/dog_repo/国赛`
> 目标：把"代码已就绪"变成"现场能跑 10 分钟拿分"

---

## 第一步：部署与建图复核
- [ ] 把 `jetson_payload/slam_maps/` 及整个 `国赛` 项目传到 Jetson（`scripts/upload_slam_maps_to_jetson.ps1`，乱码已修为 `国赛`）。
- [ ] 确认 Jetson 上 RealSense 设备节点（通常为 `/dev/video4`）。
- [ ] 确认 SLAM 地图已建好（用户已在 Jetson 用 Intel RealSense 成功建图）。

## 第二步：地图文件名对齐（极易踩坑，必须做）
- [ ] 在 Jetson 上找到建图产物真实文件名，例如：
  ```bash
  ls -lh ~/Desktop/guosai/slam_maps/*.osa
  ```
- [ ] 打开 `config/guosai_final.yaml`，把 `slam.map_path` 改成上面真实文件名（**不要**写死不存在的 `guosai_rgbd_map_FINAL.osa`）。
- [ ] 同理核对 `realsense` 配置 yaml 名是否与 `config` 引用一致。

## 第三步：语音播报配置（20 分，必做）
- [ ] 跑 `bash scripts/check_onboard_audio.sh`：
  - 出现板载声卡（含 `tegra`/`rt565x`/`i2s`）→ 狗自带扬声器可驱动，记下列名。
  - 只有 `usb` → 接外置 USB 扬声器，记下列名。
- [ ] 编辑 `config/guosai_final.yaml`：
  ```yaml
  voice_broadcast:
    enabled: true
    engine: aplay          # mock -> aplay
    device: "plughw:X,0"   # 填上一步查到的声卡
  ```
- [ ] 确认 `output/audio/` 下 12 个 wav 已随项目传到 Jetson。

## 第四步：航点采集（建图后第一步）
- [ ] `bash scripts/guosai_onekey.sh collect` 沿场地走一圈，采 13 个航点。
- [ ] 重传 `waypoints_FINAL.yaml` 到 Jetson。
- [ ] 验证 `cat jetson_payload/slam_maps/waypoints_FINAL.yaml` 不再是全 0。

## 第五步：手眼标定
- [ ] 真实光照下用棋盘格复标 `cam2arm`（当前初值 x=0.255,y=-0.06,z=-0.55），更新 `arm_grasp` 标定文件。

## 第六步：preflight 自查
- [ ] `python3 scripts/preflight_guosai_final.py --config config/guosai_final.yaml --root .`
- [ ] 确认：语音 12/12 wav 就位、engine≠mock、device 非空、航点非 0、SLAM 地图存在。
- [ ] 任一 ERROR/WARN 先消掉再往下。

## 第七步：联调
- [ ] `bash scripts/guosai_onekey.sh dry-run` —— 空跑验证时序不卡。
- [ ] `bash scripts/guosai_onekey.sh final` —— 完整 10 分钟跑避障→巡检→抓红条→放对应箱。
- [ ] 重点看：4 次语音是否出声、红条是否抓对箱（A→A 箱）、掉落/超时容错是否触发正确。

## 第八步：合规复核
- [ ] 扬声器数量：内置+外置是否被算 2 个？赛前与主办方确认。
- [ ] 设备限额：AI 主机 ≤ Xavier NX、机械臂 ≤6 自由度、无激光雷达。

---
> 凡标注"Mac 不可做"的步骤，提前在 Mac 上把命令/配置模板准备好，到现场只填值、不调试逻辑。
