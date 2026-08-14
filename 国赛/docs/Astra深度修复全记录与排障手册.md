# Astra 深度读取:完整修复记录与现场排障手册

> 适用范围:Jetson Xavier NX(aarch64)+ Orbbec/Astra 深度相机(机械臂抓取视觉,50 分项)
> 目标:让 `/rgbd_cam/depth/image_raw` 输出**真实深度**(16UC1,mm),供 vision_node 反投影抓取位姿
> 本文档既是修复复盘,也是现场排障手册——遇到问题按「第 5 节诊断流程」一步步查

---

## 1. 背景与目标

机械臂抓取(50 分)链路:

```
Astra 相机(RGB 手眼视角)
  ├─ RGB   : /dev/video0 (V4L2, cv2 读取)  → astra_camera_node 转发 → /rgbd_cam/color/image_rect_color
  └─ 深度  : 多后端获取                      → astra_camera_node 转发 → /rgbd_cam/depth/image_raw
                                                    │
                                                    ▼
vision_node: HSV 检测红条 2D 像素 → 深度采样 z → 反投影 x/y → /vision/grasp_pose
                                                    │
                                                    ▼
task_manager → arm_control → 机械臂抓取
```

- 深度单位约定:**mm,16UC1 编码**;0 = 无效(饱和/无回波/超量程)
- vision_node 反投影公式:`x_cam=(px-cx)*z/fx`, `y_cam=(py-cy)*z/fy`, `p_arm=p_cam+cam2arm`
- 关键文件:
  - `arm_grasp/astra_camera_node.py` —— 相机抽象层(多深度后端)
  - `arm_grasp/arm_grasp/vision_node.py` —— 红条检测 + 位姿反投影
  - `arm_grasp/config/grasp_config.yaml` —— 内参兜底/fixed_depth 等配置
  - `config/guosai_final.yaml` 的 `arm:` 段、`launch/guosai_final.launch.py` —— 参数透传

---

## 2. 深度后端架构(astra_camera_node 多后端自动探测)

启动时按 `depth_mode` 依次尝试,`auto` 顺序:**pyorbbec → openni → uvc → realsense → none**:

| 后端 | 原理 | 状态(2026-08-14 现场) |
|---|---|---|
| `pyorbbec` | 官方 pyorbbecsdk2 | ❌ 失败:iSerial=0 bug,枚举 0 设备 |
| `openni` | OpenNI2 + liborbbec 老驱动 | ✅ **正解**:真深度,绕开 pyorbbec 的 serial 逻辑 |
| `uvc` | V4L2 Z16 深度流(/dev/video1) | ❌ 失败:/dev/video1 是 metadata-only,无深度流 |
| `realsense` | 转发狗头 RealSense 深度 | ⚠️ 备选:仅当 RealSense 能看见机械臂台面(现场不可用,视野不重叠) |
| `none` | 不发布深度(安全) | 兜底:抓取侧报 invalid_depth 停止,不误抓 |

关键参数(launch/config 透传):
- `depth_mode`: auto | pyorbbec | openni | uvc | realsense | none
- `openni_redist`: OpenNI2 Redist 路径(空 = 自动探测 `~/openni2/OpenNI-Linux-*/Redist`)
- `openni_fx/fy/cx/cy`: openni 后端内参兜底(默认现场实测 **945.028/945.028/320/400**)
- `fake_depth_fallback`: 深度全失效时发伪深度 0.5m(**正式比赛必须 false**)
- `fixed_depth`(在 grasp_config.yaml):深度无效时用固定台面深度反投影(兜底方案)

---

## 3. 完整修复时间线(复盘)

| 时间 | 事件 | 结论 |
|---|---|---|
| 检查阶段 | 发现 astra_camera_node 发**伪深度(恒 500mm)**+ 近似内参 570.34 | 抓取 z 完全不可靠,50 分项阻塞 |
| 第一次重构 | 伪深度改为多后端(pyorbbec→uvc),不可用则停止发布 | 行为正确,但现场两后端都失败 |
| 现场排查① | pyorbbecsdk2 装上但 `query_devices()`=0;`lsusb` 能看到 2bc5:0614/0511 | 根因:**iSerial=0** |
| 根因确认 | libOrbbecSDK 用 `query_device_access_state_by_serial_number` 索引,0614 的 iSerial=0 触发空串处理,整列设备丢弃;老版本 2.0.18 同样报错 | **换 pyorbbec 版本无效,放弃此路线** |
| 否决方案 | RealSense 转发:被用户否决——Astra 在机械臂手眼,RealSense 在狗头前视,**视野不重叠** | 深度不能跨相机通用 |
| 兜底方案 | `fixed_depth` 固定台面深度反投影(红条竖放在固定高台,z 已知常数) | 已实现,零依赖,比赛保底可用 |
| OpenNI2 路线 | PC 下载 `OpenNI_2.3.0.66` 官方包(226MB 全平台,提取 Arm64 3.3MB)+ `pip install openni` | **验证成功**:深度 640×480 16UC1,645~9810mm,内参 fx=fy=945.028 cx=320 cy=400 |
| 集成 Bug 1 | `cv2.VideoCapture(idx)` 无 `CAP_V4L2` → Jetson 默认 GStreamer 后端打不开 Orbbec RGB,isOpened()=False | 加 `cv2.CAP_V4L2`(与 live_detect.py 一致) |
| 集成 Bug 2 | 8 处 `get_logger().warn('...%s', arg)` → RcutilsLogger 拒绝 positional args,`TypeError` | 全部转 f-string |
| 集成 Bug 3 | `depth.get_camera_params()` **静默失败**(python 绑定无此属性)→ camera_info 用默认 570.34 → 反投影偏差 1.65× | 加日志 + 合理性校验 + openni_fx/fy/cx/cy 参数兜底 |
| 精度增强 | 红条中心 IR 饱和(深度 0)→ 整次抓取 invalid_depth | `_sample_depth` 多级窗口重试(5/12/20px) |
| 诊断增强 | grasp payload 加 `z_src=depth\|fixed` 字段 | 现场可直接判断走了哪个深度源 |

---

## 4. 踩坑经验清单(根因 → 解决,按优先级)

### 4.1 【坑王】pyorbbecsdk2 对老 Astra 设备枚举失败(iSerial=0)
- **症状**:`Context().query_devices().get_device_count()` = 0;报 `Failed to query USB device serial number`;但 `lsusb`/pyusb 能看到 `0x2bc5:0x0614 ORBBEC Depth Sensor`
- **根因**:libOrbbecSDK 内部用 USB **serial number 索引设备**;这台 Depth Sensor 固件 iSerial=0,SDK 触发空串处理逻辑后**整列设备被丢弃**
- **解决**:换 pyorbbec 版本**无效**(2.1.2/2.0.18 同源逻辑);**改用 OpenNI2 老驱动**(见 4.2)
- **教训**:SDK 层枚举失败先查设备 serial/型号,别盲目换版本

### 4.2 OpenNI2 是老 Astra 的正路(自带 liborbbec.so)
- **包**:`OpenNI_2.3.0.66`(Orbbec 官方,含 4 平台;取 `Linux/OpenNI-Linux-Arm64-2.3.0.66/`)
  - 下载:`https://dl.orbbec3d.com/dist/openni2/OpenNI_2.3.0.66.zip`(226MB,含 Android/Linux/Windows)
  - 国内 OSS 镜像(abzg-oss)已失效(NoSuchBucket),别用
  - 已提取小包:`OpenNI_2.3.0.66_arm64_only.zip`(3.3MB)
- **驱动确认**:`Redist/OpenNI2/Drivers/` 必须有 **`liborbbec.so`**(Astra 驱动)+ `orbbec.ini`;没有它 OpenNI2 认不出设备
- **python 绑定**:`pip install openni`(纯 python,ctypes 调 libOpenNI2.so)
- **环境变量**:`OPENNI2_REDIST` 指向 `Redist` 目录(install.sh 会生成 OpenNIDevEnvironment,内含 export);astra_camera_node 会自动探测 `~/openni2/OpenNI-Linux-*/Redist`,无需手写 .bashrc
- **安装顺序坑**:**必须先解压到最终位置再跑 install.sh**——OpenNIDevEnvironment 里写的是解压目录的绝对路径,移动目录后要重新 source
- **udev**:install.sh 装 `556-orbbec-usb.rules`;现场已有 56-orbbec-usb.rules 覆盖 0511/0614 也可
- **初始化顺序**:`openni2.initialize()` → `Device.open_any()` → `create_depth_stream().start()` → `read_frame()`;每步失败给出明确告警

### 4.3 Jetson 上 cv2 打开 UVC 必须显式 CAP_V4L2
- **症状**:`cv2.VideoCapture(0).isOpened()` = False,日志 `cap_gstreamer.cpp ... unable to start pipeline`
- **根因**:Jetson 的 OpenCV 默认后端是 GStreamer,对 Orbbec RGB(UVC/MJPG)起不来
- **解决**:`cv2.VideoCapture(idx, cv2.CAP_V4L2)`(全仓库统一写法)

### 4.4 rclpy 的 logger 不接受 positional format-args
- **症状**:`TypeError: RcutilsLogger.error() takes 2 positional arguments but 3 were given`
- **根因**:`self.get_logger().warn('...%s', arg)` 这种写法在 rclpy **全部不行**(error/warn/info 都一样,只是第一个报错的行先崩)
- **解决**:一律用 f-string:`self.get_logger().warn(f'...{arg}')`;带 throttle 的保留 kwargs:`warn(f'...{exc}', throttle_duration_sec=5)`
- **检查命令**:`grep -rn "get_logger()\.[a-z]*(.*%" --include="*.py" .`(应无输出)

### 4.5 get_camera_params 在 python 绑定里可能不存在
- **症状**:`'VideoStream' object has no attribute 'get_camera_params'`(python 绑定没暴露 C++ 接口)
- **影响**:内参拿不到 → camera_info 用默认 570.34 → 反投影 1.65× 偏差(0.5m 处 x/y 偏 ~10cm+)
- **解决(三层保障)**:
  1. SDK 值优先,但必须过**合理性校验**:`fx/fy>100`、`cx ∈ [0.2w, 0.8w]`、`cy ∈ [0.1h, 0.95h]`
  2. 校验不过/异常 → 用参数 `openni_fx/fy/cx/cy`(默认现场实测 945.028/945.028/320/400)
  3. 全程打日志(哪层生效一目了然)
- **特别坑**:曾实测 SDK 返回 **cx=640**(对 640 宽是边界外异常值,会造成 0.5m 处 ~17cm 误差),校验就是为了拦这种值;**cy=400 是 Astra IR 传感器的稳定特性,别误拦**

### 4.6 RGB 与 IR 深度有 40mm 基线视差
- **现象**:RGB(彩色 sensor)与深度(IR sensor)是两套镜头,同名点像素偏移 = `baseline*fx/z`,0.5m 处约 75px
- **影响**:直接用 RGB 检测的像素去深度图采样,z/x/y 都可能取偏
- **缓解**:astra_camera_node 尝试 `set_image_registration_mode(True)`(深度注册到 RGB 视角),设备支持时生效
- **兜底**:若现场发现 z 抖动/取到背景,改用 `fixed_depth` 固定台面反投影(无视差)

### 4.7 IR 内参 ≠ RGB 内参
- **现象**:camera_info 现在发布的是 openni 后端写入的 **IR 内参(945/320/400)**,但 vision_node 的像素来自 RGB
- **风险**:若注册对齐后 RGB 内参与 IR 不同,x/y 会有系统偏差(如整体偏右)
- **验证**:现场精度测试(红条 0.3/0.5/0.8m 对比)若发现系统偏差,用棋盘格标 RGB 内参或调 openni_cx/cy

### 4.8 红条中心深度为 0(IR 饱和/无回波)
- **现象**:HSV 检测到红条,但采样点深度 0 → `invalid_depth`
- **解决**:`_sample_depth` 多级窗口重试(中心 5px → 12px → 20px,取最近有效深度中值)

---

## 5. 现场排障流程(诊断树——Qwen 照着跑)

> 总原则:从「独立验证」到「节点内验证」到「全链路验证」,逐层隔离。
> 每步跑完记录结果再进下一步。**不要改代码**。

### Step 0:确认症状
先明确"又出问题"是哪一层:
- [ ] 节点启动失败 / 崩溃?
- [ ] 启动成功但日志显示某个后端失败?
- [ ] 深度话题无数据 / 全 0 / 值异常?
- [ ] camera_info 内参不对(不是 945)?
- [ ] vision_node 一直 invalid_depth / grasp_pose 异常?

### Step 1:独立验证 OpenNI2 能否读深度(绕过 ROS,最快定位)
```bash
source ~/openni2/OpenNI-Linux-Arm64-2.3.0.66/OpenNIDevEnvironment
python3 << 'EOF'
import numpy as np
from openni import openni2
openni2.initialize()                       # 失败 → OPENNI2_REDIST 未设/包未装
dev = openni2.Device.open_any()            # 失败 → 设备未插/驱动不认(见 4.2)
print('DEVICE:', dev.get_device_info())
depth = dev.create_depth_stream(); depth.start()
import time
for i in range(5):
    f = depth.read_frame()
    d = np.frombuffer(f.get_buffer_as_uint16(), dtype=np.uint16).reshape(f.height, f.width)
    valid = d[(d>0)&(d<5000)]
    print(f'frame{i}: shape={d.shape} valid_px={valid.size} min={d.min()} max={d.max()}'
          f' mean_valid={valid.mean() if valid.size else 0:.0f}')
    time.sleep(0.1)
depth.stop(); dev.close(); openni2.unload()
EOF
```
**判读**:
- 能出数且 valid_px 稳定(如 >5 万)、min/max 合理(200~5000mm)→ OpenNI2 层正常,**问题在 ROS 集成层**,跳 Step 3
- `initialize()` 报错 → 环境变量问题
- `open_any()` 报 no device → 设备/驱动问题,`lsusb | grep 2bc5` + 重新插拔,查 4.2
- 能出但 valid_px 骤降(如某帧 1 千)→ IR AGC 收敛/光照,预热 10-20 帧再判

### Step 2:启动 astra_camera_node 看日志
```bash
source /opt/ros/humble/setup.bash
source ~/openni2/OpenNI-Linux-Arm64-2.3.0.66/OpenNIDevEnvironment
python3 arm_grasp/astra_camera_node.py
```
**期望日志**(顺序):
```
[INFO] Astra RGB 摄像头已打开 (640x480)          ← CAP_V4L2 生效(Bug1 检查点)
[INFO] OPENNI2_REDIST -> .../Redist              ← redist 探测(Bug 4.2)
[INFO] OpenNI2 SDK 内参 fx=945.03 ... 或 使用参数内参 fx=945.03 ...   ← 内参(Bug3 检查点)
[INFO] 深度后端: openni2 (liborbbec)             ← 后端选定
```
- 启动即 TypeError → 又引入 logger positional args(查 4.4)
- isOpened False → CAP_V4L2 丢了(查 4.3)
- 内参打的是 570.34 → 内参链路断了(查 4.5)
- 后端是 none → openni 分支失败,回 Step 1 隔离

### Step 3:验证 ROS 话题
```bash
# 深度(应在 23Hz 左右,数据有变化)
ros2 topic hz /rgbd_cam/depth/image_raw
ros2 topic echo --once /rgbd_cam/depth/image_raw --qos-reliability best_effort | head -12

# 内参(应 k[0]=945.028 k[2]=320 k[4]=945.028 k[5]=400)
ros2 topic echo --once /rgbd_cam/color/camera_info | grep -E 'k:'
```
- 深度话题 0Hz / 无数据 → 后端没起来(回 Step 2)
- 内参 570.34 → 4.5 内参链路
- 全 0 → 4.8 采样/饱和问题(看 vision 层)

### Step 4:vision_node 全链路
```bash
ros2 run arm_grasp vision_node --ros-args -p config_path:=$(pwd)/arm_grasp/config/grasp_config.yaml &
ros2 topic pub --once /vision/detect_request std_msgs/msg/String "data: 'red'"
ros2 topic echo /vision/grasp_pose
```
- `grasp|...|z_src=depth` → 真深度链路通
- `grasp|...|z_src=fixed` → 深度不可用,走了固定台面兜底(检查 Step 1)
- `invalid_depth` → 该点深度 0 且多级窗口也无效(fixed_depth 未开)→ 查 4.8/光照

---

## 6. 验证命令集(速查)

```bash
# 设备与驱动
lsusb | grep 2bc5                                   # 应有 2 个 Orbbec 设备
python3 -c "import openni; print('openni ok')"      # python 绑定
echo $OPENNI2_REDIST                                # 环境变量

# 节点日志
python3 arm_grasp/astra_camera_node.py              # 前台跑,看后端选择/内参日志

# ROS 话题
ros2 topic hz /rgbd_cam/depth/image_raw
ros2 topic echo --once /rgbd_cam/color/camera_info | grep -E 'k:'

# 全链路
ros2 topic pub --once /vision/detect_request std_msgs/msg/String "data: 'red'"
ros2 topic echo /vision/grasp_pose
```

---

## 7. 常见问题对照表(症状 → 根因 → 处理)

| 症状 | 根因 | 处理 |
|---|---|---|
| pyorbbec `query_devices()`=0 | iSerial=0 SDK bug(4.1) | 别折腾 pyorbbec,走 openni |
| `initialize()` 报 OPENNI2_REDIST | 环境变量(4.2) | source OpenNIDevEnvironment 或设参数 |
| `open_any()` no device | 驱动/权限(4.2) | 确认 liborbbec.so、udev、重新插拔 |
| 节点启动 TypeError | logger positional args(4.4) | 转 f-string |
| isOpened()=False | 无 CAP_V4L2(4.3) | 加 cv2.CAP_V4L2 |
| camera_info 570.34 | get_camera_params 失败(4.5) | 校验+参数兜底 945/320/400 |
| camera_info cx=640 | SDK 返回异常值(4.5) | 合理性校验拦截,回退参数 |
| 深度全 0 | 目标太近(<30cm)或全饱和 | 目标放远 30cm+,查光照 |
| 某帧 valid 骤降 | IR AGC 收敛(4.8) | 预热 10-20 帧 |
| grasp_pose invalid_depth | 中心深度 0 + 多级窗口无效(4.8) | 开 fixed_depth(填实测 depth_m)或查光照 |
| x/y 整体系统偏差 | IR/RGB 内参不一致(4.7) | 棋盘格标 RGB 内参 / 调 openni_cx/cy |
| z 抖动/取到背景 | RGB-IR 视差未对齐(4.6) | 确认 registration;或改 fixed_depth |

---

## 8. 关键参数速查

| 位置 | 参数 | 值/说明 |
|---|---|---|
| grasp_config.yaml | `fixed_depth.enabled` | 深度全失效时兜底,默认 false;现场填实测 `depth_m` 后开 |
| grasp_config.yaml | `fixed_depth.depth_m` | 相机光心到台面距离(m),现场尺测 |
| config/guosai_final.yaml `arm:` | `depth_mode` | `auto`(pyorbbec→openni→uvc→realsense→none) |
| config/guosai_final.yaml `arm:` | `openni_fx/fy/cx/cy` | 内参兜底,**945.028/945.028/320/400(现场实测)** |
| config/guosai_final.yaml `arm:` | `openni_redist` | 空=自动探测 `~/openni2/OpenNI-Linux-*/Redist` |
| config/guosai_final.yaml `arm:` | `fake_depth_fallback` | **正式比赛必须 false** |
| vision_node 反投影 | `cam2arm`(grasp_config) | 相机→机械臂平移,现场标定 [0.255,-0.06,-0.55] |

---

## 9. 关键 commit 索引

| commit | 内容 |
|---|---|
| `35720d3` | 伪深度改多后端真实深度(初版) |
| `031ecd0` | fixed_depth 固定台面兜底 + realsense 转发后端 |
| `848c618` | **OpenNI2 后端**(openni 加入 auto 链) |
| `19bc351` | 修 CAP_V4L2 + 8 处 logger f-string |
| `5f87910` | 内参三层保障(校验+参数兜底) |
| `3cfe1ab` | 深度多级窗口采样 + payload z_src |
