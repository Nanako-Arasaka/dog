# 避障策略与控制负责人任务说明

本文档交给避障策略与控制负责人。第一阶段目标是把感知模块输出的 `ConeObstacle` 列表转换成保守、低速、可解释的机器狗速度指令，并通过已有 Lite2 receiver 的 UDP JSON 接口发送。该模块不做相机接入和锥桶 3D 定位。

## 我负责什么

- 设计第一版局部反应式避障策略。
- 设计安全刹车和感知掉线停止逻辑。
- 编写避障状态机。
- 把 `ConeObstacle` 列表转换为 `vx/vy/wz`。
- 通过 UDP JSON 接入 `lite2_motion_receiver.py`。
- 使用 mock 感知数据做离线测试。
- 在 dry-run receiver 下验证 JSON 输出。
- 做低速实机验证，优先不碰撞。

## 我不负责什么

- 不负责 Intel RealSense D435i 相机驱动。
- 不负责 RGB/Depth 读取。
- 不负责 CameraInfo/内参处理。
- 不负责 YOLO 训练。
- 不负责 bbox + depth 定位。
- 不负责 SLAM 建图。
- 第一版不做目标点导航，不依赖 `/camera_pose`。

## 项目背景

本项目面向 2026 年中国高校智能机器人创意大赛足式机器人挑战赛四足大型组。当前只做线下挑战中的避障任务：障碍区域随机放置 2 个锥形桶，规格约 67 x 32 x 32 cm，机器狗需要自主绕开并通过障碍区域。

比赛不允许使用激光雷达，避障模块以 RGB-D 相机感知为核心。第一版不是完整导航系统，而是局部反应式模块：感知同学输出锥桶相对机器狗的 `x/z`，控制同学输出低速绕行动作。

## 与感知负责人的接口

从感知负责人接收 `ConeObstacle` 列表：

```yaml
ConeObstacle:
  x: float                 # 左右偏移，单位 m
  z: float                 # 前方距离，单位 m
  conf: float
  bbox: [x1, y1, x2, y2]
  age: int
  last_seen: time
```

坐标约定：

- `z` 表示前方距离，单位 m。
- `x` 表示左右偏移，单位 m。
- 默认 `x > 0` 为左侧，`x < 0` 为右侧。
- 如果感知模块实际输出相反，集成时必须统一修正。

控制模块需要容忍这些情况：

- 空列表：前方没有稳定锥桶，但仍要结合前方 depth 安全区判断。
- `conf` 偏低：可以降速或进入观察，不要高速通过。
- `last_seen` 超时：立即停止或进入 `RECOVER_STOP`。
- `x/z` 为 NaN、None、inf 或超过合理范围：立即停止。
- 感知模块报告 aligned depth 不可用、有效深度比例过低或 RealSense 帧率低于阈值：进入 `RECOVER_STOP`。

## 输出到 Lite2 receiver 的接口

向 `lite2_motion_receiver.py` 发送 UDP JSON，默认端口按题述为 `5005`，需在实机工程确认。

实机部署关系必须写清楚：

- RealSense D435i、YOLO 感知程序、`main_avoidance_run.py`、`lite2_motion_receiver.py` 都运行在 Jetson 上。
- `main_avoidance_run.py --receiver-ip 127.0.0.1` 表示发送到 Jetson 本机的 `lite2_motion_receiver.py`。
- `127.0.0.1` 不是机器狗 IP，只是 Jetson 本机 receiver 地址。
- `lite2_motion_receiver.py` 再通过 Jetson 的有线网口，把控制命令发给机器狗运动主机。
- 只有 receiver 和 avoidance 不在同一台机器上时，才把 `--receiver-ip` 改成 receiver 所在设备的真实局域网 IP。

推荐 JSON：

```json
{"source":"cone_avoidance","reason":"clear_forward","vx":0.12,"vy":0.0,"wz":0.0}
```

也可兼容 ROS 风格：

```json
{"linear":{"x":0.12,"y":0.0},"angular":{"z":0.25}}
```

字段含义：

- `vx`：前进速度，单位按 receiver 约定，第一版按 m/s 理解。
- `vy`：横移速度，第一版可固定为 `0.0`。
- `wz`：偏航角速度，单位按 receiver 约定，第一版按 rad/s 理解。
- `source`：固定 `"cone_avoidance"`，方便日志区分。
- `reason`：当前决策原因，方便调试。

必须复用已有代码思路：

- 阅读 `lite2_motion_receiver.py`，确认 UDP 监听端口和 JSON 格式。
- 阅读 `goal_controller.py`，参考其速度 JSON 发送方式。
- 第一版避障不需要依赖 `/camera_pose`。
- 第一版避障不做目标点导航，只输出 `vx/vy/wz`。

当前 checkout 未看到 `lite2_motion_receiver.py` 和 `goal_controller.py`，需要从主工程补齐或在实机目录确认。本文档先按题述接口约定编写，实际端口、方向符号、超时行为都标为需实机确认。

## 推荐参数

第一版必须保守，优先不碰撞，不追求速度。

```yaml
normal_speed: 0.15
slow_speed: 0.08
max_turn_speed: 0.25
safe_radius: 0.55
slow_distance: 1.20
stop_distance: 0.55
front_emergency_width: 0.45
front_emergency_distance: 0.50
perception_timeout: 0.5
send_rate_hz: 10-20
```

注意：

- D435i 官方最小工作距离约 0.3m，但考虑机器狗惯性、步态摆动、YOLO/Depth 延迟，第一版 `stop_distance` 建议仍从 0.5m-0.6m 开始，不建议一开始压到 0.3m。
- YOLO 和 depth 都有延迟，机器狗速度要低。
- UDP 发包频率不能太低，否则 receiver 可能 timeout 停止。
- UDP 发包频率也不要高到难以调试，建议 10-20Hz。

## 基础决策逻辑

第一版只使用前进和转向，`vy = 0.0`。如果后续确认 Lite2 横移稳定，再加入小幅 `vy`。

1. 没有锥桶，且前方 depth 安全：

```text
vx = normal_speed
wz = 0
reason = "clear_forward"
```

2. 左前方有锥桶：

```text
vx = slow_speed
wz = 右转方向
reason = "avoid_left_cone"
```

3. 右前方有锥桶：

```text
vx = slow_speed
wz = 左转方向
reason = "avoid_right_cone"
```

4. 正前方距离过近：

```text
vx = 0
vy = 0
wz = 0
reason = "emergency_stop"
```

5. 两个锥桶都在前方：

- 判断中间通道是否足够。
- 如果横向间距足够，走中间。
- 如果横向间距不足，选择外侧更宽的一边绕行。

两个锥桶通道判断建议：

```text
gap = abs(cone1.x - cone2.x) - 2 * cone_safe_radius
```

如果 `gap > robot_width + margin`，可以尝试走中间。第一版可直接使用更简单的阈值：

- 两个锥桶横向距离 `> 1.1m`：尝试走中间。
- 两个锥桶横向距离 `<= 1.1m`：不要走中间，走外侧。

## 紧急停止要求

必须实现这些停止条件：

- 如果正前方 depth 小于阈值，立即 `vx=0, vy=0, wz=0`。
- 如果连续 `perception_timeout` 时间没有收到感知数据，立即停止。
- 如果检测结果突然丢失但上一帧锥桶很近，先停止再重新判断。
- 如果输出速度出现 NaN、None、inf 或超过限幅，立即停止。
- 如果感知模块报告 aligned depth 不可用或有效深度比例过低，进入 `RECOVER_STOP`。
- 如果 RealSense 掉线或帧率低于阈值，立即停止。
- 如果进入 `RECOVER_STOP` 状态，先停止 0.3-0.5s，再小角度转向重新观察。

这里的“正前方 depth”可以由感知负责人额外提供安全区结果，或在控制模块订阅一个简化的前方最近距离。第一版不要只依赖 YOLO，depth 安全区也要参与刹车。控制模块不要假设 depth 永远可靠，低有效像素比例应按不确定危险处理。

## 状态机建议

```text
IDLE
  -> ENTER_OBSTACLE_AREA
  -> TRACK_AND_AVOID
  -> RECOVER_STOP
  -> EXIT_OBSTACLE_AREA
  -> DONE
```

状态说明：

- `IDLE`：不发运动，或持续发停止。
- `ENTER_OBSTACLE_AREA`：低速前进，开始接收感知结果。
- `TRACK_AND_AVOID`：根据 `ConeObstacle` 输出绕行动作。
- `RECOVER_STOP`：当前方太近、感知丢失、判断不确定时进入；先停止，再重新观察。
- `EXIT_OBSTACLE_AREA`：认为已经绕过锥桶后，低速直行离开障碍区。
- `DONE`：停止避障模块，交接给后续巡检模块。

通过障碍区域的判断，第一版用时间 + 前方安全：

- 进入避障模式后运行固定时间，例如 6-10 秒。
- 且连续 2 秒前方无锥桶、depth 安全。
- 满足以上条件则认为通过障碍区域。

第二版可接入 `/camera_pose` 或里程计判断前进距离：

- 从避障开始累计前进约 1.5-2.0m。
- 且前方安全。
- 满足以上条件则认为通过障碍区域。

## 方向符号必须实测

内部可以先采用 ROS 习惯：`wz > 0` 表示左转。但 Lite2 receiver 的实际转向符号必须在 dry-run 和低速实机中确认。

需要记录：

- `vx > 0` 是否确认为前进。
- `wz > 0` 实机是否确认为左转。
- 如果方向相反，使用 receiver 的 `--invert-turn` 或在 `motion_sender.py` 中统一转换。
- 最终 README 或调试记录中必须写清楚已确认的方向符号。

## 实机网络检查

实机时 Jetson 和机器狗运动主机通过网线连接，不走 Wi-Fi。上机说明必须包含：

- 确认 Jetson 有线网口 IP，例如用 `ip addr` 和 `ip route`。
- 确认 Jetson 能 ping 通机器狗运动主机 IP。
- 确认 `lite2_motion_receiver.py --robot-ip/--robot-port` 是机器狗运动主机地址。
- 确认防火墙或网络策略不会拦 UDP。
- 确认控制包没有被发到 Wi-Fi 网卡或错误网段。
- dry-run 可以只测试 Jetson 本机 `127.0.0.1:5005`。
- 实机必须先启动 `lite2_motion_receiver.py`，再启动 `main_avoidance_run.py`。
- 实机前必须确认 `vx > 0` 的前进方向和 `wz` 的左右转符号。

## 调试命令

先 dry-run receiver，不让机器狗动：

```bash
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

另一个终端发送测试 UDP JSON。可以用项目现有 sender，也可以临时用 Python/网络工具发送：

```json
{"vx":0.1,"vy":0.0,"wz":0.0}
{"vx":0.0,"vy":0.0,"wz":0.2}
{"vx":0.0,"vy":0.0,"wz":0.0}
```

需要确认 receiver 日志能看到：

- JSON 被正确解析。
- 端口确认为 `5005`。
- `vx/vy/wz` 映射到预期运动。
- dry-run 不会驱动机器狗。

## Mock 测试要求

请先做 mock，再接实机感知。至少覆盖：

- mock 没有锥桶。
- mock 左前方一个锥桶。
- mock 右前方一个锥桶。
- mock 正前方近距离锥桶。
- mock 两个锥桶中间可通过。
- mock 两个锥桶中间不可通过。
- mock 感知数据超时。
- mock 输入 NaN、None、inf 或异常值。

每个 mock case 应输出一条可读日志：

```text
state=TRACK_AND_AVOID reason=avoid_left_cone vx=0.08 vy=0.00 wz=-0.20
```

如果触发急停，应明确输出：

```text
state=RECOVER_STOP reason=emergency_stop vx=0.00 vy=0.00 wz=0.00
```

## 推荐文件结构

只推荐结构，不要求本次一次性实现全部文件：

```text
cone_avoidance/
  avoidance_policy.py
  safety_guard.py
  motion_sender.py
  avoidance_state_machine.py
  mock_perception_test.py
  main_avoidance_debug.py
  main_avoidance_run.py
  config/
    control.yaml
```

## 推荐开发步骤

1. 阅读 `cone_avoidance/README.md`，理解已有避障初步方案。
2. 从主工程补齐并阅读 `lite2_motion_receiver.py` 和 `goal_controller.py`，确认 UDP 端口、JSON 格式、方向符号。
3. 定义 `ConeObstacle` 输入结构和 `VelocityCommand` 输出结构。
4. 写 `avoidance_policy.py`，先用 mock case 输出 `vx/vy/wz`。
5. 写 `safety_guard.py`，统一做超时、NaN、限幅、近距离急停。
6. 写 `motion_sender.py`，只负责 UDP JSON 发送。
7. 写 `avoidance_state_machine.py`，实现 `IDLE` 到 `DONE` 的状态转换。
8. dry-run receiver 验证 JSON。
9. 与感知负责人联调 `ConeObstacle` 列表。
10. 低速实机测试，先设置保守速度和较大的 stop distance。

## 验收标准

- 不接机器狗时，能根据 mock `ConeObstacle` 输出合理 `vx/wz`。
- dry-run 下能看到 receiver 收到正确 JSON。
- 速度输出有上限，不会突然给大速度。
- 单锥桶场景能低速绕开。
- 双锥桶场景能选择中间或外侧。
- 近距离危险时能停止。
- 感知掉线时能停止。
- 感知输出异常时能停止。
- 实机测试时，机器人速度保守，不连续冲撞。
- 避障完成后能输出 `DONE` 或 handoff 信号给后续巡检模块。

## 当前实机管道命令

实机时先启动 `lite2_motion_receiver.py`，再启动感知 + 避障管道。

终端 1，Jetson 本机 receiver，把速度命令通过有线网口发给机器狗运动主机：

```bash
cd /path/to/国赛/controller

python3 lite2_motion_receiver.py \
  --listen-port 5005 \
  --robot-ip <robot_motion_host_ip> \
  --robot-port <robot_motion_host_port> \
  --timeout 0.8
```

终端 2，RealSense + YOLO + aligned depth 感知，并把控制 JSONL 管道交给避障控制：

```bash
cd /path/to/国赛

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

浏览器查看感知画面：

```text
http://<jetson-ip>:8080/
```

这里的 `--receiver-ip 127.0.0.1` 只是 Jetson 本机 `lite2_motion_receiver.py`，不是机器狗 IP。机器狗运动主机 IP 只写在终端 1 的 `--robot-ip`。

## 常见坑

- 不能每帧左右摇摆，要加滞回或滤波。
- 不能速度太快，主动立体 depth 和 YOLO 都有延迟。
- 不能只依赖 YOLO，正前方 depth 安全区也要参与刹车。
- D435i 官方最小工作距离约 0.3m，但第一版不要把 `stop_distance` 一开始压到 0.3m。
- 如果感知模块报告 aligned depth 不可用或有效深度比例过低，控制模块应进入 `RECOVER_STOP`。
- 如果 RealSense 掉线或帧率低于阈值，控制模块应停止。
- D435i 的 IMU 第一版不参与避障控制，除非后续单独设计姿态补偿。
- UDP 发包频率不能太低，否则 receiver 可能 timeout 停止。
- UDP 发包频率也不能太高导致调试困难，建议 10-20Hz。
- 不要在文档或日志中承诺已经完成实机验证，除非代码和测试确实完成。
- 当前 checkout 未包含 Lite2 receiver 文件，不能假设端口、方向符号和 dry-run 参数已经在本目录验证过。

## 最终集成接口约定

第一阶段控制模块必须可单独调试。最终集成时，输入来自感知模块，输出只走 receiver 的速度接口：

```yaml
control_input:
  source: "rgbd_cone_perception"
  timestamp: time
  obstacles:
    - x: 0.25
      z: 1.10
      conf: 0.86
      bbox: [320, 180, 430, 410]
      age: 5
      last_seen: time

control_output:
  source: "cone_avoidance"
  reason: "avoid_left_cone"
  vx: 0.08
  vy: 0.0
  wz: -0.20
  state: "TRACK_AND_AVOID"
```

集成前必须确认：

- 感知输出的 `x` 正方向。
- `z` 单位是否为 m。
- `last_seen` 时间基准。
- receiver UDP IP 和端口。
- `wz` 正负方向。
- receiver 超时停止行为。
- 避障完成后给巡检模块的 `DONE` 或 handoff 信号格式。
