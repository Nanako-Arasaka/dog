# Cone Avoidance 上机测试说明

本目录是第一版锥桶避障控制模块。上机时建议把整个目录和同级的
`lite2_motion_receiver.py` 放在同一个 `controller` 目录下：

```text
controller/
  lite2_motion_receiver.py
  cone_avoidance/
    avoidance_policy.py
    avoidance_state_machine.py
    safety_guard.py
    motion_sender.py
    main_avoidance_run.py
    mock_perception_test.py
    models.py
    config/control.yaml
    README.md
    ON_ROBOT_TEST.md
```

## 测试前需要准备

### 硬件

- Lite2 机器狗，电量充足。
- Jetson/工控机，通过有线网口和 Lite2 运动主机连接，并且在同一有线网段。
- RGB-D 相机和感知模块，如果还没接感知，先用 mock 测试。
- 空旷测试场地，前方至少 2m 安全空间。
- 急停手段：遥控器、断电方式或能立刻 Ctrl+C 停 receiver。

### 软件

- Python 3.8+。
- 当前目录包含：
  - `lite2_motion_receiver.py`
  - `cone_avoidance/`
- 不需要 ROS 才能跑第一版避障 mock 和 UDP 发送。
- 如果要接真实感知，需要感知模块能输出 JSON 行：

```json
{"obstacles":[{"x":0.25,"z":1.10,"conf":0.86,"bbox":[320,180,430,410],"age":5}],"front_depth":1.4}
```

字段约定：

- `x > 0` 暂按锥桶在左侧。
- `z` 单位必须是 m。
- `front_depth` 是正前方安全区最近距离，单位 m；没有时可以先不传，但实机建议接入。

## 实机部署拓扑

实机默认所有上游程序都在 Jetson 上运行：

```text
RealSense D435i
  -> YOLO + aligned depth 感知程序
  -> main_avoidance_run.py
  -> 127.0.0.1:5005
  -> lite2_motion_receiver.py
  -> Jetson 有线网口
  -> 机器狗运动主机 --robot-ip:--robot-port
```

注意：

- `main_avoidance_run.py --receiver-ip 127.0.0.1` 是因为 `lite2_motion_receiver.py` 也在 Jetson 本机运行。
- `127.0.0.1` 不是机器狗 IP，不要把它理解成运动主机地址。
- 机器狗运动主机地址由 `lite2_motion_receiver.py --robot-ip` 和 `--robot-port` 指定。
- 只有当 receiver 和 avoidance 不在同一台机器上时，才把 `--receiver-ip` 改成 receiver 所在设备的真实局域网 IP。

## 实机网络检查

先确认 Jetson 使用的是有线网口，不要把控制包发到 Wi-Fi 网卡或错误网段：

```bash
ip addr
ip route
```

记录 Jetson 有线网口 IP，例如 `eth0` 或 `enP...` 对应的地址。

确认能从 Jetson ping 通机器狗运动主机 IP：

```bash
ping <robot_motion_host_ip>
```

确认 `lite2_motion_receiver.py` 的目标地址就是机器狗运动主机：

```bash
python3 lite2_motion_receiver.py \
  --listen-port 5005 \
  --robot-ip <robot_motion_host_ip> \
  --robot-port <robot_motion_host_port> \
  --dry-run
```

检查防火墙和网络策略不会拦 UDP。Jetson 本机的 `127.0.0.1:5005` 只用于
`main_avoidance_run.py -> lite2_motion_receiver.py`；真正到机器狗的是
`lite2_motion_receiver.py -> <robot_motion_host_ip>:<robot_motion_host_port>`。

## 第 1 步：只跑 mock，不启动狗

在 Jetson/电脑上进入 `controller` 目录：

```bash
cd /path/to/controller
python3 -m cone_avoidance.mock_perception_test
```

看到最后一行：

```text
all mock cases passed
```

说明策略和状态机能正常运行。

## 第 2 步：启动 dry-run receiver

此步骤不会驱动机器狗，只打印 Lite2 命令。

dry-run 阶段可以只测试 Jetson 本机 `127.0.0.1:5005`，不要求实际发到机器狗。

终端 1：

```bash
cd /path/to/controller
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

终端 2：用避障模块发一条测试数据。

```bash
cd /path/to/controller
echo '{"obstacles":[{"x":0.35,"z":1.0,"conf":0.9}],"front_depth":1.4}' | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

预期：

- 终端 2 打印类似：

```text
state=TRACK_AND_AVOID reason=avoid_left_cone vx=0.08 vy=0.00 wz=-0.09
```

- 终端 1 能看到收到 payload，并解析成 Lite2 `MotionCommand`。

## 第 3 步：确认方向符号

仍然用 dry-run 或抬腿/悬空低风险方式确认，不要一上来放地上跑。

测试前进：

```bash
echo '{"obstacles":[],"front_depth":2.0}' | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

需要确认：

- `vx > 0` 是否确实是前进。
- `wz > 0` 是否确实是左转。
- 如果转向相反，启动 receiver 时加：

```bash
--invert-turn
```

如果前后相反，加：

```bash
--invert-forward
```

## 第 4 步：低速实机 receiver

确认 dry-run 正常后，再启动正式 receiver。先保持保守速度，不要调大。
实机时必须先启动 `lite2_motion_receiver.py`，再启动 `main_avoidance_run.py`。

终端 1：

```bash
cd /path/to/controller
python3 lite2_motion_receiver.py \
  --listen-port 5005 \
  --robot-ip 192.168.1.120 \
  --robot-port 43893 \
  --timeout 0.8 \
  --default-speed 9000 \
  --turn-speed 20000
```

如方向相反，按第 3 步追加 `--invert-forward` 或 `--invert-turn`。
这里的 `--robot-ip` 必须是机器狗运动主机的有线网地址，不是 `127.0.0.1`。

## 第 5 步：人工喂数据小步测试

终端 2 先不要接真实感知，手动发几条数据观察狗的动作。
这些命令仍然发到 Jetson 本机 receiver，所以 `--receiver-ip` 保持 `127.0.0.1`。

空场低速前进：

```bash
echo '{"obstacles":[],"front_depth":2.0}' | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

左前方锥桶，应该低速右绕：

```bash
echo '{"obstacles":[{"x":0.35,"z":1.0,"conf":0.9}],"front_depth":1.4}' | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

右前方锥桶，应该低速左绕：

```bash
echo '{"obstacles":[{"x":-0.35,"z":1.0,"conf":0.9}],"front_depth":1.4}' | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

正前方太近，应该停止：

```bash
echo '{"obstacles":[{"x":0.0,"z":0.55,"conf":0.9}],"front_depth":0.55}' | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

## 第 6 步：接真实感知

感知模块每帧输出一行 JSON，管道接给避障模块：

```bash
python3 your_cone_perception.py | \
  python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

启动顺序必须是：

1. 确认 Jetson 有线网口和机器狗运动主机连通。
2. 启动 `lite2_motion_receiver.py`，确认目标 `--robot-ip/--robot-port` 是机器狗运动主机。
3. 再启动 YOLO 感知程序和 `main_avoidance_run.py`。

建议频率：

- 感知输出 10-20Hz。
- 如果感知低于 10Hz，receiver 可能因超时停狗。
- 如果连续 0.5s 没有新感知，避障模块会进入 `RECOVER_STOP`。

## 第 7 步：现场锥桶测试顺序

按这个顺序来，不要跳：

1. 空场：确认低速直行和停止正常。
2. 单个左前锥桶：确认右绕。
3. 单个右前锥桶：确认左绕。
4. 正前近距离：确认急停。
5. 两锥桶间距大于 1.1m：确认尝试走中间。
6. 两锥桶间距小于等于 1.1m：确认走外侧。

每一步都先跑 1-2 秒，确认无异常再延长。

## 必须记录

- `vx > 0` 的实机方向。
- `wz > 0` 的实机方向。
- Jetson 有线网口 IP。
- receiver 使用的真实 `robot-ip` 和 `robot-port`。
- `main_avoidance_run.py --receiver-ip` 是否为 `127.0.0.1`，以及 receiver 是否确实在 Jetson 本机。
- 感知输出的 `x` 正方向。
- `front_depth` 是否可靠，以及缺失时如何处理。
- 哪些场景已经实机验证，哪些只是 dry-run。

## 出问题先看这里

- 狗不动：确认 receiver 终端是否收到 UDP payload。
- receiver 收到但狗不动：检查 `robot-ip`、`robot-port`、狗端网络、运动模式和 Jetson 有线网口。
- receiver 完全收不到：确认 `main_avoidance_run.py --receiver-ip` 指向 receiver 所在设备；同机运行时应为 `127.0.0.1`。
- ping 不通机器狗：检查网线、有线网口 IP、路由和是否误走 Wi-Fi 网段。
- 左右绕反：优先确认感知 `x` 正方向，再考虑 receiver 的 `--invert-turn`。
- 每帧左右摇：降低感知抖动，或调小 `turn_smoothing_alpha`。
- 近距离不刹车：确认传入了 `front_depth`，且单位是 m。
