# Cone Avoidance 初版

本目录实现第一版锥桶避障控制模块。模块边界保持很窄：输入感知侧给出的
`ConeObstacle` 列表，输出保守的 Lite2 速度 JSON：`vx/vy/wz`。

## 当前能力

- 单锥桶：左侧锥桶右绕，右侧锥桶左绕。
- 双锥桶：横向距离大于 `1.1m` 时尝试走中间，否则走外侧。
- 急停：正前方锥桶过近、front depth 过近、感知超时、输入 NaN/None/inf 或越界。
- 状态机：`IDLE -> ENTER_OBSTACLE_AREA -> TRACK_AND_AVOID -> RECOVER_STOP -> EXIT_OBSTACLE_AREA -> DONE`。
- UDP 输出：默认发到 `127.0.0.1:5005`，兼容 `lite2_motion_receiver.py`。
- Mock 测试：覆盖空场、单锥桶、双锥桶、近距离、超时和异常输入。

## 实机部署关系

实机运行时，RealSense D435i、YOLO 感知程序、`main_avoidance_run.py` 和
`lite2_motion_receiver.py` 都运行在 Jetson 上。Jetson 算力板和机器狗运动主机
之间通过网线连接，不走 Wi-Fi。

默认 `main_avoidance_run.py --receiver-ip 127.0.0.1 --receiver-port 5005`
表示把避障速度 JSON 发给 Jetson 本机上的 `lite2_motion_receiver.py`。这里的
`127.0.0.1` 不是机器狗 IP，只是 Jetson 本机 receiver 地址。

真正发给机器狗运动主机的是 `lite2_motion_receiver.py`，它通过 Jetson 的有线网口
把控制命令发送到 `--robot-ip/--robot-port`。只有当 `lite2_motion_receiver.py`
不在 Jetson 本机运行时，才需要把 `main_avoidance_run.py --receiver-ip` 改成
receiver 所在设备的真实局域网 IP。

## 调试

从 `controller` 目录运行：

```bash
python3 lite2_motion_receiver.py --listen-port 5005 --dry-run
```

另一个终端：

```bash
python3 -m cone_avoidance.mock_perception_test
```

也可以从标准输入喂 JSON 行：

```bash
echo '{"obstacles":[{"x":0.35,"z":1.0,"conf":0.9}]}' | \
  python3 -m cone_avoidance.main_avoidance_run --dry-run
```

正式向 receiver 发 UDP：

```bash
python3 -m cone_avoidance.main_avoidance_run --receiver-ip 127.0.0.1 --receiver-port 5005
```

这条命令只表示连接 Jetson 本机 receiver。机器狗运动主机 IP 由
`lite2_motion_receiver.py --robot-ip` 指定。

输入 JSON 行格式：

```json
{"obstacles":[{"x":0.25,"z":1.10,"conf":0.86,"bbox":[320,180,430,410],"age":5}],"front_depth":1.4}
```

输出示例：

```json
{"source":"cone_avoidance","reason":"avoid_left_cone","vx":0.08,"vy":0.0,"wz":-0.25,"state":"TRACK_AND_AVOID"}
```

## 待实机确认

- 感知输出的 `x > 0` 是否确认为左侧。
- `wz > 0` 在 Lite2 上是否确认为左转；如相反，用 receiver 的 `--invert-turn`。
- `vx > 0` 是否确认为前进；如相反，用 receiver 的 `--invert-forward`。
- Jetson 有线网口 IP、机器狗运动主机 IP/port，以及两者是否能互相通信。
- receiver 实机端口、超时停止行为和速度量纲。
- 避障完成后给后续巡检模块的 handoff 信号格式。
