# Hiwonder JetArm 跑通流程

> 本机(Jetson)上把 Hiwonder **JetArm** 机械臂从零驱动起来的完整记录与使用说明。
> 完成日期:2026-06-14 · 串口 `/dev/ttyUSB0` · 实测可用。

---

## 1. 硬件与架构

```
Jetson ──USB──> [控制板上的 USB HUB] ──> CH340(/dev/ttyUSB0) ──> STM32 控制板 ──> 总线舵机 ×5(ID 1~5)
```

- 机械臂:Hiwonder **JetArm**(35KG 总线舵机)。
- 控制板:Hiwonder **ros_robot_controller**(STM32F407),负责把上位机指令转发给总线舵机。
- 连接:控制板通过板载 **CH340**(USB 转串口)挂在 Jetson 上。
  - `lsusb` 特征:`1a86:7523 CH340` 挂在 `1a86:8091 QinHeng USB HUB` 之后 → 这是 ros_robot_controller 板的典型标志。

> ⚠️ 本机原本**没有**装 JetArm 的官方 SDK(只装了 Yahboom 的语音大模型工作区 `yahboom_ws`)。
> 控制代码是从 Hiwonder 官方 GitHub 仓库拉取的 `ros_robot_controller_sdk.py`。

---

## 2. 关键参数(实测确认)

| 项目 | 值 | 说明 |
|---|---|---|
| 串口设备 | `/dev/ttyUSB0` | CH340,驱动 `ch34x` 已加载 |
| **波特率** | **`1000000`(1 Mbps)** | ⚠️ 不是 115200!115200 是「STM32↔舵机」内部那段的速率,**Jetson↔STM32 是 1Mbps** |
| 协议 | Hiwonder ros_robot_controller | 帧头 `0xAA 0x55` + func + len + data + CRC8 |
| 有效舵机 ID | **1, 2, 3, 4, 5** | ID 6 无响应(本型号即 5 个总线舵机) |
| 位置范围 | `0 ~ 1000` | 对应 0~240°,**中位 500** |
| 运动时间单位 | 秒(SDK 内部转 ms) | `bus_servo_set_position(时间秒, ...)` |
| Python | **`/usr/bin/python3`** | 已装 pyserial 3.5;⚠️ 默认的 conda python **没装** pyserial |
| 端口权限 | `crwxrwxrwx`(全权限) | 无需 sudo 即可访问 |

实测开机静止姿态(参考):`[1]=512  [2]=761  [3]≈0  [4]=320  [5]=522`

---

## 3. 跑通流程(复盘)

1. **认设备**:`lsusb` / `ls /dev/ttyUSB*` → 确认 CH340 在 `/dev/ttyUSB0`,`ch34x` 驱动已加载。
2. **定协议**:查 Hiwonder 官方 wiki + GitHub,确认是 ros_robot_controller 板(帧头 `0xAA 0x55`),Jetson↔板波特率 **1Mbps**。
3. **拿 SDK**:从 `Hiwonder/LanderPi` 仓库下载官方 `ros_robot_controller_sdk.py`(含 `Board` 类、CRC8、各类舵机/外设命令)。
4. **装依赖**:用 `/usr/bin/python3`(已自带 pyserial 3.5),无需额外安装。
5. **先诊断不动臂**(安全):
   - 发蜂鸣器指令 → 确认 TX + 波特率正确;
   - 读舵机 1~5 当前位置 → 确认通信与当前姿态。
6. **再运动**:在**当前姿态附近**对称小幅摆动并回位(`jetarm_move.py`)。
7. **放大幅度**:`--amp` 调大(实测做过 ±150 ≈ ±36°,全程回原位无异常)。

---

## 4. 文件清单(都在 `~/`)

| 文件 | 作用 |
|---|---|
| `~/ros_robot_controller_sdk.py` | Hiwonder 官方控制板 SDK(`Board` 类)。已把 `buf_write` 里调试用的 `print(buf)` 注释掉 |
| `~/jetarm_move.py` | 自写的安全驱动脚本:蜂鸣确认 → 读位置 → 当前姿态附近摆动并回位 |
| `~/JetArm_跑通流程.md` | 本文档 |

---

## 5. 协议速查

帧格式(小端):

```
0xAA 0x55 | func(1) | data_len(1) | data(N) | crc8(1)
                                              └─ CRC8 校验范围 = func + data_len + data
```

`func`(功能码)常用:

| 值 | 功能 |
|---|---|
| 2 | 蜂鸣器 BUZZER |
| 4 | PWM 舵机 |
| **5** | **总线舵机 BUS_SERVO** |
| 1 / 11 | LED / RGB |
| 7 | IMU |

总线舵机(func=5)常用子命令(data 第 1 字节):

| 子命令 | 含义 | SDK 方法 |
|---|---|---|
| `0x01` | 设置位置(可多舵机) | `bus_servo_set_position(t, [[id,pos],...])` |
| `0x03` | 停止 | `bus_servo_stop([id,...])` |
| `0x05` | 读位置 | `bus_servo_read_position(id)` |
| `0x0B/0x0C` | 上力/松力 | `bus_servo_enable_torque(id, True/False)` |
| `0x10` | 改 ID | `bus_servo_set_id(old, new)` |
| `0x20/0x24` | 设/存偏差 | `bus_servo_set_offset` / `save_offset` |

---

## 6. 使用方法

```bash
cd ~

# 只诊断(读各舵机位置,不动臂)—— 排查问题时先跑这个
/usr/bin/python3 jetarm_move.py --no-move

# 默认:蜂鸣 + 读位置 + 在当前姿态附近 ±9.6° 轻摆并回位
/usr/bin/python3 jetarm_move.py

# 大幅度:±150≈±36°,每段 2 秒
/usr/bin/python3 jetarm_move.py --amp 150 --dur 2.0
```

> ⚠️ 务必用 `/usr/bin/python3`(有 pyserial);conda 的 python 会报 `No module named 'serial'`。

---

## 7. 自己写动作(最小示例)

```python
import time
from ros_robot_controller_sdk import Board

board = Board(device="/dev/ttyUSB0", baudrate=1000000, timeout=2)
board.enable_reception(True)
time.sleep(0.4)

# 读当前位置(动大动作前建议先读,避免从极限位置大幅甩动)
for i in range(1, 6):
    print(i, board.bus_servo_read_position(i))

# 运动:1.5 秒内运动到目标(位置 0~1000,务必限幅在该区间)
board.bus_servo_set_position(1.5, [[1, 500], [2, 500], [3, 500], [4, 500], [5, 500]])
time.sleep(1.6)

# 蜂鸣提示
board.set_buzzer(1900, 0.1, 0.05, 1)

# 松力(让某个舵机变软)——断电/松力前请托住机械臂,防止下坠
# board.bus_servo_enable_torque(2, False)
```

**注意**:`bus_servo_set_position` 的位置用无符号 16 位打包,**目标值必须 0~1000**,负数会 `struct.error`。
(舵机读位置可能返回负数,表示物理上略过零点;写之前要自行限幅。)

---

## 8. 排错

| 现象 | 排查 |
|---|---|
| `No module named 'serial'` | 用 `/usr/bin/python3`,不要用 conda python |
| 打不开串口 / 设备不存在 | `ls -l /dev/ttyUSB*`;`lsusb \| grep CH340`;`dmesg \| grep ch34` |
| 所有舵机「无响应」,但蜂鸣器响 | Jetson↔控制板通信 OK,问题在舵机:**机械臂没上电 / 电源开关没开 / 舵机线没插好** |
| 蜂鸣器也不响 | 波特率/端口不对,或串口被别的程序占用(关掉占用 `/dev/ttyUSB0` 的进程) |
| `struct.error: ushort ...` | 写入的位置超出 0~1000(含负数),需限幅 |
| 动作乱/抖 | 时间给太短;把 `--dur` 调大、慢一点 |

---

## 9. 安全注意

- **大幅度动作前**:先 `--no-move` 读各关节角度,清空机械臂周围(尤其正前方/下方)的障碍物。
- **抬升/下压关节(肩、肘)**是碰撞风险最高的;横向的底座旋转最安全。
- 让目标位置**始终限幅在 0~1000**,并尽量从已知稳定姿态出发、对称摆动、最后回原位。
- 运动结束后舵机默认**保持上电锁定**;若要松力(`enable_torque(id, False)`)或断电,**先托住机械臂**,防止重力下坠。

---

## 10. 参考

- JetArm 产品页:https://www.hiwonder.com/products/jetarm
- JetArm 基础控制 wiki:https://wiki.hiwonder.com/projects/JetArm/en/jetarm-jetson-nano/docs/2.ROS1_Robot_Arm_Basic_Control_User_Manual.html
- 官方 SDK 源码:https://github.com/Hiwonder/LanderPi/blob/main/src/driver/ros_robot_controller/ros_robot_controller/ros_robot_controller_sdk.py
- ROS 控制板(STM32F407):https://www.hiwonder.com/products/ros-robot-control
