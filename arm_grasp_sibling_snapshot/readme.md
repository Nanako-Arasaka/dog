# JetArm 直抓调试 — 问题排查与解决总结

> 日期: 2026-06-15 · 机械臂: Hiwonder JetArm 6DOF · 算力板: Yahboom Jetson Xavier NX

---

## 一、核心需求

从当前姿态（home）直接检测并抓取眼前的红色长条，不经过任何预移动。

- 红条位置：底座前方约 30cm，左右居中，高度约 20cm（距底座）
- 流程：视觉检测 → 直接去抓 → 拎起 → 悬空 5 秒保持（不松爪）
- 中途掉落自动重试（最多 3 次）

---

## 二、已解决的问题

### 1. 串口设备路径

**问题**：`/dev/rrc` 不存在，串口桥接启动失败。

**解决**：`/dev/ttyUSB1` 是实际设备，创建软链接：
```bash
sudo rm /dev/rrc
sudo ln -s /dev/ttyUSB1 /dev/rrc
```

---

### 2. 视觉节点无输出

**问题**：`/vision/grasp_pose` 话题从未发布。

**原因**：视觉节点需要先收到 `/vision/detect_request` 才会触发检测，或者相机话题未对齐。

**解决**：确认相机节点在运行，确认彩色图 `/rgbd_cam/color/image_rect_color` 和深度图 `/rgbd_cam/depth/image_raw` 都有数据，然后发检测请求：
```bash
ros2 topic pub --once /vision/detect_request std_msgs/msg/String "data: 'red'"
```

---

### 3. 深度无效

**问题**：视觉日志输出 `invalid_depth`，有 HSV 检测但深度采样为 0。

**原因**：Astra 深度相机有最小距离限制（约 30-40cm），物体太近时深度值为 0。

**解决**：红条放远到 30cm 以上。

---

### 4. 视觉坐标与实际位置偏差大

**问题**：视觉输出 `x=0.10`，实际 `x=0.50`，偏差 0.4m。

**原因**：`grasp_config.yaml` 中 `camera_to_arm` 参数未校准。

**解决**：用实测位置反推校准值：
```yaml
camera_to_arm:
  x: 0.255
  y: -0.146
  z: -0.385
```

---

### 5. 置信度过低

**问题**：视觉检测 `conf` 突然降到 0.06 或被拒。

**原因**：默认阈值 `min_confidence: 0.3`，公式 `conf = area / 8000`，面积小时不通过。

**解决**：`jetarm_grasp.launch.py` 中 `min_confidence` 降到 `0.1`。

---

### 6. 姿态偏置导致关节外翻

**问题**：`posture_bias` 中 `shoulder_back: 80` 和 `elbow_up: 40` 是比赛场景（50cm 高台远距离）设计的值。红条近距离时，这些偏置导致大臂后仰、小臂上翻。

**解决**：改为 0：
```yaml
posture_bias:
  shoulder_back: 0
  elbow_up: 0
```

---

### 7. L3 腕长参数导致前向距离削减

**问题**：IK 公式 `d = d_xy - self.L3`，`L3=0.08` 削减了 8cm 前向距离，红条在 30cm 处时有效距离只剩 22cm，导致 IK 算出的肘角超限。

**解决**：`self.L3 = 0.0`。

---

### 8. elbow 上限约束（核心：解决手臂朝天问题）

**问题**：IK 对近距离目标（30cm）算出 `elbow=738`（接近上限 800）。elbow 值越大小臂越朝上，导致小臂外翻、摄像头朝天。

**解决**：在 IK 函数中加 elbow 上限约束：
```python
elbow = min(elbow, 300)   # 限300，让小臂保持朝下
elbow = max(elbow, 200)   # 下限（关节限位）
```

> 调节方法：如果还是太高，继续降这个数，200 是 home 的最低位。

---

### 9. pre_grasp_offset 预抓取偏移

**问题**：`pre_grasp_offset: 0.10` 让手臂在物体上方 10cm 处悬停再下降，近距离时这 10cm 导致肘关节超限。

**解决**：改为 `0.0`：
```yaml
grasp_strategy:
  pre_grasp_offset: 0.0
```

---

### 10. 直抓流程架构

**问题**：原代码先移到一个"观察位"/"home"再检测，但移动后摄像机可能看不见红条。

**解决**：`task_manager_node.py` 中 `_cb_direct_grasp` 直接发视觉检测请求，不预先移动手臂：
```
收到直抓命令 → 直接视觉检测（手臂不动） → 获到位姿 → 直接去抓
失败 → 自动重试最多3次
```

**问题**：`_direct_grasp` 原流程包含转底座 90° + 松爪释放。

**解决**：简化流程为：开爪 → 靠近 → 下降 → 夹紧 → 拎起 → 悬空 5 秒保持（不转不松）。

---

## 三、关节方向速查表（用户实测修正版）

| ID | 关节 | Home | 方向 |
|----|------|------|------|
| 1 | 底座 | 512 | — |
| 2 | 大臂(肩) | 500 | **越小越往前倾** |
| 3 | 小臂(肘) | 200 | **越小越往下** |
| 4 | 手腕1 | 400 | **越小腕越往下低** |
| 5 | 手腕2 | 522 | — |
| 6 | 手腕3 | 500 | — |
| 10 | 夹爪 | 400 | 越小越闭合 |

---

## 四、待解决的问题

### 1. 左右位置（Y 轴）还需细调

当前位置大致对了，但左右方向还需要微调。可以通过以下方式校准：

- 调整 `grasp_config.yaml` 中 `camera_to_arm.y` 的值
- 当前值：`y: -0.146`
- 视觉 Y 偏左 → y 值减小（更负）
- 视觉 Y 偏右 → y 值增大

### 2. elbow 上限值可能需要微调

当前 `elbow = min(elbow, 300)`，如果小臂还是偏高可以继续降到 250 或 200。

### 3. ~~手腕1 的 IK 计算~~ ✅ 已解决

已修复：wrist1 现在基于**约束后的 elbow** 反算 `constrained_elbow_angle` 再计算，wrist2 固定为 522（home值）不随物体角度扭动，保持手腕水平。

### 4. 标定精度

`camera_to_arm` 是用单点反推的粗略值，要做精确抓取需要用九点标定法或 `easy_handeye` 进行完整手眼标定。

---

## 五、修改的文件清单

| 文件 | 改动摘要 |
|------|---------|
| `arm_grasp/arm_control_node.py` | L3=0, elbow≤300, wrist1用约束后肘角重算, wrist2固定522不扭, home关节值, `_direct_grasp` 简化 |
| `arm_grasp/task_manager_node.py` | 直抓不预移动, 失败自动重试3次 |
| `config/grasp_config.yaml` | camera_to_arm 校准, pre_grasp_offset=0, posture_bias=0, 夹爪张开加大到500, 限位[100,500] |
| `launch/jetarm_grasp.launch.py` | min_confidence=0.1 |

---

## 六、完整操作流程

```bash
# ===== 编译 =====
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
rm -rf build install log
colcon build
source install/setup.bash

# ===== 终端1: 串口桥接 =====
source ~/ros2_ws/install/setup.bash
ros2 run ros_robot_controller ros_robot_controller

# ===== 终端2: 相机 =====
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/arm_grasp/arm_grasp/astra_camera_node.py

# ===== 终端3: arm_grasp 全部节点 =====
source ~/ros2_ws/install/setup.bash
ros2 launch arm_grasp jetarm_grasp.launch.py

# ===== 终端4: 测试 =====
source ~/ros2_ws/install/setup.bash

# home 回位
ros2 topic pub --once /arm/command std_msgs/msg/String "data: 'home'"

# 验证视觉
ros2 topic pub --once /vision/detect_request std_msgs/msg/String "data: 'red'"
ros2 topic echo /vision/grasp_pose --once

# 直抓
ros2 topic pub /task/direct_grasp std_msgs/msg/String "data: 'red'"
```
