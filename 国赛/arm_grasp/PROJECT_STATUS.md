# JetArm 直抓项目 — 当前状态

> 日期: 2026-06-15 · 机械臂: Hiwonder JetArm 6DOF · 算力板: Yahboom Jetson Xavier NX

---

## 一、任务目标

机械臂从 home 姿态直接检测并抓取红色长条，不经过预移动。
- 红条位置：底座前方约 22cm，左右居中
- 流程：Home → 收到直抓指令 → **不动，先检测** → 拿到坐标 → 直接去抓 → 拎起 → 悬空 5 秒

---

## 二、硬件参数（实测）

| 参数 | 值 | 说明 |
|------|----|------|
| L1 大臂 | 0.18m | 肩关节→肘关节 |
| L2 小臂 | 0.16m | 肘关节→腕关节 |
| L3 腕长 | 0.18m | 腕关节→爪尖 |
| shoulder_h | 0.06m | 肩关节距底座底部 |
| 平台高度 | 0.22m | 高台上表面距底座（测试时设 0） |

### 关节方向速查表

| ID | 关节 | Home | 方向 |
|----|------|------|------|
| 1 | 底座 | 512 | — |
| 2 | 大臂(肩) | 500 | **越小越往前倾** |
| 3 | 小臂(肘) | 200 | **越小越往下** |
| 4 | 手腕1 | 350 | 越小腕越往下低 |
| 5 | 手腕2 | 522 | — |
| 6 | 手腕3 | 500 | — |
| 10 | 夹爪 | — | 抓取=小值(100), 张开=大值(500) |

---

## 三、文件结构

```
~/ros2_ws/src/arm_grasp/
├── arm_grasp/
│   ├── arm_control_node.py      ← IK + 关节控制 + _direct_grasp
│   ├── task_manager_node.py     ← 任务调度 + _cb_direct_grasp
│   ├── vision_node.py           ← HSV颜色检测 + 坐标变换
│   ├── inspection_memory_node.py
│   └── visualization_node.py
├── config/
│   └── grasp_config.yaml        ← 所有参数
├── launch/
│   └── jetarm_grasp.launch.py
└── astra_camera_node.py         ← Astra深度相机驱动
```

---

## 四、坐标链路

```
相机拍到红条
  ↓
vision_node: HSV检测 → 像素(cx,cy) → 深度采样
  ↓
_to_arm_frame():
  z_cam = depth_mm / 1000
  x_cam = (cx-u0) * z_cam / fx
  y_cam = (cy-v0) * z_cam / fy
  p_arm = [x_cam, y_cam, z_cam] + cam2arm   ← ★ 关键变换
  gz_grasp = p_arm_z + 0.075                 ← 加抓取高度
  ↓
发布 /vision/grasp_pose: "grasp|x|y|z|angle|conf"
  ↓
task_manager: 原样转发给 arm_control
  ↓
arm_control._ik(x,y,z) → 6个关节脉冲值
  ↓
舵机执行
```

### 当前 cam2arm 校准值
```yaml
camera_to_arm:
  x: 0.255
  y: -0.06
  z: -0.55      # ★ 刚改的，之前 -0.385 导致 z 偏高十几厘米
```

### 最新视觉输出（cam2arm.z=-0.55 后）
```
x=0.329 y=0.013 z=0.025 angle=93° conf=1.00
```
- x=32.9cm（实际约22cm，x 偏大约10cm，cam2arm.x 可能还需调小）
- y≈0（居中，OK）
- z=2.5cm（在肩关节下方，OK）

---

## 五、当前 IK 算法

**两遍求解**（`arm_control_node.py` 第 193-237 行）：

```
输入: (x,y,z) = 爪尖目标在基座坐标系

第一遍: 假设腕长=0
  d1 = min(d_xy, L1+L2-0.005)
  h1 = z - 0.06
  余弦定理 → 肩角s1, 肘角e1
  小臂方向 fa = s1 + e1

第二遍: 扣掉腕长18cm投影
  d2 = d_xy - 0.18×cos(fa)
  h2 = h1   - 0.18×sin(fa)
  余弦定理 → 肩角s2, 肘角e2

脉冲 = 500 + 弧度×500/π
限位: 从配置读取 [200, 850]（肘改成850，之前800卡太死）
wrist1 = 500 - (s2+e2)×500/π
wrist2 = 522 (固定不转)
wrist3 = 500 (固定)
```

---

## 六、直抓流程

### task_manager_node.py `_cb_direct_grasp`:
1. **有防重复保护**：如果 `self.direct_mode` 已为 True，忽略新指令
2. 设置 `self.direct_mode = True`
3. **不发移动指令**，只发 `pub_detect.publish()` 请求视觉检测
4. 收到视觉结果 → `_do_grasp()` → 发送 `direct_grasp|x|y|z|angle|dur` 给 arm_control

### arm_control_node.py `_direct_grasp`:
```
1. 开夹爪 (100 = 张开)
2. 移到物体上方 (z + pre_grasp_offset)
3. 下降到抓取点 (z + grasp_depth)
4. 夹紧 (500 = 闭合)
5. 拎起来 (z + lift_height)
6. 悬空保持 5 秒
```

---

## 七、当前问题

### 1. X 轴偏大
视觉报 x=0.329（32.9cm），实际约 22cm。**cam2arm.x 需要从 0.255 调小**。

### 2. 肘关节容易打满
目标较远（32.9cm）时肘算出来 840+，碰到限位。x 修正后会缓解。

### 3. 小臂方向
用户希望小臂朝下（肘小），但几何上远距离目标需要肘打开。cam2arm 修正后目标距离缩短，肘自然降低。

### 4. 标定精度
cam2arm 是单点反推的粗略值，需要多点精确标定。

---

## 八、操作流程

```bash
# ===== 编译 =====
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
rm -rf build install log
colcon build
source install/setup.bash

# ===== 终端1: 串口桥接 =====
ros2 run ros_robot_controller ros_robot_controller

# ===== 终端2: 相机 =====
python3 ~/ros2_ws/src/arm_grasp/arm_grasp/astra_camera_node.py

# ===== 终端3: arm_grasp 全部节点 =====
ros2 launch arm_grasp jetarm_grasp.launch.py

# ===== 终端4: 测试 =====
source ~/ros2_ws/install/setup.bash

# home 回位
ros2 topic pub --once /arm/command std_msgs/msg/String "data: 'home'"

# 验证视觉 (看z是不是<0.1)
ros2 topic pub --once /vision/detect_request std_msgs/msg/String "data: 'red'"
ros2 topic echo /vision/grasp_pose --once

# 直抓 (只发一次!)
ros2 topic pub /task/direct_grasp std_msgs/msg/String "data: 'red'"
```

---

## 九、已修改文件清单

| 文件 | 关键改动 |
|------|---------|
| `config/grasp_config.yaml` | cam2arm.z=-0.55, 夹爪 open=100/close=500, Y=-0.06, elbow上限850, 平台=0 |
| `arm_grasp/arm_control_node.py` | L1=0.18 L2=0.16 L3=0.18 shoulder_h=0.06, 两遍IK, 限位读配置, wrist2=522 |
| `arm_grasp/task_manager_node.py` | 防重复指令保护 |
