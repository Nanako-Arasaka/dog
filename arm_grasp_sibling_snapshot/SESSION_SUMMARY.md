# JetArm 直抓 — 今日调试总结

> 日期: 2026-06-16 · 机械臂: Hiwonder JetArm 6DOF · 算力板: Yahboom Jetson Xavier NX

---

## 一、今日完成的工作

### 1. 串口桥接修复
- **问题**: Jetson 上 `ros_robot_controller` 包是旧版，硬编码 `/dev/rrc` 串口，且可执行文件名为 `ros_robot_controller`
- **修复**: 
  - 创建 `serial_bridge_node.py`，支持 `device` 参数（默认 `/dev/ttyUSB0`）
  - 更新 `setup.py` 入口点为 `serial_bridge_node`
  - 创建 launch 文件 `ros_robot_controller.launch.py`
  - 终端1命令改为: `ros2 launch ros_robot_controller ros_robot_controller.launch.py device:=/dev/ttyXXX`
- **传输文件**: `ros_robot_controller/setup.py`, `ros_robot_controller/ros_robot_controller/serial_bridge_node.py`, `ros_robot_controller/launch/ros_robot_controller.launch.py`, `ros_robot_controller_sdk.py`(放到 `~/`)

### 2. 直抓流程简化（去掉闭环居中）
- **原因**: 底座旋转公式 `base_target = 512 + (cx-320)*0.5` 不准，反复转底座反而越转越偏
- **改为**: 检测到物体 → 不转底座 → 直接调肩肘腕去抓
- `_direct_grasp` 中 `base_target = 512`（保持底座不动）

### 3. 抓取验证（视觉二次确认）
- **方案**: z轴判断法——只比较抓取前后物体z坐标的变化
- `Δz > 0.03m` → 物体被拎起 → 成功
- `Δz ≤ 0.03m` → 物体还在桌上 → 空抓 → 重试
- **原因**: xy坐标有视觉噪声，z轴判断更可靠
- `_handle_vision_fail` 在 VERIFY 状态时检测不到物体 → 判定成功（物体被夹爪遮挡）

### 4. 失败重试机制
- 空抓后定向转底座：物体偏右→右转，偏左→左转
- 每次固定转 **10°**（`RETRY_DEGREE = 100`，100px ≈ 50舵机单位 ≈ 10°）
- 最多重试 **10次**（10×10°=100°后回home放弃）
- 回退机制：连续**4次**视觉检测失败 → 回退底座到最近成功位置
- 回退最多**3次**，超限放弃

### 5. 防重复指令
- `_cb_direct_grasp` 加保护：已在直抓流程中时忽略重复指令

### 6. 底座位置持久化
- **关键修复**: `_direct_grasp` 之前硬编码 `base_target = 512`，每次抓取都把底座拉回home，`center_base` 的旋转白做了
- **修复**: `_send_center_base` 计算 `_desired_base`，`_do_grasp` 通过命令参数传给 `_direct_grasp`
- 命令格式增加第9个字段: `direct_grasp|x|y|z|angle|dur|cx|cy|base`

### 7. 边缘微调（仅首次检测）
- 物体离左右边 < 80px 时，小幅转底座
- **重要**: 边缘微调只在首次检测（`_grasp_retries == 0`）时触发，重试时跳过
- **修复的bug**: 之前重试时边缘微调也会触发，导致"刚转对又反着转"

### 8. 手腕抬高 + 肘补偿
- 腕1基准值: 530（下限470）
- 物体5cm高，腕不能降太低剐蹭；若需更低则抬肘补偿
- `wrist1_target < 480` 时自动抬肘: `elbow_target += int((480 - wrist1_target) * 0.6)`

### 9. 夹取后大幅度抬高手臂
- 夹紧后先抬肩回500、肘350、腕520 → 再回home
- 防止爪子剐蹭平台

### 10. 抓取高度降低
- 肩关节: `shoulder_target = 500 - int(x * 700) - 25`
- 多前倾25个单位（约5°），爪子降低1-2cm

---

## 二、当前未解决问题

### ★ 底座旋转角度应自适应
- **现状**: 重试时固定转10°，不根据物体在画面中的实际位置调整
- **需要改进**: 根据物体在画面中的像素位置（cx）动态计算旋转角度
- 物体离中心越远 → 需要转的角度越大
- 物体离中心越近 → 转的角度越小
- 不能再固定死10度

---

## 三、当前参数速查

| 参数 | 值 | 说明 |
|------|----|------|
| `EDGE_MARGIN` | 80px | 边缘微调触发阈值 |
| `EDGE_ADJUST` | 60px | 首次边缘微调量 |
| `RETRY_DEGREE` | 100px | 重试底座旋转量（≈10°） |
| `GRASP_Z_TOL` | 0.03m | z轴变化判断阈值 |
| `MAX_GRASP_RETRIES` | 10 | 最多重试次数 |
| 回退触发 | 连续4次视觉失败 | |
| 回退上限 | 3次 | |

### 关节抓取参数

| 参数 | 值 |
|------|-----|
| 腕1基准 | 530 |
| 腕1下限 | 470 |
| 肘上限 | 600 |
| 肩下限 | 200 |
| 肩前倾偏移 | -25（压低抓取高度） |

---

## 四、五个终端命令

```bash
# 终端1: 串口（先 ls /dev/ttyUSB* 确认端口）
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch ros_robot_controller ros_robot_controller.launch.py device:=/dev/ttyUSB0

# 终端2: 相机
source /opt/ros/humble/setup.bash
python3 ~/ros2_ws/src/arm_grasp/arm_grasp/astra_camera_node.py

# 终端3: 主程序
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch arm_grasp jetarm_grasp.launch.py

# 终端4: 监控视觉
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 topic echo /vision/grasp_pose

# 终端5: 控制
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 topic pub --once /arm/command std_msgs/msg/String "data: 'home'"
ros2 topic pub --once /vision/detect_request std_msgs/msg/String "data: 'red'"
ros2 topic pub /task/direct_grasp std_msgs/msg/String "data: 'red'"
```

---

## 五、修改的文件清单

| 文件 | 关键改动 |
|------|---------|
| `arm_grasp/task_manager_node.py` | 状态机(DETECT/CENTER/GRASP/VERIFY)、z轴验证、重试定向旋转、回退机制、防重复指令、边缘微调仅在首次 |
| `arm_grasp/arm_control_node.py` | `center_base`命令、`_direct_grasp`使用传入base_target、抬腕+肘补偿、夹后大幅抬臂、压低抓取高度 |
| `ros_robot_controller/setup.py` | 入口点改为serial_bridge_node |
| `ros_robot_controller/ros_robot_controller/serial_bridge_node.py` | 新建，支持device参数 |
| `ros_robot_controller/launch/ros_robot_controller.launch.py` | 新建launch文件 |

---

## 六、给明天AI的要点

1. **当前最大的待解决问题**: 底座旋转角度不能固定死10度，需要根据物体在画面中的像素位置动态计算
2. **已验证可用的功能**: z轴判断法抓取验证、回退机制、边缘微调
3. **容易踩的坑**: 
   - `_direct_grasp` 不能硬编码 base_target=512，必须用传入的 `_desired_base`
   - 边缘微调只在首次检测触发，重试时必须跳过
   - 底座旋转方向：物体偏右→adj_cx变小，偏左→adj_cx变大
   - `center_base` 公式: `base_target = 512 + (cx - 320) × 0.5`
4. **视觉坐标系**: 640×480, 中央 cx=320, cy=240
5. **关节方向**: 肩越小越前倾, 肘越小越往下, 腕越小越往下低





cd ~/ros2_ws
  colcon build
  source ~/ros2_ws/install/setup.bash

  终端1 — 串口：
  ls /dev/ttyUSB*
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  ros2 launch ros_robot_controller ros_robot_controller.launch.py
  device:=/dev/ttyUSB0

  终端2 — 相机：
  source /opt/ros/humble/setup.bash
  python3 ~/ros2_ws/src/arm_grasp/arm_grasp/astra_camera_node.py

  终端3 — 主程序：
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  ros2 launch arm_grasp jetarm_grasp.launch.py

  终端4 — 监控：
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  ros2 topic echo /vision/grasp_pose

  终端5 — 控制：
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  ros2 topic pub --once /arm/command std_msgs/msg/String "data: 'home'"
  ros2 topic pub /task/direct_grasp std_msgs/msg/String "data: 'red'"
