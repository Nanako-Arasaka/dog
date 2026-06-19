# Integration Bridge

这一层只负责状态和数据转发，不做 YOLO、OpenCV、导航规划或机械臂动作。

## 职责

- 接收巡检识别结果，统一成 `/inspection/all` 需要的格式。
- 接收放置区字母识别结果，统一成 `/placement/recognized_zone`。
- 写入 JSONL 事件日志，便于赛前排查模块间数据是否通。
- 在没有 ROS2 的 Windows 环境下支持 `--no-ros` 单次格式验证。

## ROS2 Topic

输入：

```text
/bridge/inspection_result  std_msgs/String
/bridge/placement_zone     std_msgs/String
```

输出：

```text
/inspection/all            std_msgs/String
/placement/recognized_zone std_msgs/String
```

`/inspection/all` 示例：

```text
A:abnormal,B:normal,C:unknown,D:unknown
```

`/placement/recognized_zone` 示例：

```text
A
```

## Jetson 上运行

```bash
cd /home/jetson/yolo_deploy
python3 -m integration_bridge.bridge_node
```

如果没有安装成 Python 包，也可以直接运行：

```bash
python3 integration_bridge/bridge_node.py
```

## 发布巡检结果

实时巡检程序 `live_detect_yolo_opencv.py` 在 ROS2 可用时会自动发布到本桥接层输入 topic。以下命令主要用于手动联调。

单个区域 JSON：

```bash
ros2 topic pub --once /bridge/inspection_result std_msgs/msg/String \
  "data: '{\"zone\":\"A\",\"gauge_status\":\"high\",\"abnormal\":true}'"
```

整组结果：

```bash
ros2 topic pub --once /bridge/inspection_result std_msgs/msg/String \
  "data: 'A:abnormal,B:normal,C:abnormal,D:normal'"
```

桥接层会转发为：

```text
/inspection/all <- A:abnormal,B:normal,C:abnormal,D:normal
```

## 发布放置区识别结果

```bash
ros2 topic pub --once /bridge/placement_zone std_msgs/msg/String "data: 'A'"
```

或者：

```bash
ros2 topic pub --once /bridge/placement_zone std_msgs/msg/String \
  "data: '{\"zone\":\"zone_A\",\"confidence\":0.92}'"
```

桥接层会转发为：

```text
/placement/recognized_zone <- A
```

## 本地无 ROS2 验证

```powershell
python .\integration_bridge\bridge_node.py --no-ros --inspection-json "A:abnormal,B:normal,C:unknown,D:unknown"
python .\integration_bridge\bridge_node.py --no-ros --placement-zone "zone_A"
```

默认日志：

```text
output/integration_bridge/events.jsonl
```
