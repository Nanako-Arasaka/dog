# lite2_navigation_bridge

This ROS2 package contains a small goal controller for the Jueying Lite2 flow:

```text
camera -> ORB_SLAM3 ROS2 pose -> goal_controller -> UDP JSON -> lite2_motion_receiver.py -> Lite2 motion host
```

It does not modify ORB_SLAM3. It subscribes to the pose output from your SLAM node, computes a simple go-to-goal command, and sends JSON velocity packets to the dog-side receiver.

## Build

From `colcon_ws`:

```bash
colcon build --packages-select lite2_navigation_bridge
source install/setup.bash
```

## Run

Example for a `PoseStamped` pose topic:

```bash
ros2 run lite2_navigation_bridge goal_controller --ros-args \
  -p pose_topic:=/orbslam3/pose \
  -p pose_type:=pose_stamped \
  -p target_x:=2.0 \
  -p target_y:=0.0 \
  -p receiver_ip:=192.168.1.103 \
  -p receiver_port:=5005
```

Example for an `Odometry` pose topic:

```bash
ros2 run lite2_navigation_bridge goal_controller --ros-args \
  -p pose_topic:=/odom \
  -p pose_type:=odometry \
  -p target_x:=2.0 \
  -p target_y:=0.0
```

You can also send a new target at runtime:

```bash
ros2 topic pub --once /lite2/goal geometry_msgs/msg/PoseStamped \
  "{pose: {position: {x: 2.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}}"
```

Start with `lite2_motion_receiver.py --dry-run` before allowing real robot motion.
