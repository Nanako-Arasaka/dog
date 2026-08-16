#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS cone avoidance node — LocalPlanner (global-path guided + depth).

正式链路（方案甲）：cone_avoidance_node → LocalPlanner → /motion/avoid_cmd → motion_mux
- 订阅 /camera_pose（机器人位姿）、aligned depth + camera_info（锥桶 3D 位置）
- 加载 competition_map.yaml（global_path 锚点 + obstacle_zone_rect 边界）
- 输出 /motion/avoid_cmd（Twist），走 motion_mux 仲裁（watchdog 急停生效）

LocalPlanner 特性（对应障碍区设计）：
- 沿 global_path（障碍区 起始/中间/末尾 锚点）逐点引导前进
- 实时绕开随机锥桶（候选指令评分含锥桶最小距离约束 <0.36m 丢弃）
- obstacle_zone_rect 边界约束（不超地图边界）
- 深度安全停（前向过近 / 深度有效比例过低 / RealSense 掉帧）
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_avoidance.cone_detector_yolo import ConeYoloDetector  # noqa: E402
from cone_avoidance.local_planner import LocalPlanner, RobotPose  # noqa: E402
from cone_avoidance.map_config import load_map_config  # noqa: E402
from cone_avoidance.models import ConeObstacle, ControlConfig  # noqa: E402


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_control_config(path_text: str) -> ControlConfig:
    """读 cone_avoidance/config/control.yaml 构造 ControlConfig。"""
    path = Path(path_text)
    values: dict[str, Any] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            values[key.strip()] = _parse_scalar(raw_value)
    allowed = {field.name for field in fields(ControlConfig)}
    filtered = {key: value for key, value in values.items() if key in allowed}
    return ControlConfig(**filtered)


def heading_from_quaternion(qx: float, qy: float, qz: float, qw: float,
                            ground_plane: str = "xz", forward_axis: str = "z") -> float:
    """前向向量投影提取地面朝向 (与 waypoint_navigator 同约定)。

    ORB 世界系 y 竖直、地面 x-z: z-euler yaw 对绕 y 轴转身退化 (恒 ≈0/π),
    必须用前向向量投影 atan2(f_z, f_x)。
    """
    if forward_axis == "x":
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fz = 2.0 * (qx * qz - qw * qy)
    else:
        fx = 2.0 * (qx * qz + qw * qy)
        fz = 1.0 - 2.0 * (qx * qx + qy * qy)
    if ground_plane == "xz":
        return math.atan2(fz, fx)
    return math.atan2(2.0 * (qy * qz - qw * qx), fx)


class ConeAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("cone_avoidance_node")
        self.declare_parameter("model", "/home/jetson/yolo_deploy/cone_best.pt")
        self.declare_parameter("camera", "0")
        self.declare_parameter("conf", 0.45)
        self.declare_parameter("send_hz", 10.0)
        self.declare_parameter("enabled_topic", "/motion/enable_cone_avoidance")
        self.declare_parameter("cmd_topic", "/motion/avoid_cmd")
        self.declare_parameter("status_topic", "/cone_avoidance/status")
        # --- LocalPlanner 新增参数 ---
        self.declare_parameter("control_yaml", str(ROOT / "cone_avoidance" / "config" / "control.yaml"))
        self.declare_parameter("map_config", str(ROOT / "cone_avoidance" / "competition_map.yaml"))
        self.declare_parameter("pose_topic", "/camera_pose")
        # 朝向提取约定 (与 waypoint_navigator / watchdog 一致)
        self.declare_parameter("ground_plane", "xz")
        self.declare_parameter("forward_axis", "z")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("depth_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")

        self.enabled = False
        self.cap = None
        self.detector = ConeYoloDetector(str(self.get_parameter("model").value), conf=float(self.get_parameter("conf").value))

        # --- LocalPlanner ---
        config = load_control_config(str(self.get_parameter("control_yaml").value))
        global_path, obstacle_zone_rect = load_map_config(str(self.get_parameter("map_config").value))
        self.planner = LocalPlanner(
            config=config,
            global_path=global_path,
            obstacle_zone_rect=obstacle_zone_rect,
        )
        self.get_logger().info(
            f"LocalPlanner ready: path={len(global_path)} pts rect="
            f"[{obstacle_zone_rect.xmin},{obstacle_zone_rect.xmax}]x"
            f"[{obstacle_zone_rect.ymin},{obstacle_zone_rect.ymax}]"
        )

        # --- 状态 ---
        self.latest_pose: Optional[RobotPose] = None
        self.latest_depth: Optional[np.ndarray] = None  # float32 米
        self.depth_info: Optional[tuple[float, float, float, float]] = None  # fx fy cx cy
        self.last_depth_time = 0.0
        self.depth_times: deque = deque(maxlen=30)

        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(Bool, str(self.get_parameter("enabled_topic").value), self._on_enabled, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("pose_topic").value), self._on_pose, 10)
        self.create_subscription(Image, str(self.get_parameter("depth_topic").value), self._on_depth, 10)
        self.create_subscription(CameraInfo, str(self.get_parameter("depth_info_topic").value), self._on_depth_info, 10)
        self.create_timer(1.0 / max(1.0, float(self.get_parameter("send_hz").value)), self._tick)
        self.get_logger().info("cone avoidance wrapper ready (LocalPlanner)")

    # ------------------------------------------------------------------ subs
    def _on_enabled(self, msg: Bool) -> None:
        enabled = bool(msg.data)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.get_logger().info(f"enabled={self.enabled}")
        if self.enabled:
            self._ensure_camera()
        else:
            self._publish_stop()

    def _on_pose(self, msg: PoseStamped) -> None:
        p = msg.pose
        # planner 2D 平面 = ORB 世界 (x, z): y 是竖直轴不能进平面, 前进方向是 z
        self.latest_pose = RobotPose(
            x=float(p.position.x),
            y=float(p.position.z),
            yaw=heading_from_quaternion(
                float(p.orientation.x), float(p.orientation.y),
                float(p.orientation.z), float(p.orientation.w),
                str(self.get_parameter("ground_plane").value).lower(),
                str(self.get_parameter("forward_axis").value).lower(),
            ),
        )

    def _on_depth(self, msg: Image) -> None:
        try:
            if msg.encoding in ("16UC1", "mono16"):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.latest_depth = arr.astype(np.float32) / 1000.0  # mm → m
            elif msg.encoding == "32FC1":
                self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            else:
                return
        except Exception:  # noqa: BLE001
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.depth_times.append(now)
        self.last_depth_time = now

    def _on_depth_info(self, msg: CameraInfo) -> None:
        k = msg.k
        self.depth_info = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    # ------------------------------------------------------------------ camera
    def _ensure_camera(self) -> None:
        if self.cap is not None:
            return
        import cv2

        camera_value = self.get_parameter("camera").value
        try:
            camera = int(camera_value)
        except (TypeError, ValueError):
            camera = str(camera_value)
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            self.status_pub.publish(String(data=f"camera_open_failed:{camera}"))
            self.get_logger().error(f"failed to open camera: {camera}")

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        if not self.enabled:
            return
        self._ensure_camera()
        if self.cap is None or not self.cap.isOpened():
            self._publish_stop()
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status_pub.publish(String(data="read_failed"))
            self._publish_stop()
            time.sleep(0.05)
            return

        detections = self.detector.detect(frame)
        cones = self._to_cones(detections)
        front_depth, depth_valid_ratio = self._front_depth_metrics()
        now = self.get_clock().now().nanoseconds * 1e-9
        fps = self._depth_fps()
        aligned_ok = self.latest_depth is not None and (now - self.last_depth_time) <= 1.0
        depth_ok = aligned_ok and self.depth_info is not None

        command = self.planner.plan(
            cones=cones,
            robot_pose=self.latest_pose,
            front_depth=front_depth,
            depth_valid_ratio=depth_valid_ratio,
            aligned_depth_ok=aligned_ok,
            realsense_fps=fps,
            realsense_ok=depth_ok,
        )
        msg = Twist()
        msg.linear.x = float(command.vx)
        msg.linear.y = float(command.vy)
        msg.angular.z = float(command.wz)
        self.cmd_pub.publish(msg)
        self.status_pub.publish(
            String(data=f"{command.state}:{command.reason}|cones={len(cones)}|depth={front_depth if front_depth is not None else -1:.2f}")
        )

    # ------------------------------------------------------------------ helpers
    def _to_cones(self, detections) -> list[ConeObstacle]:
        """YOLO bbox + aligned depth → ConeObstacle(x/z 米)。"""
        if self.depth_info is None or self.latest_depth is None:
            return []
        fx, fy, cx, cy = self.depth_info
        cones: list[ConeObstacle] = []
        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det.xyxy]
            u = (x1 + x2) * 0.5
            v = (y1 + y2) * 0.5
            z = self._roi_depth_median(int(x1), int(y1), int(x2), int(y2))
            if z is None or z <= 0.0:
                continue
            x = -(u - cx) * z / fx  # 图像右为正 → 控制约定左为正，取反
            cones.append(ConeObstacle(x=x, z=z, conf=float(det.confidence), bbox=(x1, y1, x2, y2)))
        return cones

    def _roi_depth_median(self, x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
        if self.latest_depth is None:
            return None
        d = self.latest_depth
        h, w = d.shape
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = d[y1:y2, x1:x2]
        valid = roi[(roi > 0.05) & (roi < 5.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _front_depth_metrics(self) -> tuple[Optional[float], Optional[float]]:
        """中央 ROI（宽 20% × 高 60%）的 (最近障碍深度, 有效深度比例)。"""
        if self.latest_depth is None:
            return None, None
        d = self.latest_depth
        h, w = d.shape
        roi = d[int(h * 0.2):int(h * 0.8), int(w * 0.4):int(w * 0.6)]
        if roi.size == 0:
            return None, None
        valid = roi[(roi > 0.05) & (roi < 5.0)]
        ratio = float(valid.size / roi.size)
        front = float(np.min(valid)) if valid.size else None
        return front, ratio

    def _depth_fps(self) -> Optional[float]:
        if len(self.depth_times) < 2:
            return None
        span = self.depth_times[-1] - self.depth_times[0]
        if span <= 0:
            return None
        return float((len(self.depth_times) - 1) / span)

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())


def main() -> None:
    rclpy.init()
    node = ConeAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
