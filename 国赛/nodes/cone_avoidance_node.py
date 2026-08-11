#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS wrapper for the existing cone detector and avoidance strategy."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import Bool, String

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obstacle_avoidance.cone_detector_yolo import ConeYoloDetector  # noqa: E402
from obstacle_avoidance.cone_strategy import AvoidanceConfig, plan_cone_avoidance  # noqa: E402


class ConeAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("cone_avoidance_node")
        self.declare_parameter("model", "/home/jetson/yolo_deploy/cone_best.pt")
        self.declare_parameter(
            "camera",
            "0",
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("send_hz", 8.0)
        self.declare_parameter("enabled_topic", "/motion/enable_cone_avoidance")
        self.declare_parameter("cmd_topic", "/motion/avoid_cmd")
        self.declare_parameter("status_topic", "/cone_avoidance/status")

        self.enabled = False
        self.cap = None
        self.detector = ConeYoloDetector(str(self.get_parameter("model").value), conf=float(self.get_parameter("conf").value))
        self.config = AvoidanceConfig(min_confidence=float(self.get_parameter("conf").value))
        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(Bool, str(self.get_parameter("enabled_topic").value), self._on_enabled, 10)
        self.create_timer(1.0 / max(1.0, float(self.get_parameter("send_hz").value)), self._tick)
        self.get_logger().info("cone avoidance wrapper ready")

    def destroy_node(self) -> bool:
        self._publish_stop()
        if self.cap is not None:
            self.cap.release()
        return super().destroy_node()

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
        decision = plan_cone_avoidance(detections, frame.shape, self.config)
        msg = Twist()
        msg.linear.x = float(decision.vx)
        msg.linear.y = float(decision.vy)
        msg.angular.z = float(decision.wz)
        self.cmd_pub.publish(msg)
        self.status_pub.publish(String(data=f"{decision.state}:{decision.reason}"))

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
