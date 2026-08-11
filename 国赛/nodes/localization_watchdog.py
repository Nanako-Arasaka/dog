#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch /camera_pose and force a motion stop on pose loss or jumps."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


class LocalizationWatchdog(Node):
    def __init__(self) -> None:
        super().__init__("localization_watchdog")
        self.declare_parameter("pose_topic", "/camera_pose")
        self.declare_parameter("pose_type", "pose_stamped")
        self.declare_parameter("ok_topic", "/localization/ok")
        self.declare_parameter("status_topic", "/localization/status")
        self.declare_parameter("stop_topic", "/motion/stop")
        self.declare_parameter("stable_samples", 15)
        self.declare_parameter("stable_max_position_step", 0.08)
        self.declare_parameter("stable_max_yaw_step", 0.35)
        self.declare_parameter("pose_timeout_sec", 0.8)
        self.declare_parameter("jump_position_threshold", 0.45)
        self.declare_parameter("jump_yaw_threshold", 1.2)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.pose_type = str(self.get_parameter("pose_type").value).lower()
        self.pose_timeout_sec = float(self.get_parameter("pose_timeout_sec").value)
        self.jump_position_threshold = float(self.get_parameter("jump_position_threshold").value)
        self.jump_yaw_threshold = float(self.get_parameter("jump_yaw_threshold").value)
        self.stable_samples = int(self.get_parameter("stable_samples").value)
        self.stable_max_position_step = float(self.get_parameter("stable_max_position_step").value)
        self.stable_max_yaw_step = float(self.get_parameter("stable_max_yaw_step").value)

        self.ok_pub = self.create_publisher(Bool, str(self.get_parameter("ok_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.stop_pub = self.create_publisher(Bool, str(self.get_parameter("stop_topic").value), 10)
        self.samples: deque[Pose2D] = deque(maxlen=max(2, self.stable_samples))
        self.last_pose: Pose2D | None = None
        self.last_pose_time = None
        self.ok = False
        self.last_reason = "waiting_for_pose"

        if self.pose_type in ("pose_stamped", "posestamped", "pose"):
            self.create_subscription(PoseStamped, self.pose_topic, self._on_pose_stamped, 10)
        elif self.pose_type in ("odometry", "odom"):
            self.create_subscription(Odometry, self.pose_topic, self._on_odom, 10)
        else:
            raise ValueError("pose_type must be pose_stamped or odometry")
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f"watching {self.pose_topic} ({self.pose_type})")

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        self._update_pose(msg.pose)

    def _on_odom(self, msg: Odometry) -> None:
        self._update_pose(msg.pose.pose)

    def _update_pose(self, pose: Pose) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        current = Pose2D(float(pose.position.x), float(pose.position.y), yaw_from_pose(pose))
        if self.last_pose is not None:
            dist = math.hypot(current.x - self.last_pose.x, current.y - self.last_pose.y)
            dyaw = abs(angle_diff(current.yaw, self.last_pose.yaw))
            if dist > self.jump_position_threshold or dyaw > self.jump_yaw_threshold:
                self._set_fault(f"pose_jump dist={dist:.3f} dyaw={dyaw:.3f}")
                self.samples.clear()
            else:
                self.samples.append(current)
        else:
            self.samples.append(current)
        self.last_pose = current
        self.last_pose_time = now

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_pose_time is None:
            self._set_fault("waiting_for_pose")
            return
        age = now - self.last_pose_time
        if age > self.pose_timeout_sec:
            self._set_fault(f"pose_timeout age={age:.2f}s")
            return
        if not self._is_stable():
            self._publish_ok(False, "stabilizing_pose")
            return
        self._publish_ok(True, "stable")

    def _is_stable(self) -> bool:
        if len(self.samples) < self.samples.maxlen:
            return False
        prev = None
        for sample in self.samples:
            if prev is not None:
                dist = math.hypot(sample.x - prev.x, sample.y - prev.y)
                dyaw = abs(angle_diff(sample.yaw, prev.yaw))
                if dist > self.stable_max_position_step or dyaw > self.stable_max_yaw_step:
                    return False
            prev = sample
        return True

    def _set_fault(self, reason: str) -> None:
        self.stop_pub.publish(Bool(data=True))
        self._publish_ok(False, reason)

    def _publish_ok(self, ok: bool, reason: str) -> None:
        if ok != self.ok or reason != self.last_reason:
            level = self.get_logger().info if ok else self.get_logger().warn
            level(f"localization ok={ok}: {reason}")
        self.ok = ok
        self.last_reason = reason
        self.ok_pub.publish(Bool(data=ok))
        self.status_pub.publish(String(data=reason))


def main() -> None:
    rclpy.init()
    node = LocalizationWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_pub.publish(Bool(data=True))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
