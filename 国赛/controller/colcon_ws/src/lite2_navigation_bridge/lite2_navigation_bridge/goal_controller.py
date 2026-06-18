#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float


@dataclass
class Goal:
    x: float
    y: float
    yaw: float


class Lite2GoalController(Node):
    def __init__(self) -> None:
        super().__init__("lite2_goal_controller")

        self.declare_parameter("pose_topic", "/orbslam3/pose")
        self.declare_parameter("pose_type", "pose_stamped")
        self.declare_parameter("goal_topic", "/lite2/goal")
        self.declare_parameter("receiver_ip", "127.0.0.1")
        self.declare_parameter("receiver_port", 5005)
        self.declare_parameter("send_hz", 10.0)

        self.declare_parameter("target_x", 1.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_yaw", 0.0)
        self.declare_parameter("goal_tolerance", 0.15)
        self.declare_parameter("yaw_tolerance", 0.20)

        self.declare_parameter("kp_linear", 0.45)
        self.declare_parameter("kp_angular", 1.20)
        self.declare_parameter("max_vx", 0.35)
        self.declare_parameter("max_wz", 0.45)
        self.declare_parameter("rotate_in_place_angle", 0.75)
        self.declare_parameter("pose_timeout", 1.0)

        self.pose_topic = self.get_parameter("pose_topic").value
        self.pose_type = str(self.get_parameter("pose_type").value).strip().lower()
        self.goal_topic = self.get_parameter("goal_topic").value
        self.receiver = (
            str(self.get_parameter("receiver_ip").value),
            int(self.get_parameter("receiver_port").value),
        )
        self.send_hz = max(1.0, float(self.get_parameter("send_hz").value))

        self.goal = Goal(
            float(self.get_parameter("target_x").value),
            float(self.get_parameter("target_y").value),
            float(self.get_parameter("target_yaw").value),
        )
        self.current_pose: Optional[RobotPose] = None
        self.last_pose_time = 0.0
        self.goal_reached = False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._create_pose_subscription()
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self.create_timer(1.0 / self.send_hz, self._control_tick)

        self.get_logger().info(
            f"Listening pose={self.pose_topic} ({self.pose_type}), "
            f"goal={self.goal_topic}, UDP target={self.receiver[0]}:{self.receiver[1]}"
        )
        self.get_logger().info(f"Initial goal: x={self.goal.x:.2f}, y={self.goal.y:.2f}, yaw={self.goal.yaw:.2f}")

    def destroy_node(self) -> bool:
        self._send_velocity(0.0, 0.0, reason="shutdown")
        self.sock.close()
        return super().destroy_node()

    def _create_pose_subscription(self) -> None:
        if self.pose_type in ("pose_stamped", "posestamped", "pose"):
            self.create_subscription(PoseStamped, self.pose_topic, self._on_pose_stamped, 10)
        elif self.pose_type in ("odometry", "odom"):
            self.create_subscription(Odometry, self.pose_topic, self._on_odometry, 10)
        else:
            raise ValueError("pose_type must be 'pose_stamped' or 'odometry'")

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        self._update_pose(msg.pose)

    def _on_odometry(self, msg: Odometry) -> None:
        self._update_pose(msg.pose.pose)

    def _update_pose(self, pose: Pose) -> None:
        self.current_pose = RobotPose(
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=yaw_from_pose(pose),
        )
        self.last_pose_time = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        self.goal = Goal(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=yaw_from_pose(msg.pose),
        )
        self.goal_reached = False
        self.get_logger().info(f"New goal: x={self.goal.x:.2f}, y={self.goal.y:.2f}, yaw={self.goal.yaw:.2f}")

    def _control_tick(self) -> None:
        if self.current_pose is None:
            self._send_velocity(0.0, 0.0, reason="waiting_for_pose")
            return

        if time.monotonic() - self.last_pose_time > float(self.get_parameter("pose_timeout").value):
            self._send_velocity(0.0, 0.0, reason="pose_timeout")
            return

        dx = self.goal.x - self.current_pose.x
        dy = self.goal.y - self.current_pose.y
        distance = math.hypot(dx, dy)
        goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        yaw_tolerance = float(self.get_parameter("yaw_tolerance").value)

        if distance <= goal_tolerance:
            yaw_error = normalize_angle(self.goal.yaw - self.current_pose.yaw)
            if abs(yaw_error) <= yaw_tolerance:
                if not self.goal_reached:
                    self.get_logger().info("Goal reached, sending stop.")
                    self.goal_reached = True
                self._send_velocity(0.0, 0.0, reason="goal_reached")
                return

            wz = clamp(
                float(self.get_parameter("kp_angular").value) * yaw_error,
                -float(self.get_parameter("max_wz").value),
                float(self.get_parameter("max_wz").value),
            )
            self._send_velocity(0.0, wz, reason="align_yaw")
            return

        self.goal_reached = False
        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - self.current_pose.yaw)
        kp_linear = float(self.get_parameter("kp_linear").value)
        kp_angular = float(self.get_parameter("kp_angular").value)
        max_vx = float(self.get_parameter("max_vx").value)
        max_wz = float(self.get_parameter("max_wz").value)
        rotate_in_place_angle = float(self.get_parameter("rotate_in_place_angle").value)

        if abs(heading_error) > rotate_in_place_angle:
            vx = 0.0
        else:
            vx = clamp(kp_linear * distance * max(0.0, math.cos(heading_error)), 0.0, max_vx)

        wz = clamp(kp_angular * heading_error, -max_wz, max_wz)
        self._send_velocity(vx, wz, reason="go_to_goal")

    def _send_velocity(self, vx: float, wz: float, reason: str) -> None:
        payload = {
            "source": "lite2_navigation_bridge",
            "reason": reason,
            "vx": round(float(vx), 4),
            "vy": 0.0,
            "wz": round(float(wz), 4),
            "target": {
                "x": round(self.goal.x, 4),
                "y": round(self.goal.y, 4),
                "yaw": round(self.goal.yaw, 4),
            },
        }
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.receiver)


def main() -> None:
    rclpy.init()
    node = Lite2GoalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
