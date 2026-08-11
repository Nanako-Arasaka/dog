#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Navigate named waypoints from YAML and publish Twist to motion_mux."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


class WaypointNavigator(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_navigator")
        self.declare_parameter("waypoints_yaml", "")
        self.declare_parameter("pose_topic", "/camera_pose")
        self.declare_parameter("pose_type", "pose_stamped")
        self.declare_parameter("goal_topic", "/waypoint/goal")
        self.declare_parameter("status_topic", "/waypoint/status")
        self.declare_parameter("cmd_topic", "/motion/nav_cmd")
        self.declare_parameter("localization_ok_topic", "/localization/ok")
        self.declare_parameter("goal_tolerance", 0.16)
        self.declare_parameter("yaw_tolerance", 0.22)
        self.declare_parameter("kp_linear", 0.45)
        self.declare_parameter("kp_angular", 1.2)
        self.declare_parameter("max_vx", 0.28)
        self.declare_parameter("max_wz", 0.45)
        self.declare_parameter("rotate_in_place_angle", 0.75)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.pose_type = str(self.get_parameter("pose_type").value).lower()
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.yaw_tolerance = float(self.get_parameter("yaw_tolerance").value)
        self.kp_linear = float(self.get_parameter("kp_linear").value)
        self.kp_angular = float(self.get_parameter("kp_angular").value)
        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_wz = float(self.get_parameter("max_wz").value)
        self.rotate_in_place_angle = float(self.get_parameter("rotate_in_place_angle").value)

        self.waypoints = self._load_waypoints(str(self.get_parameter("waypoints_yaml").value))
        self.current_pose: Pose2D | None = None
        self.current_goal_name = ""
        self.current_goal: Pose2D | None = None
        self.localization_ok = False
        self.arrived = False

        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.create_subscription(String, str(self.get_parameter("goal_topic").value), self._on_goal, 10)
        self.create_subscription(Bool, str(self.get_parameter("localization_ok_topic").value), self._on_loc_ok, 10)
        if self.pose_type in ("pose_stamped", "posestamped", "pose"):
            self.create_subscription(PoseStamped, self.pose_topic, self._on_pose_stamped, 10)
        elif self.pose_type in ("odometry", "odom"):
            self.create_subscription(Odometry, self.pose_topic, self._on_odom, 10)
        else:
            raise ValueError("pose_type must be pose_stamped or odometry")
        self.create_timer(0.1, self._tick)
        self.get_logger().info(f"loaded {len(self.waypoints)} waypoints")

    def _load_waypoints(self, path_text: str) -> dict[str, Pose2D]:
        path = Path(path_text)
        if not path.exists():
            self.get_logger().warn(f"waypoints yaml not found: {path_text}")
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out: dict[str, Pose2D] = {}
        candidates = data.get("waypoints", data) if isinstance(data, dict) else data
        if isinstance(candidates, list):
            for item in candidates:
                name, pose = self._parse_waypoint_item(item)
                if name:
                    out[name] = pose
        elif isinstance(candidates, dict):
            for name, value in candidates.items():
                _, pose = self._parse_waypoint_item({"name": name, **(value or {})})
                out[str(name)] = pose
        return out

    def _parse_waypoint_item(self, item: Any) -> tuple[str, Pose2D]:
        if not isinstance(item, dict):
            return "", Pose2D(0.0, 0.0, 0.0)
        name = str(item.get("name", item.get("id", ""))).strip()
        if "pose" in item and isinstance(item["pose"], dict):
            item = {**item, **item["pose"]}
        x = float(item.get("x", item.get("target_x", 0.0)))
        y = float(item.get("y", item.get("target_y", 0.0)))
        yaw = float(item.get("yaw", item.get("theta", item.get("target_yaw", 0.0))))
        return name, Pose2D(x, y, yaw)

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        self._update_pose(msg.pose)

    def _on_odom(self, msg: Odometry) -> None:
        self._update_pose(msg.pose.pose)

    def _update_pose(self, pose: Pose) -> None:
        self.current_pose = Pose2D(float(pose.position.x), float(pose.position.y), yaw_from_pose(pose))

    def _on_loc_ok(self, msg: Bool) -> None:
        self.localization_ok = bool(msg.data)

    def _on_goal(self, msg: String) -> None:
        name = msg.data.strip()
        if not name:
            self._clear_goal("goal_cleared")
            return
        if name not in self.waypoints:
            self.get_logger().error(f"unknown waypoint: {name}")
            self.status_pub.publish(String(data=f"unknown:{name}"))
            self._publish_stop()
            return
        self.current_goal_name = name
        self.current_goal = self.waypoints[name]
        self.arrived = False
        self.get_logger().info(
            f"new waypoint {name}: x={self.current_goal.x:.2f} "
            f"y={self.current_goal.y:.2f} yaw={self.current_goal.yaw:.2f}"
        )
        self.status_pub.publish(String(data=f"active:{name}"))

    def _clear_goal(self, reason: str) -> None:
        self.current_goal_name = ""
        self.current_goal = None
        self.arrived = False
        self._publish_stop()
        self.status_pub.publish(String(data=reason))

    def _tick(self) -> None:
        if self.current_goal is None:
            self._publish_stop()
            return
        if self.current_pose is None or not self.localization_ok:
            self._publish_stop()
            self.status_pub.publish(String(data=f"blocked:{self.current_goal_name}:localization"))
            return

        dx = self.current_goal.x - self.current_pose.x
        dy = self.current_goal.y - self.current_pose.y
        distance = math.hypot(dx, dy)
        yaw_error = normalize_angle(self.current_goal.yaw - self.current_pose.yaw)

        if distance <= self.goal_tolerance and abs(yaw_error) <= self.yaw_tolerance:
            if not self.arrived:
                self.get_logger().info(f"arrived waypoint {self.current_goal_name}")
                self.status_pub.publish(String(data=f"arrived:{self.current_goal_name}"))
                self.arrived = True
            self._publish_stop()
            return

        heading_error = normalize_angle(math.atan2(dy, dx) - self.current_pose.yaw)
        twist = Twist()
        if distance <= self.goal_tolerance:
            twist.angular.z = clamp(self.kp_angular * yaw_error, -self.max_wz, self.max_wz)
        else:
            if abs(heading_error) <= self.rotate_in_place_angle:
                twist.linear.x = clamp(
                    self.kp_linear * distance * max(0.0, math.cos(heading_error)),
                    0.0,
                    self.max_vx,
                )
            twist.angular.z = clamp(self.kp_angular * heading_error, -self.max_wz, self.max_wz)
        self.cmd_pub.publish(twist)
        self.status_pub.publish(String(data=f"moving:{self.current_goal_name}:{distance:.2f}"))

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())


def main() -> None:
    rclpy.init()
    node = WaypointNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
