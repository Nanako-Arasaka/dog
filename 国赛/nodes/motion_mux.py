#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single motion output gate for the final competition flow."""

from __future__ import annotations

import json
import socket
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String


class MotionMux(Node):
    def __init__(self) -> None:
        super().__init__("motion_mux")
        self.declare_parameter("receiver_host", "127.0.0.1")
        self.declare_parameter("receiver_port", 5005)
        self.declare_parameter("send_hz", 10.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("nav_cmd_topic", "/motion/nav_cmd")
        self.declare_parameter("avoid_cmd_topic", "/motion/avoid_cmd")
        self.declare_parameter("stop_topic", "/motion/stop")
        self.declare_parameter("localization_ok_topic", "/localization/ok")
        self.declare_parameter("enable_cone_topic", "/motion/enable_cone_avoidance")
        self.declare_parameter("state_topic", "/motion_mux/state")
        self.declare_parameter("max_cmd_age_sec", 0.6)
        self.declare_parameter("max_vx", 0.35)
        self.declare_parameter("max_vy", 0.15)
        self.declare_parameter("max_wz", 0.55)
        self.declare_parameter("obstacle_priority", True)

        self.target = (
            str(self.get_parameter("receiver_host").value),
            int(self.get_parameter("receiver_port").value),
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.max_cmd_age_sec = float(self.get_parameter("max_cmd_age_sec").value)
        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)
        self.obstacle_priority = bool(self.get_parameter("obstacle_priority").value)

        self.nav_cmd = Twist()
        self.avoid_cmd = Twist()
        self.nav_time = 0.0
        self.avoid_time = 0.0
        self.localization_ok = False
        self.stop_active = True
        self.cone_enabled = False
        self.last_payload = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_pub = self.create_publisher(String, str(self.get_parameter("state_topic").value), 10)

        self.create_subscription(Twist, str(self.get_parameter("nav_cmd_topic").value), self._on_nav, 10)
        self.create_subscription(Twist, str(self.get_parameter("avoid_cmd_topic").value), self._on_avoid, 10)
        self.create_subscription(Bool, str(self.get_parameter("stop_topic").value), self._on_stop, 10)
        self.create_subscription(Bool, str(self.get_parameter("localization_ok_topic").value), self._on_loc, 10)
        self.create_subscription(Bool, str(self.get_parameter("enable_cone_topic").value), self._on_cone_enable, 10)
        self.create_timer(1.0 / max(1.0, float(self.get_parameter("send_hz").value)), self._tick)
        self.get_logger().info(f"UDP target={self.target[0]}:{self.target[1]} dry_run={self.dry_run}")

    def destroy_node(self) -> bool:
        self._send(0.0, 0.0, 0.0, "shutdown")
        self.sock.close()
        return super().destroy_node()

    def _now(self) -> float:
        return time.monotonic()

    def _on_nav(self, msg: Twist) -> None:
        self.nav_cmd = msg
        self.nav_time = self._now()

    def _on_avoid(self, msg: Twist) -> None:
        self.avoid_cmd = msg
        self.avoid_time = self._now()

    def _on_stop(self, msg: Bool) -> None:
        self.stop_active = bool(msg.data)
        if self.stop_active:
            self._send(0.0, 0.0, 0.0, "external_stop")

    def _on_loc(self, msg: Bool) -> None:
        self.localization_ok = bool(msg.data)
        if not self.localization_ok:
            self._send(0.0, 0.0, 0.0, "localization_not_ok")

    def _on_cone_enable(self, msg: Bool) -> None:
        self.cone_enabled = bool(msg.data)
        self.get_logger().info(f"cone_avoidance_enabled={self.cone_enabled}")

    def _tick(self) -> None:
        if self.stop_active:
            self._send(0.0, 0.0, 0.0, "stop_active")
            return
        if not self.localization_ok:
            self._send(0.0, 0.0, 0.0, "localization_not_ok")
            return

        now = self._now()
        nav_fresh = now - self.nav_time <= self.max_cmd_age_sec
        avoid_fresh = now - self.avoid_time <= self.max_cmd_age_sec
        if self.cone_enabled and self.obstacle_priority:
            # 避障区:命令新鲜则用避障输出;命令失联(如避障节点崩溃/停更)则停止,
            # 绝不退回导航指令继续朝锥桶冲。
            if avoid_fresh:
                self._send_twist(self.avoid_cmd, "cone_avoidance")
            else:
                self._send(0.0, 0.0, 0.0, "avoid_stale_stop")
        elif nav_fresh:
            self._send_twist(self.nav_cmd, "waypoint_navigation")
        else:
            self._send(0.0, 0.0, 0.0, "no_fresh_command")

    def _send_twist(self, msg: Twist, source: str) -> None:
        self._send(
            self._clamp(float(msg.linear.x), -self.max_vx, self.max_vx),
            self._clamp(float(msg.linear.y), -self.max_vy, self.max_vy),
            self._clamp(float(msg.angular.z), -self.max_wz, self.max_wz),
            source,
        )

    def _send(self, vx: float, vy: float, wz: float, source: str) -> None:
        payload = {
            "source": "motion_mux",
            "selected": source,
            "vx": round(float(vx), 4),
            "vy": round(float(vy), 4),
            "wz": round(float(wz), 4),
            "cone_enabled": self.cone_enabled,
            "localization_ok": self.localization_ok,
        }
        if payload != self.last_payload:
            self.state_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))
            self.last_payload = dict(payload)
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if self.dry_run:
            self.get_logger().info(f"dry_run motion {payload}")
            return
        self.sock.sendto(data, self.target)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


def main() -> None:
    rclpy.init()
    node = MotionMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
