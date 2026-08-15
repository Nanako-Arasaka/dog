#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立走航点脚本 —— 只走 13 个航点, 不含巡检/抓取。

链路:
  waypoints_FINAL.yaml → 按顺序发 /waypoint/goal → waypoint_navigator
    → /motion/nav_cmd → motion_mux → UDP 5005 → lite2_motion_receiver → 狗

用法(由 run_waypoints_only.sh 调用, 也可单独跑):
  python3 scripts/waypoint_walker.py [--waypoints PATH] [--goal-timeout 60]

行为:
  - 等 /localization/ok=True(SLAM 重定位成功)
  - 按航点顺序逐个发 goal, 等 arrived:xxx
  - 每点超时(默认 60s)自动跳到下一个(不卡死)
  - 走完所有航点 → 打印 DONE + 停狗
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Bool, String

RUNTIME_WAYPOINTS = "/home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml"


class WaypointWalker(Node):
    def __init__(self, waypoints_yaml: str, goal_timeout: float) -> None:
        super().__init__("waypoint_walker")
        self.goal_timeout = goal_timeout

        # 加载航点(保持 yaml 顺序 = FSM 路径顺序)
        data = yaml.safe_load(Path(waypoints_yaml).read_text(encoding="utf-8"))
        self.waypoints = [wp["name"] for wp in data.get("waypoints", [])]
        if not self.waypoints:
            self.get_logger().error("waypoints 为空!")
            self.waypoints = []
        self.get_logger().info(f"加载 {len(self.waypoints)} 个航点: {self.waypoints}")

        self.goal_pub = self.create_publisher(String, "/waypoint/goal", 10)
        self.status_sub = self.create_subscription(String, "/waypoint/status", self._on_status, 10)
        self.loc_sub = self.create_subscription(Bool, "/localization/ok", self._on_loc_ok, 10)

        self.localization_ok = False
        self.current_idx = -1
        self.arrived = False
        self.goal_started_at = 0.0
        self.done = False

    def _on_loc_ok(self, msg: Bool) -> None:
        self.localization_ok = msg.data
        if self.localization_ok and not self.done:
            self.get_logger().info("定位 OK, 开始走航点")

    def _on_status(self, msg: String) -> None:
        text = msg.data
        if text.startswith("arrived:"):
            name = text.split(":", 1)[1]
            self.get_logger().info(f"✅ 到达航点: {name}")
            self.arrived = True

    def _send_next_goal(self) -> None:
        if self.current_idx + 1 >= len(self.waypoints):
            self.done = True
            self.get_logger().info("🎉 所有航点走完! DONE")
            return
        self.current_idx += 1
        name = self.waypoints[self.current_idx]
        self.arrived = False
        self.goal_started_at = time.monotonic()
        self.get_logger().info(f"🚶 发航点 [{self.current_idx + 1}/{len(self.waypoints)}]: {name}")
        self.goal_pub.publish(String(data=name))

    def tick(self) -> None:
        if self.done:
            return
        if not self.localization_ok:
            # 等待定位
            return
        if self.current_idx < 0:
            self._send_next_goal()
            return
        if self.arrived:
            # 到达 → 发下一个(给 1s 稳定)
            time.sleep(1.0)
            self._send_next_goal()
            return
        # 超时跳过: 当前航点超时未到达 → 跳下一个
        if time.monotonic() - self.goal_started_at > self.goal_timeout:
            name = self.waypoints[self.current_idx]
            self.get_logger().warn(f"⚠️ 航点 {name} 超时({self.goal_timeout:.0f}s), 跳过继续")
            self.arrived = True
            self._send_next_goal()


def main() -> None:
    parser = argparse.ArgumentParser(description="独立走航点")
    parser.add_argument("--waypoints", default=RUNTIME_WAYPOINTS)
    parser.add_argument("--goal-timeout", type=float, default=60.0)
    args = parser.parse_args()

    rclpy.init()
    node = WaypointWalker(args.waypoints, args.goal_timeout)
    rclpy.spin_once(node, timeout_sec=0)  # 预热

    try:
        while rclpy.ok() and not node.done:
            node.tick()
            rclpy.spin_once(node, timeout_sec=0.2)
        # 等待完成后的最终状态
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("[waypoint_walker] 退出")


if __name__ == "__main__":
    main()
