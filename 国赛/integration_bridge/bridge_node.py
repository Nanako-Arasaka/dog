#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 node and CLI for the national competition integration bridge.

Subscriptions:
  /bridge/inspection_result  std_msgs/String JSON or "A:abnormal,B:normal"
  /bridge/placement_zone     std_msgs/String JSON, "A", or "zone_A"

Publications:
  /inspection/all            std_msgs/String "A:abnormal,B:normal,C:unknown,D:unknown"
  /placement/recognized_zone std_msgs/String "A"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integration_bridge.bridge_core import IntegrationBridge
from integration_bridge.event_logger import EventLogger
from integration_bridge.ros_publishers import RosBridgePublishers


DEFAULT_LOG_PATH = "output/integration_bridge/events.jsonl"


class PrintPublisher:
    def publish_inspection_all(self, text: str) -> None:
        print(f"/inspection/all <- {text}")

    def publish_placement_zone(self, zone: str) -> None:
        print(f"/placement/recognized_zone <- {zone}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward inspection and placement state.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--no-ros", action="store_true", help="Run format/log checks without ROS2.")
    parser.add_argument("--inspection-json", help="Forward one inspection JSON or compact string.")
    parser.add_argument("--placement-zone", help="Forward one placement zone payload.")
    return parser


def run_once(args: argparse.Namespace) -> int:
    logger = EventLogger(args.log_path)
    bridge = IntegrationBridge(publisher=PrintPublisher(), logger=logger)
    if args.inspection_json:
        bridge.handle_inspection_payload(args.inspection_json)
    if args.placement_zone:
        bridge.handle_placement_payload(args.placement_zone)
    if not args.inspection_json and not args.placement_zone:
        print("No payload provided. Use --inspection-json or --placement-zone.")
        return 2
    return 0


def run_ros(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as exc:
        print(f"ROS2 Python dependencies are not available: {exc}", file=sys.stderr)
        print("Use --no-ros for local format/log validation.", file=sys.stderr)
        return 2

    class IntegrationBridgeNode(Node):
        def __init__(self):
            super().__init__("integration_bridge_node")
            logger = EventLogger(args.log_path)
            self.bridge = IntegrationBridge(
                publisher=RosBridgePublishers(self),
                logger=logger,
            )
            self.create_subscription(
                String, "/bridge/inspection_result", self._on_inspection_result, 10
            )
            self.create_subscription(
                String, "/bridge/placement_zone", self._on_placement_zone, 10
            )
            self.get_logger().info("integration_bridge_node ready")
            self.get_logger().info("sub: /bridge/inspection_result -> pub: /inspection/all")
            self.get_logger().info("sub: /bridge/placement_zone -> pub: /placement/recognized_zone")

        def _on_inspection_result(self, msg):
            try:
                text = self.bridge.handle_inspection_payload(msg.data)
                self.get_logger().info(f"published /inspection/all: {text}")
            except Exception as exc:
                self.get_logger().error(f"inspection bridge failed: {exc}")

        def _on_placement_zone(self, msg):
            try:
                zone = self.bridge.handle_placement_payload(msg.data)
                self.get_logger().info(f"published /placement/recognized_zone: {zone}")
            except Exception as exc:
                self.get_logger().error(f"placement bridge failed: {exc}")

    rclpy.init()
    node: Optional[IntegrationBridgeNode] = IntegrationBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    has_single_payload = bool(args.inspection_json or args.placement_zone)
    if args.no_ros or has_single_payload:
        return run_once(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
