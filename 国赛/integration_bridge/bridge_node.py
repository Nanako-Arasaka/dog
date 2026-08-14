#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 node and CLI for the national competition integration bridge.

Subscriptions:
  /bridge/inspection_result  std_msgs/String JSON or "A:abnormal,B:normal"
  /bridge/placement_zone     std_msgs/String JSON, "A", or "zone_A"

Publications:
  /inspection/all            std_msgs/String "A:abnormal,B:normal,C:unknown,D:unknown" (FSM 消费)
  /inspection/all_detailed   std_msgs/String "A:low,B:normal,C:high,D:normal"        (语音播报消费，保留偏低/偏高)
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
from integration_bridge.inspection_freezer import InspectionFreezer
from integration_bridge.ros_publishers import RosBridgePublishers
from integration_bridge.schemas import inspections_from_payload


DEFAULT_LOG_PATH = "output/integration_bridge/events.jsonl"
COMPETITION_STATE_TOPIC = "/competition/state"


class PrintPublisher:
    def publish_inspection_all(self, text: str) -> None:
        print(f"/inspection/all <- {text}")

    def publish_inspection_all_detailed(self, text: str) -> None:
        print(f"/inspection/all_detailed <- {text}")

    def publish_placement_zone(self, zone: str) -> None:
        print(f"/placement/recognized_zone <- {zone}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward inspection and placement state.")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--no-ros", action="store_true", help="Run format/log checks without ROS2.")
    parser.add_argument("--inspection-json", help="Forward one inspection JSON or compact string.")
    parser.add_argument("--placement-zone", help="Forward one placement zone payload.")
    parser.add_argument(
        "--no-freeze-inspection",
        action="store_true",
        help="Immediately forward inspection results instead of waiting for stable A/B/C/D.",
    )
    parser.add_argument("--zone-stable-count", type=int, default=3)
    parser.add_argument("--frozen-publish-interval", type=float, default=1.0)
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
        from std_msgs.msg import Bool, String
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
            self.freeze_inspection = not args.no_freeze_inspection
            self.freezer = InspectionFreezer(stable_count=args.zone_stable_count)
            self.frozen_inspection_text = None
            self.frozen_inspection_text_detailed = None
            self.last_frozen_publish_time = 0.0
            self.pub_state = self.create_publisher(String, COMPETITION_STATE_TOPIC, 10)
            self.create_subscription(
                String, "/bridge/inspection_result", self._on_inspection_result, 10
            )
            self.create_subscription(
                String, "/bridge/placement_zone", self._on_placement_zone, 10
            )
            self.create_subscription(Bool, "/inspection/reset", self._on_inspection_reset, 10)
            self.create_timer(max(float(args.frozen_publish_interval), 0.2), self._on_timer)
            self.get_logger().info("integration_bridge_node ready")
            if self.freeze_inspection:
                self.get_logger().info(
                    "sub: /bridge/inspection_result -> freeze A/B/C/D -> pub: /inspection/all"
                )
            else:
                self.get_logger().info("sub: /bridge/inspection_result -> pub: /inspection/all")
            self.get_logger().info("sub: /bridge/placement_zone -> pub: /placement/recognized_zone")
            self._publish_state("WAITING_INSPECTION")

        def _on_inspection_result(self, msg):
            try:
                if self.freeze_inspection:
                    self._handle_frozen_inspection(msg.data)
                else:
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

        def _on_inspection_reset(self, msg):
            if not msg.data:
                return
            self.freezer.reset()
            self.frozen_inspection_text = None
            self.frozen_inspection_text_detailed = None
            self.bridge.inspection_memory.clear()
            self._publish_state("WAITING_INSPECTION")
            self.get_logger().info("inspection freeze reset")

        def _handle_frozen_inspection(self, payload: str):
            for result in inspections_from_payload(payload):
                self.bridge.logger.write(result.to_event())
                newly_frozen = self.freezer.update(result)
                if newly_frozen:
                    self.get_logger().info(f"zone frozen: {result.zone}:{result.zone_state}")
                    self.get_logger().info(f"inspection progress: {self.freezer.progress_text()}")
                    self._publish_state(f"INSPECTION_PROGRESS:{self.freezer.progress_text()}")

            if self.freezer.is_complete() and self.frozen_inspection_text is None:
                self.frozen_inspection_text = self.freezer.frozen_text()
                self.frozen_inspection_text_detailed = self.freezer.frozen_text_detailed()
                # 同步 memory 快照 + publish 日志(供排错查询,与 handle_inspection_results 格式一致)
                self.bridge.inspection_memory = {r.zone: r for r in self.freezer._frozen.values()}
                self.bridge.logger.write(
                    {"type": "publish", "topic": "/inspection/all", "data": self.frozen_inspection_text}
                )
                self.bridge.logger.write(
                    {"type": "publish", "topic": "/inspection/all_detailed", "data": self.frozen_inspection_text_detailed}
                )
                self.bridge.logger.write(
                    {
                        "type": "inspection_frozen",
                        "topic": "/inspection/all",
                        "data": self.frozen_inspection_text,
                    }
                )
                self.bridge.logger.write(
                    {
                        "type": "inspection_frozen_detailed",
                        "topic": "/inspection/all_detailed",
                        "data": self.frozen_inspection_text_detailed,
                    }
                )
                self.bridge.publisher.publish_inspection_all(self.frozen_inspection_text)
                self._publish_detailed(self.frozen_inspection_text_detailed)
                self._publish_state(f"INSPECTION_FROZEN:{self.frozen_inspection_text}")
                self.get_logger().info(f"published frozen /inspection/all: {self.frozen_inspection_text}")
                self.get_logger().info(
                    f"published frozen /inspection/all_detailed: {self.frozen_inspection_text_detailed}"
                )

        def _on_timer(self):
            if self.frozen_inspection_text:
                self.bridge.publisher.publish_inspection_all(self.frozen_inspection_text)
                if self.frozen_inspection_text_detailed:
                    self._publish_detailed(self.frozen_inspection_text_detailed)

        def _publish_detailed(self, text: str):
            """发布 /inspection/all_detailed（publisher 可能未实现该方法，做兜底）。"""
            pub = getattr(self.bridge.publisher, "publish_inspection_all_detailed", None)
            if callable(pub):
                pub(text)

        def _publish_state(self, state: str):
            self.pub_state.publish(String(data=state))
            self.bridge.logger.write({"type": "competition_state", "data": state})

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
