#!/usr/bin/env python3
"""Print the important ROS2 topics for real competition-flow integration tests."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable


DEFAULT_TOPICS = (
    "/competition/state",
    "/inspection/all",
    "/inspection/all_zones",
    "/inspection/target_zones",
    "/vision/detect_request",
    "/vision/grasp_pose",
    "/arm/command",
    "/arm/feedback",
    "/arm/state",
    "/placement/recognized_zone",
    "/task/status",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor real-flow ROS2 topics in one terminal.")
    parser.add_argument("--topics", nargs="*", default=list(DEFAULT_TOPICS))
    parser.add_argument("--node-list-interval", type=float, default=5.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as exc:
        print(f"ERROR: ROS2 Python dependencies are unavailable: {exc}", file=sys.stderr)
        return 2

    class MonitorNode(Node):
        def __init__(self, topics: Iterable[str]) -> None:
            super().__init__("real_flow_monitor")
            self.last_node_list = 0.0
            for topic in topics:
                self.create_subscription(String, topic, self._callback(topic), 10)

        def _callback(self, topic: str):
            def callback(msg) -> None:
                now = time.strftime("%H:%M:%S")
                print(f"[{now}] {topic}: {msg.data}", flush=True)

            return callback

        def maybe_print_nodes(self, interval: float) -> None:
            now = time.monotonic()
            if interval <= 0 or now - self.last_node_list < interval:
                return
            self.last_node_list = now
            names = sorted(set(self.get_node_names()))
            print(f"[nodes] {', '.join(names) if names else '(none)'}", flush=True)

    rclpy.init(args=None)
    node = MonitorNode(args.topics)
    print("Monitoring topics. Press Ctrl+C to stop.")
    print("Topics: " + ", ".join(args.topics))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.maybe_print_nodes(args.node_list_interval)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
