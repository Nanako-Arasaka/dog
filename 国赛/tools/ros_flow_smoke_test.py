#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 smoke test for the national competition task glue.

This script drives the real ROS topics without using cameras, the arm, or the
dog. It is meant for weak integration testing with these nodes running:

  integration_bridge_node
  inspection_memory_node
  task_manager_node

Do not run this against a live arm controller unless you pass
--allow-arm-control deliberately.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


DEFAULT_INSPECTION = "A:abnormal,B:normal,C:abnormal,D:normal"
DEFAULT_POSE = "grasp|0.30|0.00|0.12|0|0.9|320|240"


@dataclass
class SeenMessage:
    topic: str
    data: str
    timestamp: float


class SmokeTestFailure(RuntimeError):
    pass


def parse_targets(text: str) -> List[str]:
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def target_zones_from_inspection(text: str) -> List[str]:
    targets = []
    for part in text.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        zone, state = part.split(":", 1)
        if state.strip().lower() == "abnormal":
            targets.append(zone.strip().upper())
    return targets


def choose_wrong_zone(target: str) -> str:
    for zone in ("A", "B", "C", "D"):
        if zone != target:
            return zone
    return "B"


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Bool, String
    except ImportError as exc:
        print(f"ERROR: ROS2 Python dependencies are unavailable: {exc}", file=sys.stderr)
        return 2

    class FlowSmokeNode(Node):
        def __init__(self) -> None:
            super().__init__("competition_flow_smoke_test")
            self.String = String
            self.Bool = Bool
            self.messages: List[SeenMessage] = []
            self.pub_bridge_inspection = self.create_publisher(String, "/bridge/inspection_result", 10)
            self.pub_direct_targets = self.create_publisher(String, "/inspection/target_zones", 10)
            self.pub_vision_pose = self.create_publisher(String, "/vision/grasp_pose", 10)
            self.pub_arm_feedback = self.create_publisher(String, "/arm/feedback", 10)
            self.pub_placement = self.create_publisher(String, "/placement/recognized_zone", 10)
            self.pub_inspection_reset = self.create_publisher(Bool, "/inspection/reset", 10)
            self.pub_task_reset = self.create_publisher(Bool, "/task/reset", 10)

            for topic in (
                "/inspection/all",
                "/inspection/target_zones",
                "/vision/detect_request",
                "/arm/command",
                "/task/status",
                "/competition/state",
            ):
                self.create_subscription(String, topic, self._remember(topic), 10)

        def _remember(self, topic: str):
            def callback(msg) -> None:
                item = SeenMessage(topic=topic, data=msg.data, timestamp=time.monotonic())
                self.messages.append(item)
                print(f"[seen] {topic}: {msg.data}")

            return callback

        def publish_string(self, publisher, data: str, label: str) -> None:
            print(f"[pub] {label}: {data}")
            publisher.publish(self.String(data=data))

        def publish_bool(self, publisher, data: bool, label: str) -> None:
            print(f"[pub] {label}: {data}")
            publisher.publish(self.Bool(data=data))

        def count_messages(self, topic: str, predicate: Callable[[str], bool]) -> int:
            return sum(1 for msg in self.messages if msg.topic == topic and predicate(msg.data))

        def wait_for(
            self,
            label: str,
            predicate: Callable[[SeenMessage], bool],
            timeout: float,
            since: float = 0.0,
        ) -> SeenMessage:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                for msg in reversed(self.messages):
                    if msg.timestamp >= since and predicate(msg):
                        print(f"[ok] {label}: {msg.topic} {msg.data}")
                        return msg
            raise SmokeTestFailure(f"timeout waiting for {label}")

        def spin_for(self, seconds: float) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)

    rclpy.init(args=None)
    node = FlowSmokeNode()
    try:
        run_flow(node, args)
    except SmokeTestFailure as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print_recent_messages(node)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def run_flow(node, args: argparse.Namespace) -> None:
    node.spin_for(0.5)
    node_names = set(node.get_node_names())
    print(f"[nodes] {', '.join(sorted(node_names)) or '(none seen)'}")
    if "arm_control_node" in node_names and not args.allow_arm_control:
        raise SmokeTestFailure(
            "arm_control_node is running; stop it or rerun with --allow-arm-control if you really want hardware commands"
        )
    if "task_manager_node" not in node_names:
        print("[warn] task_manager_node not seen yet; continuing until topic timeout")

    expected_targets = parse_targets(args.expected_targets) if args.expected_targets else target_zones_from_inspection(args.inspection)
    if not expected_targets:
        raise SmokeTestFailure("no abnormal target zones inferred from inspection payload")
    expected_target_text = ",".join(expected_targets)

    if not args.no_reset:
        node.publish_bool(node.pub_inspection_reset, True, "/inspection/reset")
        node.publish_bool(node.pub_task_reset, True, "/task/reset")
        node.spin_for(0.5)

    if task_status_has_targets(node, expected_targets):
        print(f"[ok] task_manager already has targets: {expected_target_text}")
    elif args.direct_targets:
        marker = time.monotonic()
        node.publish_string(node.pub_direct_targets, ",".join(expected_targets), "/inspection/target_zones")
        node.wait_for(
            f"target_zones {expected_target_text}",
            lambda msg: msg.topic == "/inspection/target_zones" and msg.data.strip().upper() == expected_target_text,
            args.timeout,
            since=marker,
        )
    else:
        marker = time.monotonic()
        for index in range(max(1, args.inspection_repeat)):
            node.publish_string(node.pub_bridge_inspection, args.inspection, f"/bridge/inspection_result #{index + 1}")
            node.spin_for(args.publish_gap)
        node.wait_for(
            f"target_zones {expected_target_text}",
            lambda msg: (
                (msg.topic == "/inspection/target_zones" and msg.data.strip().upper() == expected_target_text)
                or (msg.topic == "/task/status" and status_mentions_targets(msg.data, expected_targets))
            ),
            args.timeout,
            since=marker,
        )

    for target in expected_targets:
        run_one_target(node, args, target)

    node.wait_for(
        "task completed",
        lambda msg: msg.topic == "/task/status" and any(token in msg.data for token in ("COMPLETED", "DONE", "FINISHED")),
        args.timeout,
    )
    print("[pass] ROS flow smoke test completed")


def run_one_target(node, args: argparse.Namespace, target: str) -> None:
    print(f"[phase] target={target} grasp")
    marker = time.monotonic()
    node.wait_for(
        "ready for vision pose",
        lambda msg: (
            (msg.topic == "/vision/detect_request" and msg.data.strip())
            or (msg.topic == "/task/status" and "DETECTING" in msg.data)
        ),
        args.timeout,
        since=marker,
    )
    before_grasp = node.count_messages("/arm/command", lambda data: data.startswith("grasp|"))
    marker = time.monotonic()
    node.publish_string(node.pub_vision_pose, args.pose, "/vision/grasp_pose")
    node.wait_for(
        "grasp command",
        lambda msg: msg.topic == "/arm/command" and msg.data.startswith("grasp|"),
        args.timeout,
        since=marker,
    )
    after_grasp = node.count_messages("/arm/command", lambda data: data.startswith("grasp|"))
    if after_grasp <= before_grasp:
        raise SmokeTestFailure("no new grasp command observed")

    marker = time.monotonic()
    node.publish_string(node.pub_arm_feedback, "grasp|success|mock", "/arm/feedback")
    node.wait_for(
        "waiting place zone",
        lambda msg: msg.topic == "/task/status" and "WAITING_PLACE_ZONE" in msg.data,
        args.timeout,
        since=marker,
    )

    wrong_zone = args.wrong_zone or choose_wrong_zone(target)
    before_place = node.count_messages("/arm/command", lambda data: data.startswith("place|"))
    node.publish_string(node.pub_placement, wrong_zone, "/placement/recognized_zone wrong")
    node.spin_for(args.negative_window)
    after_wrong = node.count_messages("/arm/command", lambda data: data.startswith("place|"))
    if after_wrong != before_place:
        raise SmokeTestFailure(f"place command appeared for wrong zone {wrong_zone}; target was {target}")
    print(f"[ok] wrong placement ignored: seen={wrong_zone} target={target}")

    marker = time.monotonic()
    node.publish_string(node.pub_placement, target, "/placement/recognized_zone match")
    node.wait_for(
        "place command",
        lambda msg: msg.topic == "/arm/command" and msg.data.startswith("place|"),
        args.timeout,
        since=marker,
    )
    node.publish_string(node.pub_arm_feedback, "place|success|mock", "/arm/feedback")
    node.spin_for(0.5)


def task_status_has_targets(node, targets: List[str]) -> bool:
    return any(
        msg.topic == "/task/status" and status_mentions_targets(msg.data, targets)
        for msg in node.messages
    )


def status_mentions_targets(text: str, targets: List[str]) -> bool:
    compact = text.replace(" ", "").upper()
    target_text = ",".join(targets).upper()
    return target_text in compact and "DETECTING" in compact


def print_recent_messages(node, limit: int = 30) -> None:
    print("--- recent messages ---", file=sys.stderr)
    for msg in node.messages[-limit:]:
        print(f"{msg.topic}: {msg.data}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a ROS2 dry-run smoke test for the competition flow.")
    parser.add_argument("--inspection", default=DEFAULT_INSPECTION)
    parser.add_argument("--expected-targets", default="")
    parser.add_argument("--pose", default=DEFAULT_POSE)
    parser.add_argument("--inspection-repeat", type=int, default=3)
    parser.add_argument("--publish-gap", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--negative-window", type=float, default=1.0)
    parser.add_argument("--wrong-zone", default="")
    parser.add_argument("--direct-targets", action="store_true", help="Publish /inspection/target_zones directly, bypassing bridge/memory.")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--allow-arm-control", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
