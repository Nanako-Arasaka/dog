#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Initial coordinator for inspection-driven red-bar pick and place.

This program glues the already-built modules without controlling the arm or the
robot directly:

1. Read final A/B/C/D inspection memory from /inspection/all.
2. Remember normal and abnormal zone letters.
3. When the robot reaches the red-bar pickup area, publish the abnormal target
   queue to /inspection/target_zones and trigger /task/start.
4. Let arm_grasp/task_manager_node.py handle grasping, placement-zone matching,
   and the final place command.

Expected ROS2 topics:
  sub: /inspection/all       std_msgs/String  "A:abnormal,B:normal,C:abnormal,D:normal"
  sub: /mission/event        std_msgs/String  "pick_area_arrived"
  pub: /inspection/target_zones std_msgs/String "A,C"
  pub: /task/start           std_msgs/String  "start"
  pub: /pick_place/status    std_msgs/String  compact human-readable state

Local format check:
  python3 tools/inspection_pick_place_coordinator.py --no-ros \
    --inspection "A:abnormal,B:normal,C:abnormal,D:normal" --auto-pick-ready
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration_bridge.schemas import ZONES, inspections_from_payload  # noqa: E402


WAITING_INSPECTION = "WAITING_INSPECTION"
WAITING_PICK_AREA = "WAITING_PICK_AREA"
READY_TO_START_PICK = "READY_TO_START_PICK"
NO_ABNORMAL = "NO_ABNORMAL"


@dataclass
class CoordinatorSnapshot:
    state: str
    zone_states: dict[str, str] = field(default_factory=dict)
    abnormal_targets: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def target_text(self) -> str:
        return ",".join(self.abnormal_targets)

    @property
    def inspection_text(self) -> str:
        return ",".join(f"{zone}:{self.zone_states.get(zone, 'unknown')}" for zone in ZONES)

    def status_text(self) -> str:
        return (
            f"state={self.state} inspection={self.inspection_text} "
            f"targets={self.target_text or '-'} message={self.message or '-'}"
        )


class InspectionPickPlaceCoordinator:
    """Small state holder for inspection memory and red-bar task start."""

    def __init__(self, required_zones: Iterable[str] = ZONES):
        self.required_zones = tuple(required_zones)
        self.snapshot = CoordinatorSnapshot(
            state=WAITING_INSPECTION,
            zone_states={zone: "unknown" for zone in self.required_zones},
            abnormal_targets=[],
            message="waiting final inspection",
        )

    def update_inspection(self, payload: str) -> CoordinatorSnapshot:
        results = inspections_from_payload(payload)
        zone_states = {zone: "unknown" for zone in self.required_zones}
        for result in results:
            if result.zone in zone_states:
                zone_states[result.zone] = result.zone_state

        missing = [zone for zone, state in zone_states.items() if state == "unknown"]
        abnormal = [zone for zone in self.required_zones if zone_states.get(zone) == "abnormal"]

        if missing:
            self.snapshot = CoordinatorSnapshot(
                state=WAITING_INSPECTION,
                zone_states=zone_states,
                abnormal_targets=abnormal,
                message="missing stable zones: " + ",".join(missing),
            )
            return self.snapshot

        if not abnormal:
            self.snapshot = CoordinatorSnapshot(
                state=NO_ABNORMAL,
                zone_states=zone_states,
                abnormal_targets=[],
                message="all gauges normal; skip red-bar task",
            )
            return self.snapshot

        self.snapshot = CoordinatorSnapshot(
            state=WAITING_PICK_AREA,
            zone_states=zone_states,
            abnormal_targets=abnormal,
            message="inspection complete; wait pick_area_arrived",
        )
        return self.snapshot

    def mark_pick_area_ready(self) -> CoordinatorSnapshot:
        if not self.snapshot.abnormal_targets:
            self.snapshot.message = "no abnormal targets to start"
            return self.snapshot
        if self.snapshot.state == WAITING_INSPECTION:
            self.snapshot.message = "inspection not complete"
            return self.snapshot

        self.snapshot.state = READY_TO_START_PICK
        self.snapshot.message = "start red-bar grasp task"
        return self.snapshot


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coordinate inspection memory with red-bar pick/place start.")
    parser.add_argument("--no-ros", action="store_true", help="Run one local format check without ROS2.")
    parser.add_argument("--inspection", default="", help="Inspection payload for --no-ros mode.")
    parser.add_argument("--auto-pick-ready", action="store_true", help="Start immediately after inspection is complete.")
    parser.add_argument("--mission-topic", default="/mission/event")
    parser.add_argument("--pick-ready-token", default="pick_area_arrived")
    return parser


def run_once(args: argparse.Namespace) -> int:
    coordinator = InspectionPickPlaceCoordinator()
    if not args.inspection:
        print("No --inspection payload provided.")
        return 2
    snapshot = coordinator.update_inspection(args.inspection)
    print(snapshot.status_text())
    if args.auto_pick_ready:
        snapshot = coordinator.mark_pick_area_ready()
        print(snapshot.status_text())
        if snapshot.state == READY_TO_START_PICK:
            print(f"/inspection/target_zones <- {snapshot.target_text}")
            print("/task/start <- start")
    return 0


def run_ros(args: argparse.Namespace) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as exc:
        print(f"ROS2 Python dependencies are not available: {exc}", file=sys.stderr)
        print("Use --no-ros for local format validation.", file=sys.stderr)
        return 2

    class CoordinatorNode(Node):
        def __init__(self) -> None:
            super().__init__("inspection_pick_place_coordinator")
            self.coordinator = InspectionPickPlaceCoordinator()
            self.pick_ready_token = args.pick_ready_token.strip().lower()
            self.pub_targets = self.create_publisher(String, "/inspection/target_zones", 10)
            self.pub_start = self.create_publisher(String, "/task/start", 10)
            self.pub_status = self.create_publisher(String, "/pick_place/status", 10)
            self.create_subscription(String, "/inspection/all", self._on_inspection, 10)
            self.create_subscription(String, args.mission_topic, self._on_mission_event, 10)
            self.create_timer(1.0, self._publish_status)
            self.get_logger().info("inspection_pick_place_coordinator ready")
            self.get_logger().info(f"waiting mission event '{args.pick_ready_token}' on {args.mission_topic}")

        def _on_inspection(self, msg) -> None:
            try:
                snapshot = self.coordinator.update_inspection(msg.data)
            except Exception as exc:
                self.get_logger().error(f"inspection parse failed: {exc}")
                return
            self.get_logger().info(snapshot.status_text())
            self._publish_status()
            if args.auto_pick_ready and snapshot.state == WAITING_PICK_AREA:
                self._start_pick_task()

        def _on_mission_event(self, msg) -> None:
            token = (msg.data or "").strip().lower()
            if token != self.pick_ready_token:
                return
            self._start_pick_task()

        def _start_pick_task(self) -> None:
            snapshot = self.coordinator.mark_pick_area_ready()
            self.get_logger().info(snapshot.status_text())
            self._publish_status()
            if snapshot.state != READY_TO_START_PICK:
                return
            self.pub_targets.publish(String(data=snapshot.target_text))
            self.pub_start.publish(String(data="start"))
            self.get_logger().info(
                f"published targets={snapshot.target_text} and /task/start"
            )

        def _publish_status(self) -> None:
            self.pub_status.publish(String(data=self.coordinator.snapshot.status_text()))

    rclpy.init()
    node = CoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.no_ros:
        return run_once(args)
    return run_ros(args)


if __name__ == "__main__":
    raise SystemExit(main())
