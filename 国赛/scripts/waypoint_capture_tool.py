#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture stable /camera_pose samples into waypoints_FINAL.yaml."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


DEFAULT_WAYPOINTS = [
    "start_exit",
    "obstacle_entry",
    "obstacle_exit",
    "inspection_box_1_side_1",
    "inspection_box_1_side_2",
    "inspection_box_2_side_1",
    "inspection_box_2_side_2",
    "pick_area",
    "place_A",
    "place_B",
    "place_C",
    "place_D",
    "finish",
]


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def expand_path(path_text: str) -> Path:
    return Path(os.path.expandvars(path_text)).expanduser()


def load_waypoints(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("waypoints", data) if isinstance(data, dict) else data
    out: dict[str, dict[str, float]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                out[name] = {
                    "x": float(item.get("x", 0.0)),
                    "y": float(item.get("y", 0.0)),
                    "yaw": float(item.get("yaw", 0.0)),
                }
    elif isinstance(items, dict):
        for name, value in items.items():
            value = value or {}
            out[str(name)] = {
                "x": float(value.get("x", 0.0)),
                "y": float(value.get("y", 0.0)),
                "yaw": float(value.get("yaw", 0.0)),
            }
    return out


def save_waypoints(path: Path, waypoints: dict[str, dict[str, float]], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_names = list(order)
    ordered_names += sorted(name for name in waypoints if name not in set(ordered_names))
    data: dict[str, Any] = {"waypoints": []}
    for name in ordered_names:
        pose = waypoints.get(name, {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
        data["waypoints"].append(
            {
                "name": name,
                "x": round(float(pose["x"]), 6),
                "y": round(float(pose["y"]), 6),
                "z": round(float(pose.get("z", 0.0)), 6),
                "yaw": round(float(pose["yaw"]), 6),
            }
        )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    tmp_path.replace(path)


class PoseSampler(Node):
    def __init__(self, pose_topic: str, pose_type: str, stable_samples: int) -> None:
        super().__init__("waypoint_capture_tool")
        self.samples: list[tuple[float, float, float]] = []
        self.stable_samples = stable_samples
        pose_type = pose_type.lower()
        if pose_type in ("pose_stamped", "posestamped", "pose"):
            self.create_subscription(PoseStamped, pose_topic, self._on_pose_stamped, 10)
        elif pose_type in ("odometry", "odom"):
            self.create_subscription(Odometry, pose_topic, self._on_odom, 10)
        else:
            raise ValueError("pose_type must be pose_stamped or odometry")

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        self._append_pose(msg.pose)

    def _on_odom(self, msg: Odometry) -> None:
        self._append_pose(msg.pose.pose)

    def _append_pose(self, pose: Pose) -> None:
        self.samples.append((float(pose.position.x), float(pose.position.y),
                             float(pose.position.z), yaw_from_pose(pose)))
        if len(self.samples) > max(2, self.stable_samples * 3):
            self.samples = self.samples[-self.stable_samples * 3 :]

    def wait_stable(
        self,
        max_position_step: float,
        max_yaw_step: float,
        timeout_sec: float,
    ) -> tuple[float, float, float]:
        start = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            stable = self._stable_pose(max_position_step, max_yaw_step)
            if stable is not None:
                return stable
            if timeout_sec > 0 and time.monotonic() - start > timeout_sec:
                raise TimeoutError("pose did not become stable before timeout")
        raise RuntimeError("rclpy stopped before a pose was captured")

    def _stable_pose(self, max_position_step: float, max_yaw_step: float) -> tuple[float, float, float, float] | None:
        if len(self.samples) < self.stable_samples:
            return None
        window = self.samples[-self.stable_samples :]
        for prev, current in zip(window, window[1:]):
            step = math.sqrt((current[0]-prev[0])**2 +
                             (current[1]-prev[1])**2 +
                             (current[2]-prev[2])**2)
            yaw_step = abs(normalize_angle(current[3] - prev[3]))
            if step > max_position_step or yaw_step > max_yaw_step:
                return None
        x = sum(item[0] for item in window) / len(window)
        y = sum(item[1] for item in window) / len(window)
        z = sum(item[2] for item in window) / len(window)
        sin_yaw = sum(math.sin(item[3]) for item in window) / len(window)
        cos_yaw = sum(math.cos(item[3]) for item in window) / len(window)
        return x, y, z, math.atan2(sin_yaw, cos_yaw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture ROS pose waypoints into YAML.")
    parser.add_argument("--output", required=True, help="waypoints_FINAL.yaml output path")
    parser.add_argument("--pose-topic", default="/camera_pose")
    parser.add_argument("--pose-type", default="pose_stamped")
    parser.add_argument("--stable-samples", type=int, default=10)
    parser.add_argument("--stable-max-position-step", type=float, default=0.04)
    parser.add_argument("--stable-max-yaw-step", type=float, default=0.18)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--waypoints", nargs="*", default=DEFAULT_WAYPOINTS)
    parser.add_argument("--single", help="capture one named waypoint and exit")
    return parser.parse_args()


def capture_one(args: argparse.Namespace, sampler: PoseSampler, name: str) -> dict[str, float]:
    x, y, z, yaw = sampler.wait_stable(
        max_position_step=args.stable_max_position_step,
        max_yaw_step=args.stable_max_yaw_step,
        timeout_sec=args.timeout_sec,
    )
    return {"x": x, "y": y, "z": z, "yaw": yaw}


def main() -> int:
    args = parse_args()
    output = expand_path(args.output)
    order = [str(item).strip() for item in args.waypoints if str(item).strip()]
    if not order:
        print("[ERROR] waypoint list is empty", file=sys.stderr)
        return 2

    waypoints = load_waypoints(output)
    rclpy.init()
    sampler = PoseSampler(args.pose_topic, args.pose_type, max(2, args.stable_samples))
    try:
        if args.single:
            name = args.single.strip()
            waypoints[name] = capture_one(args, sampler, name)
            if name not in order:
                order.append(name)
            save_waypoints(output, waypoints, order)
            pose = waypoints[name]
            print(f"  - name: {name}")
            print(f"    x: {pose['x']:.6f}")
            print(f"    y: {pose['y']:.6f}")
            print(f"    yaw: {pose['yaw']:.6f}")
            return 0

        print("[INFO] Interactive waypoint capture")
        print(f"[INFO] output={output}")
        print("[INFO] Press Enter at each position after the robot/camera is stable.")
        print("[INFO] Type 's' to skip a point, 'q' to quit.")
        for index, name in enumerate(order, start=1):
            while True:
                answer = input(f"\n[{index}/{len(order)}] Move to {name}, then press Enter: ").strip().lower()
                if answer in ("q", "quit", "exit"):
                    save_waypoints(output, waypoints, order)
                    print(f"[INFO] saved partial waypoints: {output}")
                    return 0
                if answer in ("s", "skip"):
                    print(f"[SKIP] {name}")
                    break
                try:
                    waypoints[name] = capture_one(args, sampler, name)
                except TimeoutError as exc:
                    print(f"[WARN] {exc}; keep /camera_pose alive and try again.")
                    continue
                save_waypoints(output, waypoints, order)
                pose = waypoints[name]
                print(f"[OK] {name}: x={pose['x']:.6f} y={pose['y']:.6f} yaw={pose['yaw']:.6f}")
                print(f"[INFO] saved: {output}")
                break
        print("\n[DONE] waypoint capture complete")
        print(output)
        return 0
    finally:
        sampler.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
