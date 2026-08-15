#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Localization watchdog with SLAM + AprilTag arbitration.

Watches two pose sources:
  1. SLAM        /camera_pose            (primary)
  2. AprilTag    /tag_localizer/pose     (fallback, absolute relocalization)

and publishes the winning source on /camera_pose_fused for downstream nodes
(waypoint_navigator subscribes to the fused topic).

Arbitration rules:
  - SLAM is preferred while it is fresh + stable.
  - When SLAM times out or jumps, switch to the AprilTag source if it is
    fresh + stable (fallback engaged, run keeps going).
  - When SLAM recovers, switch back after a hysteresis window.
  - A source switch never triggers the jump guard (sources are tracked
    separately), and ping-pong is prevented by switch_suppress_sec.
  - ok=False (and motion stop) is only raised when EVERY source has been
    unusable for fault_grace_sec — a transient SLAM dropout covered by tags
    no longer kills the run.
  - On recovery from a fault, stop=False is published so motion_mux unlatches.

During a short source gap (< fault_grace_sec) the node keeps ok=True with
reason "holding:source_gap"; waypoint_navigator independently stops the robot
when the fused pose goes stale, so this grace window is safe.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    yaw: float


def yaw_from_pose(pose: Pose) -> float:
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


class PoseTrack:
    """Per-source pose history: freshness, stability and jump detection."""

    def __init__(
        self,
        name: str,
        stable_samples: int,
        max_position_step: float,
        max_yaw_step: float,
        timeout_sec: float,
        jump_position_threshold: float,
        jump_yaw_threshold: float,
        distrust_sec: float,
    ) -> None:
        self.name = name
        self.timeout_sec = timeout_sec
        self.max_position_step = max_position_step
        self.max_yaw_step = max_yaw_step
        self.jump_position_threshold = jump_position_threshold
        self.jump_yaw_threshold = jump_yaw_threshold
        self.distrust_sec = distrust_sec
        self.samples: deque[Pose2D] = deque(maxlen=max(2, stable_samples))
        self.last_pose: Optional[Pose2D] = None
        self.last_msg: Optional[PoseStamped] = None
        self.last_time: Optional[float] = None
        self.distrusted_until = 0.0

    def update(self, pose: Pose, msg: PoseStamped, now: float) -> str:
        """Ingest one pose. Returns 'ok' or 'jump'."""
        current = Pose2D(float(pose.position.x), float(pose.position.y),
                         float(pose.position.z), yaw_from_pose(pose))
        event = "ok"
        if self.last_pose is not None:
            dist = math.sqrt((current.x - self.last_pose.x) ** 2 +
                             (current.y - self.last_pose.y) ** 2 +
                             (current.z - self.last_pose.z) ** 2)
            dyaw = abs(angle_diff(current.yaw, self.last_pose.yaw))
            if dist > self.jump_position_threshold or dyaw > self.jump_yaw_threshold:
                event = "jump"
                self.samples.clear()
                self.distrusted_until = now + self.distrust_sec
            else:
                self.samples.append(current)
        else:
            self.samples.append(current)
        self.last_pose = current
        self.last_msg = msg
        self.last_time = now
        return event

    def fresh(self, now: float) -> bool:
        return self.last_time is not None and (now - self.last_time) <= self.timeout_sec

    def stable(self) -> bool:
        if len(self.samples) < self.samples.maxlen:
            return False
        prev = None
        for sample in self.samples:
            if prev is not None:
                dist = math.sqrt((sample.x - prev.x) ** 2 +
                                 (sample.y - prev.y) ** 2 +
                                 (sample.z - prev.z) ** 2)
                dyaw = abs(angle_diff(sample.yaw, prev.yaw))
                if dist > self.max_position_step or dyaw > self.max_yaw_step:
                    return False
            prev = sample
        return True

    def usable(self, now: float) -> bool:
        return self.fresh(now) and self.stable() and now >= self.distrusted_until


class LocalizationWatchdog(Node):
    def __init__(self) -> None:
        super().__init__("localization_watchdog")
        self._log_warn = self.get_logger().get_child("fault")
        # --- SLAM source params (unchanged names, backward compatible) ---
        self.declare_parameter("pose_topic", "/camera_pose")
        self.declare_parameter("pose_type", "pose_stamped")
        self.declare_parameter("ok_topic", "/localization/ok")
        self.declare_parameter("status_topic", "/localization/status")
        self.declare_parameter("stop_topic", "/motion/stop")
        self.declare_parameter("stable_samples", 15)
        self.declare_parameter("stable_max_position_step", 0.08)
        self.declare_parameter("stable_max_yaw_step", 0.35)
        self.declare_parameter("pose_timeout_sec", 0.8)
        self.declare_parameter("jump_position_threshold", 0.45)
        self.declare_parameter("jump_yaw_threshold", 1.2)
        # --- AprilTag fallback params (new) ---
        self.declare_parameter("tag_pose_topic", "/tag_localizer/pose")
        self.declare_parameter("fused_pose_topic", "/camera_pose_fused")
        self.declare_parameter("enable_tag_fallback", True)
        self.declare_parameter("tag_timeout_sec", 1.0)
        self.declare_parameter("tag_stable_samples", 3)
        self.declare_parameter("switch_suppress_sec", 2.5)
        self.declare_parameter("slam_distrust_sec", 2.0)
        self.declare_parameter("fault_grace_sec", 0.5)

        pose_topic = str(self.get_parameter("pose_topic").value)
        pose_type = str(self.get_parameter("pose_type").value).lower()
        self.enable_tag_fallback = bool(self.get_parameter("enable_tag_fallback").value)
        fused_topic = str(self.get_parameter("fused_pose_topic").value)
        tag_topic = str(self.get_parameter("tag_pose_topic").value)
        self.switch_suppress_sec = float(self.get_parameter("switch_suppress_sec").value)
        self.fault_grace_sec = float(self.get_parameter("fault_grace_sec").value)

        max_pos_step = float(self.get_parameter("stable_max_position_step").value)
        max_yaw_step = float(self.get_parameter("stable_max_yaw_step").value)
        jump_pos = float(self.get_parameter("jump_position_threshold").value)
        jump_yaw = float(self.get_parameter("jump_yaw_threshold").value)

        self.slam = PoseTrack(
            "slam",
            int(self.get_parameter("stable_samples").value),
            max_pos_step,
            max_yaw_step,
            float(self.get_parameter("pose_timeout_sec").value),
            jump_pos,
            jump_yaw,
            float(self.get_parameter("slam_distrust_sec").value),
        )
        self.tag = PoseTrack(
            "tag",
            int(self.get_parameter("tag_stable_samples").value),
            max_pos_step,
            max_yaw_step,
            float(self.get_parameter("tag_timeout_sec").value),
            jump_pos,
            jump_yaw,
            0.0,  # tag poses are absolute; no distrust window needed
        )

        self.fused_pub = self.create_publisher(PoseStamped, fused_topic, 10)
        self.ok_pub = self.create_publisher(Bool, str(self.get_parameter("ok_topic").value), 10)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.stop_pub = self.create_publisher(Bool, str(self.get_parameter("stop_topic").value), 10)

        self.active: Optional[str] = None  # "slam" | "tag"
        self.switch_after = 0.0
        self.fault_since: Optional[float] = None
        self.ok = False
        self.last_reason = "waiting_for_pose"

        if pose_type in ("pose_stamped", "posestamped", "pose"):
            self.create_subscription(PoseStamped, pose_topic, self._on_slam_stamped, 10)
        elif pose_type in ("odometry", "odom"):
            self.create_subscription(Odometry, pose_topic, self._on_slam_odom, 10)
        else:
            raise ValueError("pose_type must be pose_stamped or odometry")
        if self.enable_tag_fallback:
            self.create_subscription(PoseStamped, tag_topic, self._on_tag_pose, 10)
        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"watching slam={pose_topic} ({pose_type}) tag={tag_topic if self.enable_tag_fallback else 'disabled'} "
            f"-> {fused_topic}"
        )

    # ------------------------------------------------------------------ subs
    def _on_slam_stamped(self, msg: PoseStamped) -> None:
        self._ingest(self.slam, msg.pose, msg)

    def _on_slam_odom(self, msg: Odometry) -> None:
        stamped = PoseStamped(header=msg.header, pose=msg.pose.pose)
        self._ingest(self.slam, msg.pose.pose, stamped)

    def _on_tag_pose(self, msg: PoseStamped) -> None:
        self._ingest(self.tag, msg.pose, msg)

    def _ingest(self, track: PoseTrack, pose: Pose, msg: PoseStamped) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        event = track.update(pose, msg, now)
        if event == "jump":
            self._log_warn.warning(f"{track.name} pose jump detected; distrusting {track.name} "
                                   f"for {track.distrust_sec:.1f}s")

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        slam_ok = self.slam.usable(now)
        tag_ok = self.tag.usable(now) if self.enable_tag_fallback else False

        # --- source arbitration ---
        if self.active is None:
            if slam_ok:
                self._switch("slam", now, "initial lock")
            elif tag_ok:
                self._switch("tag", now, "initial lock via tag")
        elif self.active == "slam":
            if not slam_ok and tag_ok:
                self._switch("tag", now, "slam lost -> AprilTag fallback")
        else:  # active == "tag"
            if tag_ok:
                if slam_ok and now >= self.switch_after:
                    self._switch("slam", now, "slam recovered")
            elif slam_ok:
                self._switch("slam", now, "tag lost -> back to slam")

        track = self.slam if self.active == "slam" else self.tag if self.active == "tag" else None

        # --- healthy: forward fused pose ---
        if track is not None and track.usable(now) and track.last_msg is not None:
            self.fused_pub.publish(track.last_msg)
            if self.fault_since is not None:
                self.fault_since = None
                self.stop_pub.publish(Bool(data=False))
                self.get_logger().info("localization recovered; motion stop released")
            self._publish_ok(True, f"stable:{self.active}")
            return

        # --- no usable source: grace window then hard fault ---
        if self.fault_since is None:
            self.fault_since = now
        elapsed = now - self.fault_since
        if elapsed < self.fault_grace_sec:
            # Keep ok=True briefly: navigator stops on stale fused pose anyway,
            # and a sub-grace gap must not make the FSM fail the whole run.
            self._publish_ok(True, "holding:source_gap")
            return
        self._set_fault(self._loss_reason(now, slam_ok, tag_ok))

    def _switch(self, source: str, now: float, reason: str) -> None:
        self.active = source
        self.switch_after = now + self.switch_suppress_sec
        self.get_logger().info(f"localization source -> {source} ({reason})")

    def _loss_reason(self, now: float, slam_ok: bool, tag_ok: bool) -> str:
        if self.slam.last_time is None and self.tag.last_time is None:
            return "waiting_for_pose"
        parts = []
        for track in (self.slam, self.tag):
            if track.last_time is None:
                parts.append(f"{track.name}:no_data")
            elif not track.fresh(now):
                parts.append(f"{track.name}:timeout age={now - track.last_time:.2f}s")
            elif now < track.distrusted_until:
                parts.append(f"{track.name}:jump_distrust")
            else:
                parts.append(f"{track.name}:stabilizing")
        return "all_sources_lost " + " ".join(parts)

    def _set_fault(self, reason: str) -> None:
        self.stop_pub.publish(Bool(data=True))
        self._publish_ok(False, reason)

    def _publish_ok(self, ok: bool, reason: str) -> None:
        if ok != self.ok or reason != self.last_reason:
            if ok:
                self.get_logger().info(f"localization ok={ok}: {reason}")
            else:
                self._log_warn.warning(f"localization ok={ok}: {reason}")
        self.ok = ok
        self.last_reason = reason
        self.ok_pub.publish(Bool(data=ok))
        self.status_pub.publish(String(data=reason))


def main() -> None:
    rclpy.init()
    node = LocalizationWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_pub.publish(Bool(data=True))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
