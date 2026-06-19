#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based cone avoidance from YOLO bounding boxes.

YOLO only detects where the cone is. This module decides a conservative
vx/vy/wz command from bbox position and size, so it can be tested without a
camera, model, or robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ConeDetection:
    """Single cone detection in pixel coordinates."""

    xyxy: Box
    confidence: float = 1.0
    class_name: str = "cone"

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.xyxy
        return (x1 + x2) * 0.5

    @property
    def center_y(self) -> float:
        _, y1, _, y2 = self.xyxy
        return (y1 + y2) * 0.5

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class AvoidanceConfig:
    """Tunable parameters for the obstacle-zone rule controller."""

    min_confidence: float = 0.35
    center_left_ratio: float = 0.36
    center_right_ratio: float = 0.64
    near_area_ratio: float = 0.08
    stop_area_ratio: float = 0.20
    forward_speed: float = 0.16
    slow_speed: float = 0.08
    avoid_turn_speed: float = 0.28
    emergency_turn_speed: float = 0.38


@dataclass(frozen=True)
class MotionDecision:
    """Normalized velocity decision sent to the Lite2 UDP receiver."""

    vx: float
    vy: float
    wz: float
    state: str
    reason: str
    target_box: Optional[Box] = None

    def to_payload(self) -> dict:
        return {
            "vx": round(float(self.vx), 4),
            "vy": round(float(self.vy), 4),
            "wz": round(float(self.wz), 4),
            "state": self.state,
            "reason": self.reason,
        }


def plan_cone_avoidance(
    detections: Iterable[ConeDetection],
    frame_size: Sequence[int],
    config: AvoidanceConfig = AvoidanceConfig(),
) -> MotionDecision:
    """Return a simple obstacle-zone velocity command.

    `frame_size` accepts `(width, height)` or `(height, width, channels)`.
    Positive `wz` means turn left in the usual ROS convention. If the dog turns
    the opposite way on site, use the receiver-side reverse parameter instead of
    changing this planner first.
    """

    frame_width, frame_height = _normalize_frame_size(frame_size)
    frame_area = max(1.0, frame_width * frame_height)
    valid = [
        det
        for det in detections
        if det.confidence >= config.min_confidence and det.area > 1.0
    ]

    if not valid:
        return MotionDecision(
            vx=config.forward_speed,
            vy=0.0,
            wz=0.0,
            state="clear",
            reason="no cone detected",
        )

    nearest = max(valid, key=lambda item: item.area)
    nearest_area_ratio = nearest.area / frame_area
    center_left = frame_width * config.center_left_ratio
    center_right = frame_width * config.center_right_ratio

    if nearest_area_ratio >= config.stop_area_ratio:
        turn = _free_side_turn(valid, frame_width, config.emergency_turn_speed)
        return MotionDecision(
            vx=0.0,
            vy=0.0,
            wz=turn,
            state="too_close",
            reason=f"largest cone area ratio {nearest_area_ratio:.3f}",
            target_box=nearest.xyxy,
        )

    center_blocked = any(center_left <= det.center_x <= center_right for det in valid)
    if center_blocked or nearest_area_ratio >= config.near_area_ratio:
        turn = _free_side_turn(valid, frame_width, config.avoid_turn_speed)
        return MotionDecision(
            vx=config.slow_speed,
            vy=0.0,
            wz=turn,
            state="avoid",
            reason="center path blocked or cone is near",
            target_box=nearest.xyxy,
        )

    if nearest.center_x < frame_width * 0.5:
        return MotionDecision(
            vx=config.forward_speed,
            vy=0.0,
            wz=-config.avoid_turn_speed * 0.6,
            state="bias_right",
            reason="cone on left",
            target_box=nearest.xyxy,
        )

    return MotionDecision(
        vx=config.forward_speed,
        vy=0.0,
        wz=config.avoid_turn_speed * 0.6,
        state="bias_left",
        reason="cone on right",
        target_box=nearest.xyxy,
    )


def _free_side_turn(detections: List[ConeDetection], frame_width: float, turn_speed: float) -> float:
    left_weight = 0.0
    right_weight = 0.0
    for det in detections:
        if det.center_x < frame_width * 0.5:
            left_weight += det.area
        else:
            right_weight += det.area
    return turn_speed if left_weight <= right_weight else -turn_speed


def _normalize_frame_size(frame_size: Sequence[int]) -> Tuple[float, float]:
    if len(frame_size) >= 3:
        height = float(frame_size[0])
        width = float(frame_size[1])
        return width, height
    if len(frame_size) == 2:
        width = float(frame_size[0])
        height = float(frame_size[1])
        return width, height
    raise ValueError("frame_size must be (width, height) or image.shape")

