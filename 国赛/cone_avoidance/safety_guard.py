from __future__ import annotations

import math
import time
from typing import Iterable, List, Tuple

from .models import ConeObstacle, ControlConfig, VelocityCommand


class SafetyGuard:
    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self._last_close_obstacle_time: float | None = None

    def validate_inputs(
        self,
        obstacles: Iterable[ConeObstacle],
        now: float | None = None,
        front_depth: float | None = None,
        aligned_depth_ok: bool | None = None,
        depth_valid_ratio: float | None = None,
        realsense_fps: float | None = None,
        realsense_ok: bool | None = None,
    ) -> Tuple[List[ConeObstacle], VelocityCommand | None]:
        now = time.monotonic() if now is None else now
        cfg = self.config

        if realsense_ok is False:
            return [], VelocityCommand.stop("realsense_unavailable")
        if aligned_depth_ok is False:
            return [], VelocityCommand.stop("aligned_depth_unavailable")
        if realsense_fps is not None:
            if not math.isfinite(float(realsense_fps)):
                return [], VelocityCommand.stop("invalid_realsense_fps")
            if float(realsense_fps) < cfg.min_realsense_fps:
                return [], VelocityCommand.stop("realsense_fps_low")
        if depth_valid_ratio is not None:
            if not math.isfinite(float(depth_valid_ratio)):
                return [], VelocityCommand.stop("invalid_depth_valid_ratio")
            if float(depth_valid_ratio) < cfg.min_depth_valid_ratio:
                return [], VelocityCommand.stop("depth_valid_ratio_low")

        if front_depth is not None:
            if not math.isfinite(float(front_depth)):
                return [], VelocityCommand.stop("invalid_front_depth")
            if float(front_depth) < cfg.front_emergency_distance:
                return [], VelocityCommand.stop("front_depth_emergency_stop")

        valid: List[ConeObstacle] = []
        wall_now = time.time()
        for cone in obstacles:
            error = self._validate_cone(cone, now, wall_now)
            if error is not None:
                return [], VelocityCommand.stop(error)
            valid.append(cone)
            assert cone.z is not None
            if cone.z <= cfg.slow_distance:
                self._last_close_obstacle_time = now

        if not valid and self._last_close_obstacle_time is not None:
            if now - self._last_close_obstacle_time <= cfg.perception_timeout:
                return [], VelocityCommand.stop("close_obstacle_lost")

        return valid, None

    def sanitize_command(self, command: VelocityCommand) -> VelocityCommand:
        cfg = self.config
        values = (command.vx, command.vy, command.wz)
        if any(value is None or not math.isfinite(float(value)) for value in values):
            return VelocityCommand.stop("invalid_output")

        if abs(command.vx) > cfg.normal_speed or abs(command.vy) > cfg.normal_speed:
            return VelocityCommand.stop("output_limit_exceeded")
        if abs(command.wz) > cfg.max_turn_speed:
            return VelocityCommand.stop("output_limit_exceeded")

        return command

    def _validate_cone(self, cone: ConeObstacle, now: float, wall_now: float) -> str | None:
        cfg = self.config
        if cone.x is None or cone.z is None:
            return "invalid_obstacle"
        if not math.isfinite(float(cone.x)) or not math.isfinite(float(cone.z)):
            return "invalid_obstacle"
        if not math.isfinite(float(cone.conf)):
            return "invalid_obstacle"
        if cone.conf < 0.0 or cone.conf > 1.0:
            return "invalid_obstacle"
        if abs(float(cone.x)) > cfg.max_abs_x or float(cone.z) <= 0.0 or float(cone.z) > cfg.max_z:
            return "invalid_obstacle"
        if cone.last_seen is not None:
            stamp = float(cone.last_seen)
            reference_now = wall_now if stamp > 10_000_000.0 else now
            if reference_now - stamp > cfg.perception_timeout:
                return "perception_timeout"
        return None
