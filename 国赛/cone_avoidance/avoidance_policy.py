from __future__ import annotations

import math
from typing import Iterable, List

from .models import ConeObstacle, ControlConfig, VelocityCommand


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class AvoidancePolicy:
    """Local reactive policy: ConeObstacle list -> conservative vx/vy/wz."""

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self._last_turn_sign = 1.0
        self._last_wz = 0.0

    def decide(self, obstacles: Iterable[ConeObstacle]) -> VelocityCommand:
        cfg = self.config
        near = [
            cone
            for cone in obstacles
            if cone.z is not None
            and cone.x is not None
            and 0.0 < cone.z <= cfg.slow_distance
        ]
        low_conf = [cone for cone in near if cone.conf < cfg.min_confidence]
        front = [cone for cone in near if cone.conf >= cfg.min_confidence]
        front.sort(key=lambda cone: float(cone.z))
        low_conf.sort(key=lambda cone: float(cone.z))

        emergency = self._front_emergency(front)
        if emergency is not None:
            return emergency

        if not front and low_conf:
            return self._smooth(VelocityCommand(cfg.slow_speed, 0.0, 0.0, "low_confidence_observe"))

        if not front:
            return self._smooth(VelocityCommand(cfg.normal_speed, 0.0, 0.0, "clear_forward"))

        if len(front) >= 2:
            return self._decide_two_cones(front[0], front[1])

        return self._decide_single_cone(front[0])

    def _front_emergency(self, obstacles: List[ConeObstacle]) -> VelocityCommand | None:
        cfg = self.config
        for cone in obstacles:
            assert cone.x is not None and cone.z is not None
            if abs(cone.x) <= cfg.front_emergency_width and cone.z <= cfg.stop_distance:
                return VelocityCommand.stop("emergency_stop")
        return None

    def _decide_single_cone(self, cone: ConeObstacle) -> VelocityCommand:
        cfg = self.config
        assert cone.x is not None
        if abs(cone.x) <= cfg.center_deadband:
            turn_sign = -self._last_turn_sign
            reason = "avoid_center_cone"
        elif cone.x > 0.0:
            turn_sign = -1.0
            reason = "avoid_left_cone"
        else:
            turn_sign = 1.0
            reason = "avoid_right_cone"

        self._last_turn_sign = turn_sign
        return self._smooth(
            VelocityCommand(
                vx=cfg.slow_speed,
                vy=0.0,
                wz=turn_sign * cfg.max_turn_speed,
                reason=reason,
            )
        )

    def _decide_two_cones(self, cone_a: ConeObstacle, cone_b: ConeObstacle) -> VelocityCommand:
        cfg = self.config
        assert cone_a.x is not None and cone_b.x is not None
        left, right = sorted((cone_a, cone_b), key=lambda cone: float(cone.x), reverse=True)
        lateral_gap = abs(float(left.x) - float(right.x))

        if lateral_gap > cfg.gap_pass_width:
            gap_center = (float(left.x) + float(right.x)) / 2.0
            wz = _clamp(gap_center * cfg.max_turn_speed, -cfg.max_turn_speed, cfg.max_turn_speed)
            if abs(wz) < 0.03:
                wz = 0.0
            elif wz > 0:
                self._last_turn_sign = 1.0
            else:
                self._last_turn_sign = -1.0
            return self._smooth(VelocityCommand(cfg.slow_speed, 0.0, wz, "pass_between_cones"))

        midpoint = (float(left.x) + float(right.x)) / 2.0
        if midpoint >= 0.0:
            turn_sign = -1.0
            reason = "two_cones_outside_right"
        else:
            turn_sign = 1.0
            reason = "two_cones_outside_left"
        self._last_turn_sign = turn_sign
        return self._smooth(VelocityCommand(cfg.slow_speed, 0.0, turn_sign * cfg.max_turn_speed, reason))

    def _smooth(self, command: VelocityCommand) -> VelocityCommand:
        cfg = self.config
        if not math.isfinite(command.wz):
            return VelocityCommand.stop("invalid_output")
        alpha = _clamp(cfg.turn_smoothing_alpha, 0.0, 1.0)
        wz = alpha * command.wz + (1.0 - alpha) * self._last_wz
        self._last_wz = wz
        return VelocityCommand(command.vx, command.vy, wz, command.reason, command.state, command.source)
