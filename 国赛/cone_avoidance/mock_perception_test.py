from __future__ import annotations

import math
import time
from typing import Iterable

from .avoidance_state_machine import AvoidanceStateMachine
from .models import ConeObstacle, ControlConfig, VelocityCommand


def _run_case(
    name: str,
    obstacles: Iterable[ConeObstacle] | None,
    expected_reason: str,
    now: float,
    front_depth: float | None = None,
    **quality: object,
) -> VelocityCommand:
    machine = AvoidanceStateMachine(ControlConfig(turn_smoothing_alpha=1.0))
    machine.start(now)
    command = machine.tick(obstacles, now=now, front_depth=front_depth, **quality)
    print(f"{name}: {command.log_line()}")
    assert command.reason == expected_reason, (name, command.reason, expected_reason)
    return command


def main() -> None:
    now = time.monotonic()
    fresh = now

    _run_case("clear", [], "clear_forward", now)
    _run_case("left_cone", [ConeObstacle(x=0.35, z=1.0, conf=0.9, last_seen=fresh)], "avoid_left_cone", now)
    _run_case("right_cone", [ConeObstacle(x=-0.35, z=1.0, conf=0.9, last_seen=fresh)], "avoid_right_cone", now)
    _run_case("front_close", [ConeObstacle(x=0.02, z=0.55, conf=0.9, last_seen=fresh)], "emergency_stop", now)
    _run_case(
        "two_cones_gap",
        [
            ConeObstacle(x=0.75, z=1.0, conf=0.9, last_seen=fresh),
            ConeObstacle(x=-0.75, z=1.0, conf=0.9, last_seen=fresh),
        ],
        "pass_between_cones",
        now,
    )
    _run_case(
        "two_cones_no_gap",
        [
            ConeObstacle(x=0.25, z=1.0, conf=0.9, last_seen=fresh),
            ConeObstacle(x=-0.25, z=1.0, conf=0.9, last_seen=fresh),
        ],
        "two_cones_outside_right",
        now,
    )
    _run_case("front_depth_stop", [], "front_depth_emergency_stop", now, front_depth=0.4)
    _run_case("aligned_depth_missing", [], "aligned_depth_unavailable", now, aligned_depth_ok=False)
    _run_case("depth_ratio_low", [], "depth_valid_ratio_low", now, depth_valid_ratio=0.1)
    _run_case("realsense_offline", [], "realsense_unavailable", now, realsense_ok=False)
    _run_case("realsense_fps_low", [], "realsense_fps_low", now, realsense_fps=3.0)
    _run_case("low_confidence", [ConeObstacle(x=0.2, z=1.0, conf=0.2, last_seen=fresh)], "low_confidence_observe", now)
    _run_case("invalid_nan", [ConeObstacle(x=math.nan, z=1.0, conf=0.9, last_seen=fresh)], "invalid_obstacle", now)
    _run_case("invalid_none", [ConeObstacle(x=None, z=1.0, conf=0.9, last_seen=fresh)], "invalid_obstacle", now)
    _run_case("invalid_inf", [ConeObstacle(x=0.1, z=math.inf, conf=0.9, last_seen=fresh)], "invalid_obstacle", now)
    _run_case(
        "stale_last_seen",
        [ConeObstacle(x=0.1, z=1.0, conf=0.9, last_seen=now - 1.0)],
        "perception_timeout",
        now,
    )
    _run_case(
        "stale_epoch_last_seen",
        [ConeObstacle(x=0.1, z=1.0, conf=0.9, last_seen=time.time() - 1.0)],
        "perception_timeout",
        now,
    )

    timeout_machine = AvoidanceStateMachine(ControlConfig(turn_smoothing_alpha=1.0))
    timeout_machine.start(now)
    timeout_machine.tick([], now=now)
    command = timeout_machine.tick(None, now=now + 0.6)
    print(f"packet_timeout: {command.log_line()}")
    assert command.reason == "perception_timeout"

    print("all mock cases passed")


if __name__ == "__main__":
    main()
