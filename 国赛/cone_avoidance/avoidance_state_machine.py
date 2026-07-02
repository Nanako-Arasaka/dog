from __future__ import annotations

import time
from enum import Enum
from typing import Iterable

from .avoidance_policy import AvoidancePolicy
from .models import ConeObstacle, ControlConfig, VelocityCommand
from .safety_guard import SafetyGuard


class AvoidanceState(str, Enum):
    IDLE = "IDLE"
    ENTER_OBSTACLE_AREA = "ENTER_OBSTACLE_AREA"
    TRACK_AND_AVOID = "TRACK_AND_AVOID"
    RECOVER_STOP = "RECOVER_STOP"
    EXIT_OBSTACLE_AREA = "EXIT_OBSTACLE_AREA"
    DONE = "DONE"


class AvoidanceStateMachine:
    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self.policy = AvoidancePolicy(self.config)
        self.guard = SafetyGuard(self.config)
        self.state = AvoidanceState.IDLE
        self._start_time: float | None = None
        self._last_input_time: float | None = None
        self._recover_until: float | None = None
        self._clear_since: float | None = None
        self._exit_until: float | None = None

    def start(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.state = AvoidanceState.ENTER_OBSTACLE_AREA
        self._start_time = now
        self._last_input_time = now
        self._recover_until = None
        self._clear_since = None
        self._exit_until = None

    def stop(self) -> None:
        self.state = AvoidanceState.IDLE

    def tick(
        self,
        obstacles: Iterable[ConeObstacle] | None,
        now: float | None = None,
        front_depth: float | None = None,
        aligned_depth_ok: bool | None = None,
        depth_valid_ratio: float | None = None,
        realsense_fps: float | None = None,
        realsense_ok: bool | None = None,
    ) -> VelocityCommand:
        now = time.monotonic() if now is None else now
        if self.state == AvoidanceState.IDLE:
            return VelocityCommand.stop("idle", self.state.value)
        if self.state == AvoidanceState.DONE:
            return VelocityCommand.stop("done", self.state.value)

        if obstacles is None:
            if self._timed_out(now):
                return self._enter_recover(now, "perception_timeout")
            obstacles = []
        else:
            self._last_input_time = now

        if self.state == AvoidanceState.RECOVER_STOP:
            if self._recover_until is not None and now < self._recover_until:
                return VelocityCommand.stop("recover_stop", self.state.value)
            self.state = AvoidanceState.TRACK_AND_AVOID

        if self.state == AvoidanceState.EXIT_OBSTACLE_AREA:
            if self._exit_until is not None and now >= self._exit_until:
                self.state = AvoidanceState.DONE
                return VelocityCommand.stop("done", self.state.value)
            return VelocityCommand(
                vx=self.config.slow_speed,
                vy=0.0,
                wz=0.0,
                reason="exit_obstacle_area",
                state=self.state.value,
            )

        valid, stop = self.guard.validate_inputs(
            obstacles,
            now=now,
            front_depth=front_depth,
            aligned_depth_ok=aligned_depth_ok,
            depth_valid_ratio=depth_valid_ratio,
            realsense_fps=realsense_fps,
            realsense_ok=realsense_ok,
        )
        if stop is not None:
            return self._enter_recover(now, stop.reason)

        command = self.guard.sanitize_command(self.policy.decide(valid))
        if command.reason in {
            "invalid_output",
            "output_limit_exceeded",
            "emergency_stop",
            "front_depth_emergency_stop",
            "realsense_unavailable",
            "aligned_depth_unavailable",
            "realsense_fps_low",
            "depth_valid_ratio_low",
        }:
            return self._enter_recover(now, command.reason)

        if self.state == AvoidanceState.ENTER_OBSTACLE_AREA:
            self.state = AvoidanceState.TRACK_AND_AVOID

        self._update_clear_progress(valid, command, now)
        return command.with_state(self.state.value)

    def _timed_out(self, now: float) -> bool:
        if self._last_input_time is None:
            return True
        return now - self._last_input_time > self.config.perception_timeout

    def _enter_recover(self, now: float, reason: str) -> VelocityCommand:
        self.state = AvoidanceState.RECOVER_STOP
        self._recover_until = now + self.config.recover_stop_seconds
        self._clear_since = None
        return VelocityCommand.stop(reason, self.state.value)

    def _update_clear_progress(
        self,
        obstacles: list[ConeObstacle],
        command: VelocityCommand,
        now: float,
    ) -> None:
        if self._start_time is None:
            self._start_time = now
        front_clear = not obstacles and command.reason == "clear_forward"
        if front_clear:
            if self._clear_since is None:
                self._clear_since = now
        else:
            self._clear_since = None

        enough_time = now - self._start_time >= self.config.min_run_seconds
        clear_long_enough = self._clear_since is not None and now - self._clear_since >= self.config.clear_done_seconds
        if enough_time and clear_long_enough:
            self.state = AvoidanceState.EXIT_OBSTACLE_AREA
            self._exit_until = now + self.config.exit_seconds
