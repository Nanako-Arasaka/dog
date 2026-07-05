from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .map_config import ObstacleZoneRect
from .models import ConeObstacle, ControlConfig, VelocityCommand


class LocalPlannerState(str, Enum):
    FOLLOW_GLOBAL_PATH = "FOLLOW_GLOBAL_PATH"
    LOCAL_AVOID = "LOCAL_AVOID"
    REJOIN_GLOBAL_PATH = "REJOIN_GLOBAL_PATH"
    RECOVER_STOP = "RECOVER_STOP"


@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    yaw: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "RobotPose | None":
        if not data:
            return None
        return cls(float(data.get("x", 0.0)), float(data.get("y", 0.0)), float(data.get("yaw", 0.0)))


@dataclass(frozen=True)
class CandidateCommand:
    name: str
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class CandidateScore:
    command: CandidateCommand
    score: float
    min_cone_distance: float
    end_path_distance: float
    progress: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class LocalPlanner:
    """Small local planner for depth-cone avoidance and global-path rejoin."""

    def __init__(
        self,
        config: ControlConfig | None = None,
        global_path: Sequence[tuple[float, float]] | None = None,
        obstacle_zone_rect: ObstacleZoneRect | None = None,
    ) -> None:
        self.config = config or ControlConfig()
        self.global_path = list(global_path or [(0.0, 0.0), (5.0, 0.0)])
        self.obstacle_zone_rect = obstacle_zone_rect or ObstacleZoneRect(0.0, 5.0, -1.5, 1.5)
        self.state = LocalPlannerState.FOLLOW_GLOBAL_PATH
        self._last_avoid_sign = 0
        self._same_side_frames = 0
        self._clear_frames = 0

    def plan(
        self,
        cones: Iterable[ConeObstacle] | None = None,
        robot_pose: RobotPose | Mapping[str, object] | None = None,
        global_path: Sequence[tuple[float, float]] | None = None,
        obstacle_zone_rect: ObstacleZoneRect | Mapping[str, object] | None = None,
        front_depth: float | None = None,
        depth_valid_ratio: float | None = None,
        aligned_depth_ok: bool | None = None,
        realsense_fps: float | None = None,
        realsense_ok: bool | None = None,
    ) -> VelocityCommand:
        if global_path is not None:
            self.global_path = list(global_path)
        if obstacle_zone_rect is not None:
            self.obstacle_zone_rect = (
                ObstacleZoneRect.from_mapping(obstacle_zone_rect)
                if isinstance(obstacle_zone_rect, Mapping)
                else obstacle_zone_rect
            )
        pose = RobotPose.from_mapping(robot_pose) if isinstance(robot_pose, Mapping) else robot_pose

        safety_stop = self._safety_stop(
            front_depth=front_depth,
            depth_valid_ratio=depth_valid_ratio,
            aligned_depth_ok=aligned_depth_ok,
            realsense_fps=realsense_fps,
            realsense_ok=realsense_ok,
        )
        if safety_stop is not None:
            self.state = LocalPlannerState.RECOVER_STOP
            return safety_stop

        valid_cones = self._valid_cones(cones or [])
        near_cones = [cone for cone in valid_cones if float(cone.z) <= self.config.slow_distance]
        self._update_state(near_cones, pose)

        if self._too_close(near_cones, front_depth):
            self.state = LocalPlannerState.RECOVER_STOP
            return VelocityCommand(
                vx=min(self.config.slow_speed, 0.04),
                vy=0.0,
                wz=0.0,
                reason="local_too_close_crawl",
                state=self.state.value,
            )

        scores = self._score_candidates(valid_cones, pose)
        if not scores:
            self.state = LocalPlannerState.RECOVER_STOP
            return VelocityCommand.stop("local_no_valid_trajectory", self.state.value)

        best = max(scores, key=lambda item: item.score)
        self._track_turn_side(best.command.wz)
        reason_prefix = {
            LocalPlannerState.FOLLOW_GLOBAL_PATH: "follow_global_path",
            LocalPlannerState.LOCAL_AVOID: "local_avoid",
            LocalPlannerState.REJOIN_GLOBAL_PATH: "rejoin_global_path",
            LocalPlannerState.RECOVER_STOP: "recover",
        }[self.state]
        return VelocityCommand(
            vx=best.command.vx,
            vy=best.command.vy,
            wz=best.command.wz,
            reason=f"{reason_prefix}:{best.command.name}",
            state=self.state.value,
        )

    def _safety_stop(
        self,
        front_depth: float | None,
        depth_valid_ratio: float | None,
        aligned_depth_ok: bool | None,
        realsense_fps: float | None,
        realsense_ok: bool | None,
    ) -> VelocityCommand | None:
        cfg = self.config
        if realsense_ok is False:
            return VelocityCommand.stop("realsense_unavailable", LocalPlannerState.RECOVER_STOP.value)
        if aligned_depth_ok is False:
            return VelocityCommand.stop("aligned_depth_unavailable", LocalPlannerState.RECOVER_STOP.value)
        if depth_valid_ratio is not None and float(depth_valid_ratio) < cfg.min_depth_valid_ratio:
            return VelocityCommand.stop("depth_valid_ratio_low", LocalPlannerState.RECOVER_STOP.value)
        if realsense_fps is not None and float(realsense_fps) < cfg.min_realsense_fps:
            return VelocityCommand.stop("realsense_fps_low", LocalPlannerState.RECOVER_STOP.value)
        if front_depth is not None and float(front_depth) < cfg.front_emergency_distance:
            return VelocityCommand.stop("front_depth_emergency_stop", LocalPlannerState.RECOVER_STOP.value)
        return None

    def _valid_cones(self, cones: Iterable[ConeObstacle]) -> list[ConeObstacle]:
        valid = []
        for cone in cones:
            if cone.x is None or cone.z is None or cone.conf < self.config.min_confidence:
                continue
            if not all(math.isfinite(float(value)) for value in (cone.x, cone.z, cone.conf)):
                continue
            if float(cone.z) <= 0.0:
                continue
            valid.append(cone)
        return valid

    def _update_state(self, near_cones: list[ConeObstacle], pose: RobotPose | None) -> None:
        if near_cones:
            self.state = LocalPlannerState.LOCAL_AVOID
            self._clear_frames = 0
            return

        self._clear_frames += 1
        if self.state == LocalPlannerState.LOCAL_AVOID and self._clear_frames >= 3:
            self.state = LocalPlannerState.REJOIN_GLOBAL_PATH
            return
        if self.state == LocalPlannerState.REJOIN_GLOBAL_PATH and pose is not None:
            if self._distance_to_path(pose.x, pose.y) <= 0.15:
                self.state = LocalPlannerState.FOLLOW_GLOBAL_PATH
            return
        if self.state == LocalPlannerState.RECOVER_STOP:
            self.state = LocalPlannerState.REJOIN_GLOBAL_PATH if pose is not None else LocalPlannerState.FOLLOW_GLOBAL_PATH

    def _too_close(self, cones: list[ConeObstacle], front_depth: float | None) -> bool:
        if front_depth is not None and float(front_depth) <= self.config.stop_distance:
            return True
        for cone in cones:
            if abs(float(cone.x)) <= self.config.front_emergency_width and float(cone.z) <= self.config.stop_distance:
                return True
        return False

    def _score_candidates(self, cones: list[ConeObstacle], pose: RobotPose | None) -> list[CandidateScore]:
        candidates = self._candidate_commands()
        start_progress = self._path_progress(pose.x, pose.y) if pose is not None else 0.0
        scored: list[CandidateScore] = []
        for command in candidates:
            rollout = self._rollout(command)
            min_cone_distance = self._min_cone_distance(rollout, cones)
            if min_cone_distance < 0.36:
                continue
            end_lat, end_fwd, _ = rollout[-1]
            if pose is not None:
                end_x, end_y = self._local_to_map(pose, end_fwd, end_lat)
                if not self.obstacle_zone_rect.contains(end_x, end_y, margin=0.10):
                    continue
                path_distance = self._distance_to_path(end_x, end_y)
                progress = self._path_progress(end_x, end_y) - start_progress
            else:
                path_distance = abs(end_lat)
                progress = end_fwd

            score = 0.0
            score += progress * 3.0
            score -= path_distance * 1.6
            score -= abs(command.wz) * 0.55
            score -= abs(command.vy) * 0.35
            score += min(min_cone_distance, 1.2) * 0.35
            if not cones and command.name == "straight":
                score += 2.0
            if self.state == LocalPlannerState.REJOIN_GLOBAL_PATH:
                score -= abs(end_lat) * 1.2
            if self._same_side_frames >= 8 and self._turn_sign(command.wz) == self._last_avoid_sign:
                score -= 1.0
            scored.append(CandidateScore(command, score, min_cone_distance, path_distance, progress))
        return scored

    def _candidate_commands(self) -> list[CandidateCommand]:
        cfg = self.config
        turn = cfg.max_turn_speed
        return [
            CandidateCommand("straight", cfg.normal_speed, 0.0, 0.0),
            CandidateCommand("left_light", cfg.slow_speed, 0.05, turn * 0.55),
            CandidateCommand("left_medium", cfg.slow_speed, 0.10, turn),
            CandidateCommand("right_light", cfg.slow_speed, -0.05, -turn * 0.55),
            CandidateCommand("right_medium", cfg.slow_speed, -0.10, -turn),
            CandidateCommand("crawl", min(cfg.slow_speed, 0.04), 0.0, 0.0),
        ]

    def _rollout(self, command: CandidateCommand) -> list[tuple[float, float, float]]:
        dt = 0.25
        steps = int(math.ceil(3.0 / dt))
        heading = 0.0
        lateral = 0.0
        forward = 0.0
        points: list[tuple[float, float, float]] = []
        for _ in range(steps):
            heading += command.wz * dt
            forward += command.vx * math.cos(heading) * dt
            lateral += (command.vy + command.vx * math.sin(heading)) * dt
            points.append((lateral, forward, heading))
        return points

    def _min_cone_distance(self, rollout: list[tuple[float, float, float]], cones: list[ConeObstacle]) -> float:
        if not cones:
            return 99.0
        min_distance = 99.0
        for lateral, forward, _ in rollout:
            for cone in cones:
                distance = math.hypot(lateral - float(cone.x), forward - float(cone.z))
                min_distance = min(min_distance, distance)
        end_lateral, end_forward, _ = rollout[-1]
        if end_forward > 0.03:
            slope = end_lateral / end_forward
            for cone in cones:
                cone_forward = float(cone.z)
                if end_forward < cone_forward <= self.config.slow_distance + 0.2:
                    projected_lateral = slope * cone_forward
                    distance = abs(projected_lateral - float(cone.x))
                    min_distance = min(min_distance, distance)
        return min_distance

    def _local_to_map(self, pose: RobotPose, forward: float, lateral: float) -> tuple[float, float]:
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        return (
            pose.x + forward * cos_yaw - lateral * sin_yaw,
            pose.y + forward * sin_yaw + lateral * cos_yaw,
        )

    def _distance_to_path(self, x: float, y: float) -> float:
        return self._path_metrics(x, y)[0]

    def _path_progress(self, x: float, y: float) -> float:
        return self._path_metrics(x, y)[1]

    def _path_metrics(self, x: float, y: float) -> tuple[float, float]:
        if len(self.global_path) < 2:
            px, py = self.global_path[0] if self.global_path else (0.0, 0.0)
            return math.hypot(x - px, y - py), 0.0

        best_distance = 99.0
        best_progress = 0.0
        cumulative = 0.0
        for start, end in zip(self.global_path, self.global_path[1:]):
            sx, sy = start
            ex, ey = end
            dx = ex - sx
            dy = ey - sy
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-9:
                continue
            t = _clamp(((x - sx) * dx + (y - sy) * dy) / length_sq, 0.0, 1.0)
            px = sx + t * dx
            py = sy + t * dy
            distance = math.hypot(x - px, y - py)
            segment_length = math.sqrt(length_sq)
            if distance < best_distance:
                best_distance = distance
                best_progress = cumulative + t * segment_length
            cumulative += segment_length
        return best_distance, best_progress

    def _track_turn_side(self, wz: float) -> None:
        sign = self._turn_sign(wz)
        if sign == 0:
            self._same_side_frames = 0
            self._last_avoid_sign = 0
        elif sign == self._last_avoid_sign:
            self._same_side_frames += 1
        else:
            self._last_avoid_sign = sign
            self._same_side_frames = 1

    @staticmethod
    def _turn_sign(wz: float) -> int:
        if wz > 0.02:
            return 1
        if wz < -0.02:
            return -1
        return 0
