from __future__ import annotations

from cone_avoidance.local_planner import LocalPlanner, LocalPlannerState, RobotPose
from cone_avoidance.map_config import ObstacleZoneRect
from cone_avoidance.models import ConeObstacle, ControlConfig


def make_planner() -> LocalPlanner:
    return LocalPlanner(
        config=ControlConfig(
            normal_speed=0.15,
            slow_speed=0.08,
            max_turn_speed=0.25,
            min_depth_valid_ratio=0.35,
            min_realsense_fps=8.0,
        ),
        global_path=[(0.0, 0.0), (5.0, 0.0)],
        obstacle_zone_rect=ObstacleZoneRect(0.0, 5.0, -1.0, 1.0),
    )


def test_clear_path_follows_global_route() -> None:
    planner = make_planner()

    command = planner.plan(cones=[], robot_pose=RobotPose(0.2, 0.0, 0.0), aligned_depth_ok=True, depth_valid_ratio=0.8)

    assert command.vx > 0.0
    assert abs(command.wz) < 0.03
    assert "follow" in command.reason or "clear" in command.reason


def test_center_cone_left_side_available_turns_left() -> None:
    planner = make_planner()
    cones = [
        ConeObstacle(x=0.0, z=0.7, conf=0.9),
        ConeObstacle(x=-0.45, z=0.75, conf=0.9),
    ]

    command = planner.plan(cones=cones, robot_pose=RobotPose(0.2, 0.0, 0.0), aligned_depth_ok=True, depth_valid_ratio=0.8)

    assert command.vx > 0.0
    assert command.wz > 0.0
    assert command.state == LocalPlannerState.LOCAL_AVOID.value


def test_center_cone_right_side_available_turns_right() -> None:
    planner = make_planner()
    cones = [
        ConeObstacle(x=0.0, z=0.7, conf=0.9),
        ConeObstacle(x=0.45, z=0.75, conf=0.9),
    ]

    command = planner.plan(cones=cones, robot_pose=RobotPose(0.2, 0.0, 0.0), aligned_depth_ok=True, depth_valid_ratio=0.8)

    assert command.vx > 0.0
    assert command.wz < 0.0
    assert command.state == LocalPlannerState.LOCAL_AVOID.value


def test_too_close_cone_does_not_drive_fast() -> None:
    planner = make_planner()

    command = planner.plan(
        cones=[ConeObstacle(x=0.0, z=0.35, conf=0.95)],
        robot_pose=RobotPose(0.2, 0.0, 0.0),
        aligned_depth_ok=True,
        depth_valid_ratio=0.8,
    )

    assert command.vx <= 0.04
    assert command.state == LocalPlannerState.RECOVER_STOP.value


def test_candidate_crossing_zone_boundary_is_rejected() -> None:
    planner = make_planner()
    cones = [
        ConeObstacle(x=0.0, z=0.7, conf=0.9),
        ConeObstacle(x=-0.45, z=0.75, conf=0.9),
    ]

    command = planner.plan(cones=cones, robot_pose=RobotPose(0.2, 0.92, 0.0), aligned_depth_ok=True, depth_valid_ratio=0.8)

    assert command.wz <= 0.0
    assert "left" not in command.reason


def test_without_pose_still_uses_local_obstacle_avoidance() -> None:
    planner = make_planner()
    cones = [
        ConeObstacle(x=0.0, z=0.7, conf=0.9),
        ConeObstacle(x=-0.45, z=0.75, conf=0.9),
    ]

    command = planner.plan(cones=cones, robot_pose=None, aligned_depth_ok=True, depth_valid_ratio=0.8)

    assert command.vx > 0.0
    assert command.wz > 0.0
    assert command.state == LocalPlannerState.LOCAL_AVOID.value


def test_rejoin_after_obstacle_clears_for_several_frames() -> None:
    planner = make_planner()
    planner.plan(
        cones=[ConeObstacle(x=0.0, z=0.7, conf=0.9)],
        robot_pose=RobotPose(0.2, 0.35, 0.0),
        aligned_depth_ok=True,
        depth_valid_ratio=0.8,
    )

    for _ in range(3):
        command = planner.plan(cones=[], robot_pose=RobotPose(0.4, 0.35, 0.0), aligned_depth_ok=True, depth_valid_ratio=0.8)

    assert command.state == LocalPlannerState.REJOIN_GLOBAL_PATH.value
    assert "rejoin" in command.reason


def test_bad_depth_inputs_stop_safely() -> None:
    planner = make_planner()

    command = planner.plan(cones=[], aligned_depth_ok=False, depth_valid_ratio=0.8)
    assert command.vx == 0.0
    assert command.state == LocalPlannerState.RECOVER_STOP.value
    assert command.reason == "aligned_depth_unavailable"

    command = planner.plan(cones=[], aligned_depth_ok=True, depth_valid_ratio=0.1)
    assert command.vx == 0.0
    assert command.reason == "depth_valid_ratio_low"
