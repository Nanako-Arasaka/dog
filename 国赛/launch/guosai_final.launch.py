#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-launch final competition flow."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _expand(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value.replace("${GUOSAI_ROOT}", str(root)))
    return value


def _bool_text(value: Any) -> str:
    return "true" if _as_bool(str(value)) else "false"


def _py_node(root: Path, rel_path: str, params: dict, name: str, on_exit=None) -> ExecuteProcess:
    cmd = ["python3", str(root / rel_path)]
    if params:
        cmd += ["--ros-args"]
        for key, value in params.items():
            cmd += ["-p", f"{key}:={value}"]
    return ExecuteProcess(
        cmd=cmd,
        cwd=str(root),
        name=name,
        output="screen",
        on_exit=on_exit,
    )


def _shell_process(root: Path, command: str, name: str) -> ExecuteProcess:
    return ExecuteProcess(
        cmd=["bash", "-lc", f"cd {shlex.quote(str(root))} && {command}"],
        cwd=str(root),
        name=name,
        output="screen",
    )


def _setup(context, *args, **kwargs):
    root = Path(__file__).resolve().parents[1]
    config_path = Path(LaunchConfiguration("config_path").perform(context))
    dry_run = _as_bool(LaunchConfiguration("dry_run").perform(context))
    start_realsense = _as_bool(LaunchConfiguration("start_realsense").perform(context))
    start_orbslam3 = _as_bool(LaunchConfiguration("start_orbslam3").perform(context))
    start_perception = _as_bool(LaunchConfiguration("start_perception").perform(context))
    start_arm = _as_bool(LaunchConfiguration("start_arm").perform(context))
    start_voice = _as_bool(LaunchConfiguration("start_voice").perform(context))

    os.environ["GUOSAI_ROOT"] = str(root)
    cfg = _load_config(str(config_path))
    slam = cfg.get("slam", {})
    motion = cfg.get("motion", {})
    nav = cfg.get("navigation", {})
    cone = cfg.get("cone_avoidance", {})
    inspection = cfg.get("inspection", {})
    arm = cfg.get("arm", {})
    realsense = cfg.get("realsense", {})
    fsm = cfg.get("fsm", {})

    actions = []
    if not dry_run and start_realsense:
        actions.append(
            _shell_process(
                root,
                _expand(realsense.get("command", "ros2 launch realsense2_camera rs_launch.py"), root),
                "realsense",
            )
        )
    if not dry_run and start_orbslam3:
        actions.append(
            _shell_process(
                root,
                _expand(cfg.get("orbslam3", {}).get("command", ""), root),
                "orbslam3_final_map",
            )
        )

    actions.append(
        _py_node(
            root,
            "nodes/localization_watchdog.py",
            {
                "pose_topic": slam.get("pose_topic", "/camera_pose"),
                "pose_type": slam.get("pose_type", "pose_stamped"),
                "stable_samples": slam.get("stable_samples", 15),
                "stable_max_position_step": slam.get("stable_max_position_step", 0.08),
                "stable_max_yaw_step": slam.get("stable_max_yaw_step", 0.35),
                "pose_timeout_sec": slam.get("pose_timeout_sec", 0.8),
                "jump_position_threshold": slam.get("jump_position_threshold", 0.45),
                "jump_yaw_threshold": slam.get("jump_yaw_threshold", 1.2),
                "stop_topic": motion.get("stop_topic", "/motion/stop"),
            },
            "localization_watchdog",
        )
    )
    actions.append(
        _py_node(
            root,
            "nodes/waypoint_navigator.py",
            {
                "waypoints_yaml": _expand(slam.get("waypoints_yaml", ""), root),
                "pose_topic": slam.get("pose_topic", "/camera_pose"),
                "pose_type": slam.get("pose_type", "pose_stamped"),
                "goal_topic": nav.get("goal_topic", "/waypoint/goal"),
                "status_topic": nav.get("status_topic", "/waypoint/status"),
                "cmd_topic": nav.get("cmd_topic", "/motion/nav_cmd"),
                "goal_tolerance": nav.get("goal_tolerance", 0.16),
                "yaw_tolerance": nav.get("yaw_tolerance", 0.22),
                "kp_linear": nav.get("kp_linear", 0.45),
                "kp_angular": nav.get("kp_angular", 1.2),
                "max_vx": nav.get("max_vx", 0.28),
                "max_wz": nav.get("max_wz", 0.45),
                "rotate_in_place_angle": nav.get("rotate_in_place_angle", 0.75),
            },
            "waypoint_navigator",
        )
    )
    actions.append(
        _py_node(
            root,
            "nodes/motion_mux.py",
            {
                "receiver_host": motion.get("receiver_host", "127.0.0.1"),
                "receiver_port": motion.get("receiver_port", 5005),
                "send_hz": motion.get("send_hz", 10.0),
                "dry_run": _bool_text(dry_run),
                "nav_cmd_topic": motion.get("nav_cmd_topic", "/motion/nav_cmd"),
                "avoid_cmd_topic": motion.get("avoid_cmd_topic", "/motion/avoid_cmd"),
                "stop_topic": motion.get("stop_topic", "/motion/stop"),
                "state_topic": motion.get("mux_state_topic", "/motion_mux/state"),
                "max_cmd_age_sec": motion.get("max_cmd_age_sec", 0.6),
                "max_vx": motion.get("max_vx", 0.35),
                "max_vy": motion.get("max_vy", 0.15),
                "max_wz": motion.get("max_wz", 0.55),
                "obstacle_priority": _bool_text(motion.get("obstacle_priority", True)),
            },
            "motion_mux",
        )
    )

    if not dry_run and start_perception:
        actions += [
            _py_node(
                root,
                "nodes/cone_avoidance_node.py",
                {
                    "model": _expand(cone.get("model", ""), root),
                    "camera": cone.get("camera", "0"),
                    "conf": cone.get("conf", 0.35),
                    "send_hz": cone.get("send_hz", 8.0),
                    "enabled_topic": cone.get("enabled_topic", "/motion/enable_cone_avoidance"),
                    "cmd_topic": cone.get("cmd_topic", "/motion/avoid_cmd"),
                },
                "cone_avoidance_node",
            ),
            _shell_process(
                root,
                f"python3 integration_bridge/bridge_node.py --log-path "
                f"{shlex.quote(_expand(inspection.get('bridge_log_path', 'output/integration_bridge/final_events.jsonl'), root))}",
                "integration_bridge",
            ),
            _shell_process(
                root,
                _expand(inspection.get("live_detect_command", "python3 live_detect_yolo_opencv.py --no-gui --no-stream"), root),
                "inspection_live_detect",
            ),
        ]

    if not dry_run and start_arm:
        actions += [
            _py_node(
                root,
                "arm_grasp/astra_camera_node.py",
                {"camera_index": 0},
                "astra_camera_node",
            ),
            _py_node(
                root,
                "arm_grasp/arm_grasp/vision_node.py",
                {
                    "config_path": _expand(arm.get("grasp_config", ""), root),
                },
                "arm_vision_node",
            ),
            _py_node(
                root,
                "arm_grasp/arm_grasp/arm_control_node.py",
                {"config_path": _expand(arm.get("grasp_config", ""), root)},
                "arm_control_node",
            ),
            _py_node(root, "arm_grasp/arm_grasp/inspection_memory_node.py", {}, "inspection_memory_node"),
            _py_node(root, "arm_grasp/arm_grasp/visualization_node.py", {}, "visualization_node"),
        ]

    actions.append(
        _py_node(
            root,
            "arm_grasp/arm_grasp/task_manager_node.py",
            {
                "config_path": str(config_path),
                "dry_run": _bool_text(dry_run),
                "auto_exit_on_done": _bool_text(dry_run),
                "auto_exit_delay_sec": 2.0,
                "auto_start": _bool_text(fsm.get("auto_start", True)),
            },
            "task_manager_node",
            on_exit=[Shutdown(reason="task_manager_node exited")],
        )
    )

    vb = cfg.get("voice_broadcast", {})
    if start_voice:
        voice_params = {
            "enabled": _bool_text(vb.get("enabled", True)),
            "audio_dir": _expand(vb.get("audio_dir", "output/audio"), root),
            "engine": vb.get("engine", "mock"),
            "gap_sec": vb.get("gap_sec", 0.4),
            "result_topic": vb.get("result_topic", "/inspection/all"),
            "detailed_topic": vb.get("detailed_topic", "/inspection/all_detailed"),
            "state_topic": vb.get("state_topic", "/competition/state"),
            "playback_log_path": _expand(vb.get("playback_log_path", "output/voice_broadcast/playback.tsv"), root),
        }
        # rclpy rejects empty -p device:= args; default to "default" so aplay uses the system card.
        voice_params["device"] = vb.get("device", "") or "default"
        actions.append(
            _py_node(
                root,
                "nodes/voice_broadcast_node.py",
                voice_params,
                "voice_broadcast_node",
            )
        )

    return actions


def generate_launch_description() -> LaunchDescription:
    root = Path(__file__).resolve().parents[1]
    default_config = root / "config" / "guosai_final.yaml"
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=str(default_config)),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("log_dir", default_value=str(root / "logs" / "final_manual")),
            DeclareLaunchArgument("start_realsense", default_value="true"),
            DeclareLaunchArgument("start_orbslam3", default_value="true"),
            DeclareLaunchArgument("start_perception", default_value="true"),
            DeclareLaunchArgument("start_arm", default_value="true"),
            DeclareLaunchArgument("start_voice", default_value="true"),
            SetEnvironmentVariable("GUOSAI_ROOT", str(root)),
            SetEnvironmentVariable("ROS_LOG_DIR", LaunchConfiguration("log_dir")),
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            OpaqueFunction(function=_setup),
        ]
    )

