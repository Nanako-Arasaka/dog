#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight checks for the final competition runner."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check guosai final runtime dependencies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dry-run", default="false")
    parser.add_argument("--start-realsense", default="true")
    parser.add_argument("--start-orbslam3", default="true")
    parser.add_argument("--start-perception", default="true")
    parser.add_argument("--start-arm", default="true")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class Reporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERROR] {message}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand(value: Any, root: Path) -> Any:
    if isinstance(value, str):
        os.environ.setdefault("GUOSAI_ROOT", str(root))
        return os.path.expandvars(value.replace("${GUOSAI_ROOT}", str(root)))
    return value


def check_path(reporter: Reporter, label: str, value: str, root: Path, required: bool = True) -> None:
    path_text = expand(value, root)
    if not path_text:
        if required:
            reporter.error(f"{label} is empty")
        return
    if Path(path_text).exists():
        reporter.ok(f"{label}: {path_text}")
    elif required:
        reporter.error(f"{label} not found: {path_text}")
    else:
        reporter.warn(f"{label} not found: {path_text}")


def ros_pkg_exists(package: str) -> bool:
    if not package:
        return False
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_ros_command(reporter: Reporter, label: str, command: str, root: Path) -> None:
    command = expand(command, root)
    if not command:
        reporter.error(f"{label} command is empty")
        return
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        reporter.error(f"{label} command cannot be parsed: {exc}")
        return
    if len(parts) >= 3 and parts[0] == "ros2" and parts[1] in ("run", "launch"):
        package = parts[2]
        if ros_pkg_exists(package):
            reporter.ok(f"{label} ROS package found: {package}")
        else:
            reporter.error(
                f"{label} ROS package not found: {package}. "
                f"Install/source it or change {label}.command in config/guosai_final.yaml."
            )
    else:
        reporter.warn(f"{label} command is not a ros2 run/launch command; package check skipped")


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_python_module(reporter: Reporter, name: str, hint: str = "") -> None:
    if module_exists(name):
        reporter.ok(f"python module found: {name}")
    else:
        suffix = f" ({hint})" if hint else ""
        reporter.error(f"python module missing: {name}{suffix}")


def option_value(parts: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, part in enumerate(parts):
        if part == name and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None


def check_live_detect(reporter: Reporter, command: str, root: Path) -> None:
    command = expand(command, root)
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        reporter.error(f"inspection.live_detect_command cannot be parsed: {exc}")
        return
    model = option_value(parts, "--model")
    camera_id = option_value(parts, "--camera-id")
    camera_path = option_value(parts, "--camera-path")
    if model:
        check_path(reporter, "inspection model", model, root)
    if camera_path:
        check_path(reporter, "inspection camera path", camera_path, root)
    elif camera_id is not None:
        check_path(reporter, "inspection camera path", f"/dev/video{camera_id}", root)
    else:
        reporter.warn("inspection camera not specified; live_detect default will be used")
    check_python_module(reporter, "cv2", "pip install opencv-python")
    check_python_module(reporter, "ultralytics", "pip install ultralytics")


def check_waypoint_values(reporter: Reporter, path_text: str, root: Path) -> None:
    path = Path(expand(path_text, root))
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        reporter.error(f"slam waypoints yaml cannot be read: {exc}")
        return
    items = data.get("waypoints", data) if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        reporter.error("slam waypoints yaml has no waypoint list")
        return
    numeric = []
    names = []
    for item in items:
        if not isinstance(item, dict):
            continue
        names.append(str(item.get("name", "")).strip())
        numeric.append(
            (
                abs(float(item.get("x", 0.0))),
                abs(float(item.get("y", 0.0))),
                abs(float(item.get("yaw", 0.0))),
            )
        )
    if not names:
        reporter.error("slam waypoints yaml has no named waypoints")
        return
    if all(max(values) < 1e-9 for values in numeric):
        reporter.error(
            "slam waypoints yaml still looks like the zero template. "
            "Run: bash scripts/guosai_onekey.sh collect"
        )
    else:
        reporter.ok(f"slam waypoints look populated: {len(names)} points")


def check_voice_broadcast(reporter: Reporter, cfg: dict[str, Any], root: Path) -> None:
    """Validate the voice-broadcast subsystem so a silent config cannot pass preflight."""
    voice = cfg.get("voice_broadcast", {})
    if not voice:
        reporter.warn("voice_broadcast section missing in config; voice check skipped")
        return
    if not as_bool(str(voice.get("enabled", "true"))):
        reporter.warn("voice_broadcast disabled in config; voice check skipped")
        return

    audio_dir = Path(expand(voice.get("audio_dir", ""), root))
    if not audio_dir.exists():
        reporter.error(f"voice audio_dir not found: {audio_dir}")
    else:
        zones = ["A", "B", "C", "D"]
        states = ["low", "normal", "high"]
        missing = [
            f"{z}_{s}.wav"
            for z in zones
            for s in states
            if not (audio_dir / f"{z}_{s}.wav").exists()
        ]
        if missing:
            reporter.error(
                f"voice broadcast wav missing ({len(missing)}/12): "
                + ", ".join(missing)
                + ". Regenerate with: bash scripts/gen_voice_audio.sh"
            )
        else:
            reporter.ok(f"voice broadcast wav present: 12/12 in {audio_dir}")

    engine = str(voice.get("engine", "mock")).strip().lower()
    if engine == "mock":
        reporter.warn(
            "voice_broadcast.engine is 'mock' -> NO sound will play. "
            "Set to 'aplay' (or 'ffplay') on the Jetson before the match."
        )
    elif engine not in ("aplay", "ffplay"):
        reporter.warn(f"voice_broadcast.engine '{engine}' unexpected; expected aplay/ffplay/mock")

    device = str(voice.get("device", "")).strip()
    if not device:
        reporter.warn(
            "voice_broadcast.device is empty -> uses default audio card. "
            "Run scripts/check_onboard_audio.sh to confirm the dog speaker is the default."
        )
    else:
        reporter.ok(f"voice_broadcast.device set: {device}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    os.environ["GUOSAI_ROOT"] = str(root)
    reporter = Reporter()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        reporter.error(f"config not found: {config_path}")
        return 2
    cfg = load_config(config_path)
    reporter.ok(f"config: {config_path}")

    dry_run = as_bool(args.dry_run)
    if dry_run:
        reporter.ok("dry-run mode: hardware/package preflight skipped")
        check_python_module(reporter, "rclpy")
        check_python_module(reporter, "yaml")
        return 0 if not reporter.errors else 2

    slam = cfg.get("slam", {})
    realsense = cfg.get("realsense", {})
    orbslam3 = cfg.get("orbslam3", {})
    cone = cfg.get("cone_avoidance", {})
    inspection = cfg.get("inspection", {})
    arm = cfg.get("arm", {})

    check_python_module(reporter, "rclpy")
    check_python_module(reporter, "yaml")

    check_path(reporter, "slam map", slam.get("map_path", ""), root)
    check_path(reporter, "slam settings yaml", slam.get("settings_yaml", ""), root)
    check_path(reporter, "slam waypoints yaml", slam.get("waypoints_yaml", ""), root)
    check_waypoint_values(reporter, slam.get("waypoints_yaml", ""), root)
    check_path(reporter, "ORB vocabulary", slam.get("vocabulary_path", ""), root)

    if as_bool(args.start_realsense):
        check_ros_command(reporter, "realsense", realsense.get("command", ""), root)
    if as_bool(args.start_orbslam3):
        check_ros_command(reporter, "orbslam3", orbslam3.get("command", ""), root)

    if as_bool(args.start_perception):
        check_path(reporter, "cone model", cone.get("model", ""), root, required=False)
        camera = str(cone.get("camera", "0"))
        check_path(reporter, "cone camera path", camera if camera.startswith("/dev/") else f"/dev/video{camera}", root, required=False)
        check_live_detect(reporter, inspection.get("live_detect_command", ""), root)

    if as_bool(args.start_arm):
        check_path(reporter, "arm grasp config", arm.get("grasp_config", ""), root)
        if ros_pkg_exists("ros_robot_controller_msgs"):
            reporter.ok("arm ROS package found: ros_robot_controller_msgs")
        else:
            reporter.error(
                "arm ROS package not found: ros_robot_controller_msgs. "
                "Build/source arm_grasp dependencies or run with --no-arm for chassis-only tests."
            )
        check_python_module(reporter, "numpy", "pip install numpy")

    check_voice_broadcast(reporter, cfg, root)

    if reporter.errors:
        print("")
        print("[SUMMARY] Preflight failed. Nothing was launched.")
        print("Fix the errors above, or use --dry-run / --no-realsense / --no-orbslam3 / --no-perception / --no-arm for partial tests.")
        return 2
    print("[SUMMARY] Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
