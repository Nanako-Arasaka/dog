from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .avoidance_state_machine import AvoidanceStateMachine
from .motion_sender import MotionSender
from .models import ConeObstacle, ControlConfig, VelocityCommand


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_control_config(path: str | None, receiver_ip: str, receiver_port: int) -> ControlConfig:
    values: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
    else:
        config_path = Path(__file__).resolve().parent / "config" / "control.yaml"

    if config_path.exists():
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            values[key.strip()] = _parse_scalar(raw_value)

    allowed = {field.name for field in fields(ControlConfig)}
    filtered = {key: value for key, value in values.items() if key in allowed}
    filtered["receiver_ip"] = receiver_ip
    filtered["receiver_port"] = receiver_port
    return ControlConfig(**filtered)


def _obstacles_from_payload(payload: dict[str, Any]) -> list[ConeObstacle]:
    raw = payload.get("obstacles", payload.get("cones", []))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("payload field 'obstacles' must be a list")
    return [ConeObstacle.from_mapping(item) for item in raw]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read ConeObstacle JSON lines and send Lite2 avoidance velocity.")
    parser.add_argument("--receiver-ip", default="127.0.0.1")
    parser.add_argument("--receiver-port", type=int, default=5005)
    parser.add_argument("--config", default=None, help="Path to control.yaml; defaults to cone_avoidance/config/control.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without sending UDP.")
    parser.add_argument("--send-stop-on-exit", action="store_true", help="Send a final stop command on shutdown.")
    parser.add_argument("--motion-status-url", default=None, help="Optional HTTP endpoint for browser visualization, e.g. http://127.0.0.1:8080/motion.json.")
    return parser.parse_args()


def post_motion_status(url: str | None, command: VelocityCommand, input_payload: dict[str, Any]) -> None:
    if not url:
        return
    payload = command.to_payload()
    payload["timestamp"] = time.time()
    payload["input_status"] = input_payload.get("status")
    payload["input_frame"] = input_payload.get("frame")
    payload["obstacle_count"] = len(input_payload.get("obstacles") or [])
    payload["front_depth"] = input_payload.get("front_depth", input_payload.get("front_min_depth"))
    payload["depth_valid_ratio"] = input_payload.get("depth_valid_ratio")
    payload["realsense_fps"] = input_payload.get("realsense_fps", input_payload.get("camera_fps"))
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=0.05):
            pass
    except (OSError, URLError, TimeoutError):
        return


def main() -> None:
    args = parse_args()
    config = load_control_config(args.config, args.receiver_ip, args.receiver_port)
    machine = AvoidanceStateMachine(config)
    machine.start()
    sender = None if args.dry_run else MotionSender(config=config)

    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("input must be JSON object per line")

            obstacles = _obstacles_from_payload(payload)
            front_depth = payload.get("front_depth", payload.get("front_min_depth"))
            command = machine.tick(
                obstacles,
                now=time.monotonic(),
                front_depth=float(front_depth) if front_depth is not None else None,
                aligned_depth_ok=payload.get("aligned_depth_ok"),
                depth_valid_ratio=payload.get("depth_valid_ratio"),
                realsense_fps=payload.get("realsense_fps", payload.get("camera_fps")),
                realsense_ok=payload.get("realsense_ok", payload.get("camera_ok")),
            )
            print(command.log_line(), flush=True)
            post_motion_status(args.motion_status_url, command, payload)
            if sender is not None:
                sender.send(command)
    finally:
        if sender is not None:
            if args.send_stop_on_exit:
                sender.send(VelocityCommand.stop("shutdown"))
            sender.close()


if __name__ == "__main__":
    main()
