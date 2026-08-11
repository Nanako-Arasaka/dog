#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP command bridge for Jueying Lite2.

Default data flow in this repo:
  Jetson/vision/SLAM side -> this script: UDP JSON on 0.0.0.0:5005
  this script -> Lite2 motion host: UDP command on 192.168.1.120:43893

Accepted JSON examples:
  {"action":"forward","speed":12000,"duration":0.5}
  {"cmd":"turn_left","speed":8000}
  {"vx":0.2,"vy":0.0,"wz":-0.3}
  {"linear":{"x":0.2,"y":0.0},"angular":{"z":-0.3}}
  {"class":"normal","detected":true,"confidence":0.8,"cx":160}
"""

import argparse
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


CMD_STAND_SIT = 0x21010202
CMD_FORWARD_BACK = 0x21010130
CMD_LEFT_RIGHT = 0x21010131
CMD_TURN = 0x21010135
CMD_WALK_GAIT = 0x21010300
CMD_CRAWL_GAIT = 0x21010406
CMD_STAIR_GAIT = 0x21010401
CMD_STAND_MODE = 0x21010D05
CMD_MOVE_MODE = 0x21010D06
CMD_HEARTBEAT = 0x21040001


ACTION_ALIASES = {
    "stop": "stop",
    "halt": "stop",
    "idle": "stop",
    "forward": "forward",
    "go": "forward",
    "back": "backward",
    "backward": "backward",
    "left": "left",
    "move_left": "left",
    "right": "right",
    "move_right": "right",
    "turn_left": "turn_left",
    "yaw_left": "turn_left",
    "turn_right": "turn_right",
    "yaw_right": "turn_right",
    "stand": "stand_sit",
    "sit": "stand_sit",
    "stand_sit": "stand_sit",
    "toggle_stand": "stand_sit",
    "walk": "walk_gait",
    "walk_gait": "walk_gait",
    "crawl": "crawl_gait",
    "crawl_gait": "crawl_gait",
    "stair": "stair_gait",
    "stair_gait": "stair_gait",
    "move_mode": "move_mode",
    "stand_mode": "stand_mode",
}


@dataclass
class MotionCommand:
    forward: int = 0
    lateral: int = 0
    turn: int = 0
    action: str = "velocity"
    duration: Optional[float] = None


class Lite2Controller:
    def __init__(
        self,
        robot_ip: str,
        robot_port: int,
        heartbeat_hz: float,
        command_repeat: int = 1,
        command_repeat_interval: float = 0.02,
        dry_run: bool = False,
    ) -> None:
        self.target = (robot_ip, robot_port)
        self.dry_run = dry_run
        self.command_repeat = max(1, int(command_repeat))
        self.command_repeat_interval = max(0.0, float(command_repeat_interval))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event = threading.Event()
        self._heartbeat_interval = 1.0 / max(heartbeat_hz, 0.1)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self.stop()
        self._heartbeat_thread.join(timeout=1.0)
        self.sock.close()

    def send(self, code: int, value: int = 0, cmd_type: int = 0) -> None:
        packet = struct.pack("<3i", int(code), int(value), int(cmd_type))
        for index in range(self.command_repeat):
            if self.dry_run:
                print(f"[dry-run] -> {self.target[0]}:{self.target[1]} code=0x{code:08X} value={value} type={cmd_type}")
            else:
                self.sock.sendto(packet, self.target)
            if index + 1 < self.command_repeat and self.command_repeat_interval > 0.0:
                time.sleep(self.command_repeat_interval)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.send(CMD_HEARTBEAT, 0, 0)
            except OSError as exc:
                print(f"[heartbeat] send failed: {exc}")
            self._stop_event.wait(self._heartbeat_interval)

    def stop(self) -> None:
        self.send(CMD_FORWARD_BACK, 0, 0)
        self.send(CMD_LEFT_RIGHT, 0, 0)
        self.send(CMD_TURN, 0, 0)

    def apply_velocity(self, forward: int, lateral: int, turn: int) -> None:
        self.send(CMD_FORWARD_BACK, forward, 0)
        self.send(CMD_LEFT_RIGHT, lateral, 0)
        self.send(CMD_TURN, turn, 0)

    def apply_action(self, action: str) -> None:
        if action == "stop":
            self.stop()
        elif action == "stand_sit":
            self.send(CMD_STAND_SIT, 0, 0)
        elif action == "walk_gait":
            self.send(CMD_WALK_GAIT, 0, 0)
        elif action == "crawl_gait":
            self.send(CMD_CRAWL_GAIT, 0, 0)
        elif action == "stair_gait":
            self.send(CMD_STAIR_GAIT, 0, 0)
        elif action == "move_mode":
            self.send(CMD_MOVE_MODE, 0, 0)
        elif action == "stand_mode":
            self.send(CMD_STAND_MODE, 0, 0)
        else:
            raise ValueError(f"unsupported action: {action}")


def parse_startup_actions(raw: str) -> list[str]:
    actions = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        action = ACTION_ALIASES.get(name)
        if action is None:
            raise ValueError(f"unknown startup action: {item}")
        if action not in {"walk_gait", "crawl_gait", "stair_gait", "move_mode", "stand_mode", "stand_sit"}:
            raise ValueError(f"startup action is not a Lite2 mode/gait action: {item}")
        actions.append(action)
    return actions


def run_startup_sequence(controller: Lite2Controller, actions: Iterable[str], interval: float) -> None:
    actions = list(actions)
    if not actions:
        return
    print(f"[init] sending startup actions: {', '.join(actions)}")
    for action in actions:
        controller.apply_action(action)
        time.sleep(max(0.0, interval))


def clamp_int(value: float, limit: int) -> int:
    return int(max(-limit, min(limit, round(value))))


def normalize_scalar(value: Any, limit: int) -> int:
    number = float(value)
    if -1.0 <= number <= 1.0:
        return clamp_int(number * limit, limit)
    return clamp_int(number, limit)


def normalize_velocity_field(value: Any, limit: int, minimum: int, normalized_deadband: float) -> int:
    number = float(value)
    if -1.0 <= number <= 1.0:
        if abs(number) < normalized_deadband:
            return 0
        return ensure_effective_speed(clamp_int(number * limit, limit), minimum)
    return ensure_effective_speed(clamp_int(number, limit), minimum)


def ensure_effective_speed(value: int, minimum: int) -> int:
    if value == 0 or minimum <= 0:
        return value
    sign = 1 if value > 0 else -1
    return sign * max(abs(value), minimum)


def pick(payload: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return default


def parse_linear_angular(
    payload: Dict[str, Any],
    max_fb: int,
    max_lr: int,
    max_turn: int,
    min_forward_speed: int,
    min_lateral_speed: int,
    min_turn_speed: int,
    normalized_deadband: float,
) -> Optional[MotionCommand]:
    linear = payload.get("linear")
    angular = payload.get("angular")
    if not isinstance(linear, dict) and not isinstance(angular, dict):
        return None

    vx = linear.get("x", 0.0) if isinstance(linear, dict) else 0.0
    vy = linear.get("y", 0.0) if isinstance(linear, dict) else 0.0
    wz = angular.get("z", 0.0) if isinstance(angular, dict) else 0.0
    return MotionCommand(
        forward=normalize_velocity_field(vx, max_fb, min_forward_speed, normalized_deadband),
        lateral=normalize_velocity_field(vy, max_lr, min_lateral_speed, normalized_deadband),
        turn=normalize_velocity_field(wz, max_turn, min_turn_speed, normalized_deadband),
    )


def parse_velocity(
    payload: Dict[str, Any],
    max_fb: int,
    max_lr: int,
    max_turn: int,
    min_forward_speed: int,
    min_lateral_speed: int,
    min_turn_speed: int,
    normalized_deadband: float,
) -> Optional[MotionCommand]:
    if not any(name in payload for name in ("vx", "vy", "wz", "forward", "lateral", "turn")):
        return None

    vx = pick(payload, "vx", "forward", default=0.0)
    vy = pick(payload, "vy", "lateral", default=0.0)
    wz = pick(payload, "wz", "turn", default=0.0)
    return MotionCommand(
        forward=normalize_velocity_field(vx, max_fb, min_forward_speed, normalized_deadband),
        lateral=normalize_velocity_field(vy, max_lr, min_lateral_speed, normalized_deadband),
        turn=normalize_velocity_field(wz, max_turn, min_turn_speed, normalized_deadband),
        duration=parse_duration(payload),
    )


def parse_duration(payload: Dict[str, Any]) -> Optional[float]:
    if "duration" not in payload:
        return None
    duration = max(0.0, float(payload["duration"]))
    return duration if duration > 0.0 else None


def parse_action(
    payload: Dict[str, Any],
    default_speed: int,
    turn_speed: int,
    min_forward_speed: int,
    min_lateral_speed: int,
    min_turn_speed: int,
) -> Optional[MotionCommand]:
    raw = pick(payload, "action", "cmd", "command", "move", default=None)
    if raw is None:
        return None

    action = ACTION_ALIASES.get(str(raw).strip().lower())
    if action is None:
        raise ValueError(f"unknown action/cmd: {raw}")

    raw_speed = int(abs(float(payload.get("speed", default_speed))))
    yaw_speed = abs(ensure_effective_speed(
        int(abs(float(payload.get("turn_speed", payload.get("yaw_speed", payload.get("speed", turn_speed)))))),
        min_turn_speed,
    ))
    duration = parse_duration(payload)

    if action == "forward":
        speed = abs(ensure_effective_speed(raw_speed, min_forward_speed))
        return MotionCommand(forward=speed, action=action, duration=duration)
    if action == "backward":
        speed = abs(ensure_effective_speed(raw_speed, min_forward_speed))
        return MotionCommand(forward=-speed, action=action, duration=duration)
    if action == "left":
        speed = abs(ensure_effective_speed(raw_speed, min_lateral_speed))
        return MotionCommand(lateral=-speed, action=action, duration=duration)
    if action == "right":
        speed = abs(ensure_effective_speed(raw_speed, min_lateral_speed))
        return MotionCommand(lateral=speed, action=action, duration=duration)
    if action == "turn_left":
        return MotionCommand(turn=-yaw_speed, action=action, duration=duration)
    if action == "turn_right":
        return MotionCommand(turn=yaw_speed, action=action, duration=duration)
    return MotionCommand(action=action, duration=duration)


def parse_class_result(
    payload: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[MotionCommand]:
    class_name = str(pick(payload, "class_en", "class", default="")).strip().lower()
    if not class_name:
        return None

    detected = bool(payload.get("detected", True))
    confidence = float(payload.get("confidence", 1.0))
    if (not detected) or confidence < args.min_confidence:
        return MotionCommand(action="stop")

    action_by_class = {
        "high": args.class_high_action,
        "normal": args.class_normal_action,
        "low": args.class_low_action,
    }
    action = action_by_class.get(class_name, "stop")
    mapped = parse_action(
        {"action": action, "speed": args.default_speed, "turn_speed": args.turn_speed},
        args.default_speed,
        args.turn_speed,
        args.min_forward_speed,
        args.min_lateral_speed,
        args.min_turn_speed,
    )
    if mapped is None:
        return MotionCommand(action="stop")

    cx = payload.get("cx", None)
    if args.center_x >= 0 and isinstance(cx, (int, float)) and int(cx) >= 0:
        offset = float(cx) - float(args.center_x)
        if abs(offset) > args.deadzone:
            ratio = max(-1.0, min(1.0, offset / max(1.0, float(args.center_x))))
            mapped.turn = clamp_int(ratio * args.turn_speed, args.turn_speed)

    return mapped


def parse_packet(payload: Dict[str, Any], args: argparse.Namespace) -> MotionCommand:
    for parser in (
        lambda data: parse_linear_angular(
            data,
            args.max_forward,
            args.max_lateral,
            args.max_turn,
            args.min_forward_speed,
            args.min_lateral_speed,
            args.min_turn_speed,
            args.normalized_deadband,
        ),
        lambda data: parse_velocity(
            data,
            args.max_forward,
            args.max_lateral,
            args.max_turn,
            args.min_forward_speed,
            args.min_lateral_speed,
            args.min_turn_speed,
            args.normalized_deadband,
        ),
        lambda data: parse_action(
            data,
            args.default_speed,
            args.turn_speed,
            args.min_forward_speed,
            args.min_lateral_speed,
            args.min_turn_speed,
        ),
        lambda data: parse_class_result(data, args),
    ):
        command = parser(payload)
        if command is not None:
            return command
    return MotionCommand(action="stop")


def apply_command(controller: Lite2Controller, command: MotionCommand) -> None:
    if command.action in {"stop", "stand_sit", "walk_gait", "crawl_gait", "stair_gait", "move_mode", "stand_mode"}:
        controller.apply_action(command.action)
        return

    controller.apply_velocity(command.forward, command.lateral, command.turn)
    if command.duration:
        time.sleep(command.duration)
        controller.stop()


def decode_payload(data: bytes) -> Dict[str, Any]:
    text = data.decode("utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("UDP payload must be a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive UDP JSON and control Jueying Lite2 motion.")
    parser.add_argument("--listen-ip", default="0.0.0.0", help="Local UDP listen IP.")
    parser.add_argument("--listen-port", type=int, default=5005, help="Local UDP listen port for upstream data.")
    parser.add_argument("--robot-ip", default="192.168.1.120", help="Lite2 motion host IP.")
    parser.add_argument("--robot-port", type=int, default=43893, help="Lite2 motion command port.")
    parser.add_argument("--heartbeat-hz", type=float, default=4.0, help="Heartbeat frequency sent to Lite2.")
    parser.add_argument("--command-repeat", type=int, default=2, help="Repeat each low-level Lite2 UDP command this many times.")
    parser.add_argument("--command-repeat-interval", type=float, default=0.02, help="Delay between repeated Lite2 UDP commands.")
    parser.add_argument("--startup-actions", default="move_mode,walk_gait", help="Comma-separated Lite2 mode/gait actions sent on startup; set empty to disable.")
    parser.add_argument("--startup-action-interval", type=float, default=0.25, help="Delay between startup actions.")
    parser.add_argument("--timeout", type=float, default=0.8, help="Stop robot if no packet arrives for this many seconds.")
    parser.add_argument("--socket-timeout", type=float, default=0.05, help="UDP receive timeout.")
    parser.add_argument("--buffer-size", type=int, default=65535, help="UDP receive buffer size.")
    parser.add_argument("--dry-run", action="store_true", help="Print Lite2 commands without sending them.")

    parser.add_argument("--default-speed", type=int, default=9000, help="Default translation command value.")
    parser.add_argument("--turn-speed", type=int, default=20000, help="Default turn command value.")
    parser.add_argument("--min-forward-speed", type=int, default=6553, help="Lift non-zero forward/back commands below this value.")
    parser.add_argument("--min-lateral-speed", type=int, default=12553, help="Lift non-zero left/right commands below this value.")
    parser.add_argument("--min-turn-speed", type=int, default=9553, help="Lift non-zero turn commands below this value.")
    parser.add_argument("--normalized-deadband", type=float, default=0.05, help="Treat normalized vx/vy/wz below this magnitude as zero.")
    parser.add_argument("--max-forward", type=int, default=32767, help="Clamp for forward/back command.")
    parser.add_argument("--max-lateral", type=int, default=32767, help="Clamp for left/right command.")
    parser.add_argument("--max-turn", type=int, default=32767, help="Clamp for turn command.")
    parser.add_argument("--invert-forward", action="store_true", help="Invert forward/back command sign.")
    parser.add_argument("--invert-lateral", action="store_true", help="Invert left/right command sign.")
    parser.add_argument("--invert-turn", action="store_true", help="Invert turn command sign.")

    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum vision confidence before acting.")
    parser.add_argument("--center-x", type=int, default=160, help="Image center x for class-result steering; set -1 to disable.")
    parser.add_argument("--deadzone", type=int, default=35, help="Pixel deadzone around center-x.")
    parser.add_argument("--class-high-action", default="turn_left", help="Action for vision class=high.")
    parser.add_argument("--class-normal-action", default="forward", help="Action for vision class=normal.")
    parser.add_argument("--class-low-action", default="turn_right", help="Action for vision class=low.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    startup_actions = parse_startup_actions(args.startup_actions)
    controller = Lite2Controller(
        args.robot_ip,
        args.robot_port,
        args.heartbeat_hz,
        command_repeat=args.command_repeat,
        command_repeat_interval=args.command_repeat_interval,
        dry_run=args.dry_run,
    )
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind((args.listen_ip, args.listen_port))
    rx_sock.settimeout(args.socket_timeout)

    print(f"Listening on {args.listen_ip}:{args.listen_port}")
    print(f"Sending Lite2 commands to {args.robot_ip}:{args.robot_port}")
    if args.dry_run:
        print("Dry-run mode is enabled; robot commands are only printed.")
    run_startup_sequence(controller, startup_actions, args.startup_action_interval)

    last_packet_time = time.monotonic()
    stopped_for_timeout = False
    try:
        while True:
            now = time.monotonic()
            if now - last_packet_time > args.timeout and not stopped_for_timeout:
                controller.stop()
                stopped_for_timeout = True
                print(f"[safety] no packet for {args.timeout:.2f}s, stop sent")

            try:
                data, addr = rx_sock.recvfrom(args.buffer_size)
            except socket.timeout:
                continue

            try:
                payload = decode_payload(data)
                command = parse_packet(payload, args)
                if args.invert_forward:
                    command.forward = -command.forward
                if args.invert_lateral:
                    command.lateral = -command.lateral
                if args.invert_turn:
                    command.turn = -command.turn
                apply_command(controller, command)
                last_packet_time = time.monotonic()
                stopped_for_timeout = False
                print(f"[{addr[0]}:{addr[1]}] payload={payload} -> {command}")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OSError) as exc:
                print(f"[warn] ignored packet from {addr[0]}:{addr[1]}: {exc}")
    except KeyboardInterrupt:
        print("Interrupted, stopping robot.")
    finally:
        controller.close()
        rx_sock.close()


if __name__ == "__main__":
    main()
