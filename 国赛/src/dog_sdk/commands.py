from __future__ import annotations

import struct

from .protocol import build_complex_command, build_simple_command

# 常用指令（基于现有文档）
CMD_STAND_OR_LIE = 0x21010202
CMD_HEARTBEAT = 0x21040001

CMD_JOYSTICK_FORWARD = 0x21010130
CMD_JOYSTICK_TURN = 0x21010135
CMD_GAIT_WALK = 0x21010300
CMD_OBSTACLE_AVOID_ON = 0x21011102


def cmd_stand_toggle() -> bytes:
    return build_simple_command(CMD_STAND_OR_LIE)


def cmd_heartbeat() -> bytes:
    return build_simple_command(CMD_HEARTBEAT)


def cmd_joystick_forward(value: int) -> bytes:
    payload = struct.pack("<i", int(value))
    return build_complex_command(CMD_JOYSTICK_FORWARD, payload)


def cmd_joystick_turn(value: int) -> bytes:
    payload = struct.pack("<i", int(value))
    return build_complex_command(CMD_JOYSTICK_TURN, payload)


def cmd_gait_walk() -> bytes:
    return build_simple_command(CMD_GAIT_WALK)


def cmd_obstacle_avoid_on() -> bytes:
    return build_simple_command(CMD_OBSTACLE_AVOID_ON)
