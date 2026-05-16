from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from dog_sdk.commands import (
    cmd_gait_walk,
    cmd_heartbeat,
    cmd_joystick_forward,
    cmd_joystick_turn,
    cmd_obstacle_avoid_on,
)
from dog_sdk.telemetry import CODE_STATE_UPLOAD, try_parse_packet
from dog_sdk.transport import RobotEndpoint, UdpTransport


@dataclass(frozen=True)
class RuntimeConfig:
    robot_ip: str
    robot_command_port: int
    local_ip: str
    local_telemetry_port: int
    heartbeat_hz: float
    main_loop_hz: float
    log_telemetry: bool


class DogController:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self._cfg = cfg
        self._transport = UdpTransport(
            endpoint=RobotEndpoint(ip=cfg.robot_ip, command_port=cfg.robot_command_port),
            local_ip=cfg.local_ip,
            local_telemetry_port=cfg.local_telemetry_port,
        )
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._rx_thread: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._last_motion = (0, 0)

    def send(self, packet: bytes) -> None:
        with self._send_lock:
            self._transport.send(packet)

    def set_walk_gait(self) -> None:
        self.send(cmd_gait_walk())

    def enable_obstacle_avoidance(self) -> None:
        self.send(cmd_obstacle_avoid_on())

    def set_motion(self, forward: int, turn: int) -> None:
        # 接口文档中摇杆有死区，框架侧先做一次裁剪。
        f = max(min(int(forward), 32768), -32768)
        t = max(min(int(turn), 32768), -32768)
        if abs(f) < 6553:
            f = 0
        if abs(t) < 9553:
            t = 0

        if (f, t) == self._last_motion:
            return
        self.send(cmd_joystick_forward(f))
        self.send(cmd_joystick_turn(t))
        self._last_motion = (f, t)

    def stop_motion(self) -> None:
        self.set_motion(0, 0)

    def start_background_loops(self) -> None:
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._rx_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._heartbeat_thread.start()
        self._rx_thread.start()

    def stop_background_loops(self) -> None:
        self._running = False
        self.stop_motion()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        self._transport.close()

    def sleep_for_main_tick(self) -> None:
        hz = max(self._cfg.main_loop_hz, 1.0)
        time.sleep(1.0 / hz)

    def _heartbeat_loop(self) -> None:
        hz = max(self._cfg.heartbeat_hz, 1.0)
        interval = 1.0 / hz
        while self._running:
            self.send(cmd_heartbeat())
            time.sleep(interval)

    def _telemetry_loop(self) -> None:
        while self._running:
            raw = self._transport.recv()
            if not raw:
                continue
            pkt = try_parse_packet(raw)
            if pkt is None:
                continue
            if self._cfg.log_telemetry and pkt.header.code == CODE_STATE_UPLOAD:
                logging.info(
                    "telemetry code=0x%04X type=%d payload=%d",
                    pkt.header.code,
                    pkt.header.msg_type,
                    pkt.header.payload_size,
                )
