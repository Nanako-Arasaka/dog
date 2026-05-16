from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotEndpoint:
    ip: str
    command_port: int


class UdpTransport:
    def __init__(self, endpoint: RobotEndpoint, local_ip: str, local_telemetry_port: int) -> None:
        self._endpoint = endpoint
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._telemetry_sock.bind((local_ip, local_telemetry_port))
        self._telemetry_sock.settimeout(0.1)

    def send(self, packet: bytes) -> None:
        self._cmd_sock.sendto(packet, (self._endpoint.ip, self._endpoint.command_port))

    def recv(self, buf_size: int = 2048) -> bytes | None:
        try:
            raw, _ = self._telemetry_sock.recvfrom(buf_size)
            return raw
        except TimeoutError:
            return None
        except socket.timeout:
            return None

    def close(self) -> None:
        self._cmd_sock.close()
        self._telemetry_sock.close()
