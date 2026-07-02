from __future__ import annotations

import json
import socket
from typing import Tuple

from .models import ControlConfig, VelocityCommand


class MotionSender:
    def __init__(self, host: str | None = None, port: int | None = None, config: ControlConfig | None = None) -> None:
        cfg = config or ControlConfig()
        self.target: Tuple[str, int] = (host or cfg.receiver_ip, int(port or cfg.receiver_port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: VelocityCommand) -> None:
        data = json.dumps(command.to_payload(), separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.target)

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> "MotionSender":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
