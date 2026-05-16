from __future__ import annotations

from dataclasses import dataclass

from .protocol import CommandHeader

CODE_STATE_UPLOAD = 0x0901


@dataclass(frozen=True)
class TelemetryPacket:
    header: CommandHeader
    payload: bytes


def try_parse_packet(raw: bytes) -> TelemetryPacket | None:
    if len(raw) < 12:
        return None
    header = CommandHeader.unpack(raw[:12])
    payload = raw[12 : 12 + header.payload_size]
    return TelemetryPacket(header=header, payload=payload)
