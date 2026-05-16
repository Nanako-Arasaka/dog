from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER_STRUCT = struct.Struct("<III")


@dataclass(frozen=True)
class CommandHeader:
    code: int
    payload_size: int
    msg_type: int  # 0: simple, 1: complex

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(self.code, self.payload_size, self.msg_type)

    @staticmethod
    def unpack(raw: bytes) -> "CommandHeader":
        code, payload_size, msg_type = HEADER_STRUCT.unpack(raw[: HEADER_STRUCT.size])
        return CommandHeader(code=code, payload_size=payload_size, msg_type=msg_type)


def build_simple_command(code: int) -> bytes:
    return CommandHeader(code=code, payload_size=0, msg_type=0).pack()


def build_complex_command(code: int, payload: bytes) -> bytes:
    return CommandHeader(code=code, payload_size=len(payload), msg_type=1).pack() + payload
