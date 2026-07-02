from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else number


def _parse_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    return None


@dataclass
class ConeObstacle:
    x: Optional[float]
    z: Optional[float]
    conf: float = 1.0
    bbox: Tuple[float, float, float, float] = field(default_factory=tuple)
    age: int = 0
    last_seen: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConeObstacle":
        bbox_raw = data.get("bbox") or ()
        if isinstance(bbox_raw, Sequence) and not isinstance(bbox_raw, (str, bytes)):
            bbox = tuple(float(v) for v in bbox_raw[:4])
        else:
            bbox = ()
        return cls(
            x=_to_optional_float(data.get("x")),
            z=_to_optional_float(data.get("z")),
            conf=float(data.get("conf", data.get("confidence", 1.0))),
            bbox=bbox,
            age=int(data.get("age", 0)),
            last_seen=_parse_time(data.get("last_seen")),
        )


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    reason: str = "stop"
    state: str = "IDLE"
    source: str = "cone_avoidance"

    @classmethod
    def stop(cls, reason: str, state: str = "RECOVER_STOP") -> "VelocityCommand":
        return cls(vx=0.0, vy=0.0, wz=0.0, reason=reason, state=state)

    def with_state(self, state: str) -> "VelocityCommand":
        return VelocityCommand(
            vx=self.vx,
            vy=self.vy,
            wz=self.wz,
            reason=self.reason,
            state=state,
            source=self.source,
        )

    def to_payload(self) -> dict:
        return {
            "source": self.source,
            "reason": self.reason,
            "vx": round(float(self.vx), 4),
            "vy": round(float(self.vy), 4),
            "wz": round(float(self.wz), 4),
            "state": self.state,
        }

    def log_line(self) -> str:
        return (
            f"state={self.state} reason={self.reason} "
            f"vx={self.vx:.2f} vy={self.vy:.2f} wz={self.wz:.2f}"
        )


@dataclass
class ControlConfig:
    normal_speed: float = 0.15
    slow_speed: float = 0.08
    max_turn_speed: float = 0.25
    safe_radius: float = 0.55
    slow_distance: float = 1.20
    stop_distance: float = 0.55
    front_emergency_width: float = 0.45
    front_emergency_distance: float = 0.50
    perception_timeout: float = 0.50
    recover_stop_seconds: float = 0.40
    send_rate_hz: float = 10.0
    min_confidence: float = 0.45
    max_abs_x: float = 5.0
    max_z: float = 3.0
    gap_pass_width: float = 1.10
    center_deadband: float = 0.08
    turn_smoothing_alpha: float = 0.35
    clear_done_seconds: float = 2.0
    min_run_seconds: float = 6.0
    exit_seconds: float = 1.0
    receiver_ip: str = "127.0.0.1"
    receiver_port: int = 5005
    min_depth_valid_ratio: float = 0.35
    min_realsense_fps: float = 8.0
