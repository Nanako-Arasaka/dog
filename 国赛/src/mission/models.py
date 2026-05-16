from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

VALID_ZONES = ("A", "B", "C", "D")


class MeterStatus(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"

    @property
    def is_abnormal(self) -> bool:
        return self in (MeterStatus.LOW, MeterStatus.HIGH)

    @property
    def cn_display(self) -> str:
        if self == MeterStatus.NORMAL:
            return "正常"
        if self == MeterStatus.LOW:
            return "偏低"
        return "偏高"

    @property
    def cn_health(self) -> str:
        return "异常" if self.is_abnormal else "正常"


@dataclass(frozen=True)
class InspectionReading:
    zone: str
    status: MeterStatus

    def broadcast_text(self) -> str:
        return f"{self.zone}区域仪表盘显示{self.status.cn_display}，状态{self.status.cn_health}"


class PickupOutcome(str, Enum):
    SUCCESS = "success"
    DROP = "drop"
    RETRY = "retry"
