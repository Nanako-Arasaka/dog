"""领域共享类型定义 —— 所有枚举、数据类、类型别名。

所有模块从这里导入基础类型，避免循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, TypeAlias

# ── 场地与区域 ──────────────────────────────────────────


class Zone(str, Enum):
    """比赛区域字母标识"""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


VALID_ZONES: tuple[str, ...] = tuple(z.value for z in Zone)

# ── 仪表状态 ────────────────────────────────────────────


class MeterStatus(str, Enum):
    """仪表盘读数状态"""

    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"

    @property
    def is_abnormal(self) -> bool:
        return self in (MeterStatus.LOW, MeterStatus.HIGH)

    @property
    def cn_display(self) -> str:
        _map = {
            MeterStatus.NORMAL: "正常",
            MeterStatus.LOW: "偏低",
            MeterStatus.HIGH: "偏高",
        }
        return _map[self]

    @property
    def cn_health(self) -> str:
        return "异常" if self.is_abnormal else "正常"

    @classmethod
    def from_value(cls, raw_value: float, normal_range: tuple[float, float]) -> "MeterStatus":
        """根据原始读数和正常区间判断状态。"""
        lo, hi = normal_range
        if raw_value < lo:
            return cls.LOW
        if raw_value > hi:
            return cls.HIGH
        return cls.NORMAL


# ── 任务阶段 ────────────────────────────────────────────


class MissionPhase(str, Enum):
    """国赛任务阶段（10 个执行阶段 + 3 个终态）"""

    INIT = "INIT"
    OBSTACLE_APPROACH = "OBSTACLE_APPROACH"
    OBSTACLE_DETECT = "OBSTACLE_DETECT"
    OBSTACLE_CROSS = "OBSTACLE_CROSS"
    INSPECTION_NAV = "INSPECTION_NAV"
    INSPECTION_SCAN = "INSPECTION_SCAN"
    INSPECTION_READ = "INSPECTION_READ"
    PICKUP_PLAN = "PICKUP_PLAN"
    PICKUP_NAV = "PICKUP_NAV"
    PICKUP_GRAB = "PICKUP_GRAB"
    PICKUP_TRANSPORT = "PICKUP_TRANSPORT"
    PICKUP_PLACE = "PICKUP_PLACE"
    DONE = "DONE"
    FAILED = "FAILED"
    STOPPED = "STOPPED"

    @property
    def is_terminal(self) -> bool:
        return self in (MissionPhase.DONE, MissionPhase.FAILED, MissionPhase.STOPPED)


# ── 抓取结果 ────────────────────────────────────────────


class PickupOutcome(str, Enum):
    SUCCESS = "success"
    DROP = "drop"
    RETRY = "retry"
    ARM_ERROR = "arm_error"


# ── 导航状态 ────────────────────────────────────────────


class NavigationStatus(str, Enum):
    MOVING = "moving"
    ARRIVED = "arrived"
    BLOCKED = "blocked"
    LOST = "lost"


# ── BBox ─────────────────────────────────────────────────


class BBox(NamedTuple):
    """边界框（像素坐标）"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


# ── 巡检读数 ────────────────────────────────────────────


@dataclass(frozen=True)
class InspectionReading:
    """单次巡检播报结果"""

    zone: Zone
    meter_status: MeterStatus
    confidence: float  # 综合置信度 0.0 ~ 1.0
    meter_raw_value: float | None = None  # 原始仪表读数（如有）
    timestamp: float = 0.0

    def broadcast_text(self) -> str:
        return (
            f"{self.zone.value}区域仪表盘显示{self.meter_status.cn_display}"
            f"，状态{self.meter_status.cn_health}"
        )


# ── 检测结果 ────────────────────────────────────────────


@dataclass(frozen=True)
class ConeDetection:
    """锥桶检测（避障阶段）"""

    bbox: BBox
    center_3d: tuple[float, float, float]  # (x, y, z) 相机坐标系，单位米
    confidence: float


@dataclass(frozen=True)
class EquipmentDetection:
    """电气设备检测（巡检阶段）—— 配电柜或变压器"""

    bbox: BBox
    equipment_type: str  # "power_cabinet" | "transformer"
    zone_letter: str | None = None  # OCR 识别后填充
    zone_confidence: float = 0.0


@dataclass(frozen=True)
class StripDetection:
    """红色异常长条检测（抓取阶段）"""

    bbox: BBox
    center_3d: tuple[float, float, float]  # 相机坐标系
    confidence: float
    timestamp: float = 0.0


@dataclass(frozen=True)
class ZoneLetterResult:
    """区域字母识别结果"""

    zone: Zone
    confidence: float
    bbox: BBox | None = None
    timestamp: float = 0.0


@dataclass(frozen=True)
class GaugeReading:
    """仪表盘读数结果"""

    zone: Zone
    status: MeterStatus
    confidence: float
    raw_value: float | None = None
    timestamp: float = 0.0

    def broadcast_text(self) -> str:
        return (
            f"{self.zone.value}区域仪表盘显示{self.status.cn_display}"
            f"，状态{self.status.cn_health}"
        )


@dataclass(frozen=True)
class TargetPose:
    """机械臂目标位姿估计（抓取/放置用）"""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    confidence: float = 0.0
    timestamp: float = 0.0


# ── 位姿 ────────────────────────────────────────────────


@dataclass
class RobotPose:
    """机器人位姿（世界坐标系）"""

    x: float = 0.0  # 米
    y: float = 0.0
    yaw: float = 0.0  # 弧度
    timestamp: float = 0.0


@dataclass(frozen=True)
class ArmPose:
    """机械臂末端位姿（基座坐标系）"""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class JointAngles:
    """机械臂关节角度"""

    joints: tuple[float, ...]  # 弧度


@dataclass(frozen=True)
class CameraIntrinsics:
    """相机内参"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


# ── 3D 点 ───────────────────────────────────────────────

Point3D: TypeAlias = tuple[float, float, float]
