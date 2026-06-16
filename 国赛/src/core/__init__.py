"""core 模块 —— 领域类型与异常。"""

from core.exceptions import (
    CameraError,
    DetectionFailedError,
    HardwareError,
    InspectionError,
    InspectionReadError,
    InspectionTimeoutError,
    PerceptionError,
    SpeakerError,
    ZoneOCRError,
)
from core.types import (
    VALID_ZONES,
    BBox,
    CameraIntrinsics,
    GaugeReading,
    InspectionReading,
    MeterStatus,
    Point3D,
    Zone,
    ZoneLetterResult,
)

__all__ = [
    # enums
    "Zone",
    "MeterStatus",
    # structs
    "BBox",
    "InspectionReading",
    "ZoneLetterResult",
    "GaugeReading",
    "CameraIntrinsics",
    "Point3D",
    # constants
    "VALID_ZONES",
    # exceptions
    "InspectionError",
    "InspectionTimeoutError",
    "InspectionReadError",
    "ZoneOCRError",
    "PerceptionError",
    "DetectionFailedError",
    "HardwareError",
    "CameraError",
    "SpeakerError",
]
