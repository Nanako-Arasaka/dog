"""向后兼容 re-export —— 所有领域类型已迁移至 core.types。

新代码请直接:
    from core.types import Zone, MeterStatus, InspectionReading, ...
"""

from __future__ import annotations

# 从 core 重导出，保持旧导入路径可用
from core.types import (
    VALID_ZONES,
    InspectionReading,
    MeterStatus,
    PickupOutcome,
)

__all__ = [
    "VALID_ZONES",
    "InspectionReading",
    "MeterStatus",
    "PickupOutcome",
]
