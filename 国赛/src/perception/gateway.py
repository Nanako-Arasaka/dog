"""感知层统一入口 —— 编排多个检测器，对上暴露简化接口。

任务状态机通过此接口获取感知结果，无需关心内部分工。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.types import (
    BBox,
    ConeDetection,
    EquipmentDetection,
    InspectionReading,
    MeterStatus,
    StripDetection,
    Zone,
)


class PerceptionGateway(ABC):
    """感知层抽象。

    内部编排：
    - ConeDetector       (YOLO 锥桶检测)
    - EquipmentDetector  (YOLO 电力设备检测)
    - MeterReader        (仪表盘读数)
    - ZoneOCR            (字母识别)
    - StripDetector      (红色长条检测 + 3D 定位)
    - DropMonitor        (掉落视觉监控)

    实现：
    - JsonScenarioPerception  (JSON 仿真，无 CV)
    - VisionPerception        (真实 CV 管线，后续填充)
    """

    # ── 避障阶段 ────────────────────────────────────────

    @abstractmethod
    def detect_cones(self, rgb: np.ndarray, depth: np.ndarray) -> list[ConeDetection]:
        """检测锥桶，返回 3D 位置列表。

        Args:
            rgb:   对齐后的 RGB 图像。
            depth: 对齐后的深度图 (m)。

        Returns:
            视野中所有锥桶的检测结果（按距离排序）。
        """
        ...

    @abstractmethod
    def obstacle_cleared(self) -> bool:
        """判断是否已通过障碍区。

        规则：前方路径内无锥桶阻挡。
        """
        ...

    # ── 巡检阶段 ────────────────────────────────────────

    @abstractmethod
    def detect_equipment(self, rgb: np.ndarray) -> list[EquipmentDetection]:
        """检测视野中的配电柜/变压器。

        Returns:
            设备列表，每个包含 bbox 和类型。可通过 ROI 进一步识别。
        """
        ...

    @abstractmethod
    def read_zone_letter(self, rgb: np.ndarray, roi: BBox) -> tuple[str, float]:
        """OCR 识别设备上的区域字母。

        Args:
            rgb: 完整图像。
            roi: 设备 ROI（已由 detect_equipment 给出或手动指定）。

        Returns:
            (letter, confidence): letter ∈ {"A","B","C","D"}，confidence 0~1。
        """
        ...

    @abstractmethod
    def read_meter(self, rgb: np.ndarray, roi: BBox) -> tuple[MeterStatus, float, float | None]:
        """读取仪表盘状态。

        Args:
            rgb: 完整图像。
            roi: 仪表盘 ROI。

        Returns:
            (status, confidence, raw_value): 状态 + 置信度 + 原始读数（如有）。
        """
        ...

    @abstractmethod
    def poll_inspection(self) -> list[InspectionReading]:
        """轮询巡检结果。

        在 INSPECTION_READ 阶段每 tick 调用一次，每次返回一个未处理的读数。
        返回空列表表示本轮已无更多读数。

        Returns:
            单元素列表（含本次读数），或空列表。
        """
        ...

    # ── 抓取阶段 ────────────────────────────────────────

    @abstractmethod
    def detect_red_strip(self, rgb: np.ndarray, depth: np.ndarray) -> StripDetection | None:
        """检测红色异常长条并返回 3D 位置。

        Returns:
            StripDetection 或 None（视野中未发现）。
        """
        ...

    @abstractmethod
    def check_drop(self, rgb: np.ndarray) -> bool:
        """视觉检测：长条是否从夹爪中掉落。

        Returns:
            True 表示检测到掉落。
        """
        ...

    @abstractmethod
    def execute_pickup_for_zone(self, zone: str) -> str:
        """执行一次针对指定区域的长条抓取/投放。

        内部编排：
        1. 检测红色长条位置
        2. 机械臂抓取
        3. 搬运至目标区域箱
        4. 放置 + 确认

        Args:
            zone: 目标区域字母 "A"/"B"/"C"/"D"。

        Returns:
            "success" | "drop" | "retry" | "arm_error"
        """
        ...
