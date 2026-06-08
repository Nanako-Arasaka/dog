"""感知层统一入口 —— 纯检测接口，不做任何运动/机械臂控制。

职责边界：
  算力板（NVIDIA）→ 相机取流 + 模型推理 → 返回结构化检测结果
  机器狗本地      → 接收结果 → 决策 → 调运动/机械臂

PerceptionGateway 的所有方法只返回结构化数据，不产生任何副作用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.types import (
    BBox,
    ConeDetection,
    EquipmentDetection,
    GaugeReading,
    InspectionReading,
    MeterStatus,
    StripDetection,
    TargetPose,
    Zone,
    ZoneLetterResult,
)


class PerceptionGateway(ABC):
    """感知层抽象 —— 纯检测，不控制任何硬件。

    实现：
    - JsonScenarioPerception    (JSON 场景仿真)
    - LocalPerceptionGateway    (本机 CV，相机取流 + 本地推理)
    - RemotePerceptionGateway   (外接 NVIDIA 算力板，TCP 通信)
    """

    # ── 避障阶段 ────────────────────────────────────────

    @abstractmethod
    def detect_obstacles(self, rgb: np.ndarray | None = None) -> list[ConeDetection]:
        """检测障碍物（锥桶），返回按距离排序的检测列表。

        Args:
            rgb: RGB 图像。None 时由实现自行获取（如远程已持有相机）。
        Returns:
            锥桶检测列表（空列表 = 无障碍物）。
        """
        ...

    @abstractmethod
    def obstacle_cleared(self) -> bool:
        """判断前方路径是否已无障碍物。"""
        ...

    # ── 巡检阶段 ────────────────────────────────────────

    @abstractmethod
    def detect_zone_letters(self, rgb: np.ndarray | None = None) -> list[ZoneLetterResult]:
        """识别视野中所有区域字母 A/B/C/D。

        Returns:
            区域字母列表，含置信度。
        """
        ...

    @abstractmethod
    def detect_gauges(self, rgb: np.ndarray | None = None) -> list[GaugeReading]:
        """读取视野中所有仪表盘状态。

        Returns:
            仪表读数列表，含状态 + 置信度。
        """
        ...

    @abstractmethod
    def poll_inspection(self) -> list[InspectionReading]:
        """轮询巡检播报结果（兼容旧接口，内部将 detect_zone_letters + detect_gauges
        合并为逐区域的 InspectionReading 序列）。

        在 INSPECTION_READ 阶段每 tick 调用一次，每次返回一个待播报结果。
        返回空列表 = 本轮已无更多读数。
        """
        ...

    # ── 抓取阶段 ────────────────────────────────────────

    @abstractmethod
    def detect_red_strips(self, rgb: np.ndarray | None = None) -> list[StripDetection]:
        """检测红色异常长条。

        Returns:
            红色长条检测列表（空 = 视野中未发现）。
        """
        ...

    @abstractmethod
    def estimate_target_pose(self, rgb: np.ndarray | None = None) -> TargetPose | None:
        """估计当前目标物体（红色长条 or 放置箱）的 3D 位姿。

        用于机械臂抓取前获取目标的 (x, y, z, roll, pitch, yaw)。

        Returns:
            目标位姿，含置信度。None = 无法估计。
        """
        ...

    # ── 生命周期 ────────────────────────────────────────

    @abstractmethod
    def is_ready(self) -> bool:
        """感知层是否就绪（模型加载完成 / 远程连接建立）。"""
        ...
