"""感知层统一入口 —— 纯检测接口，不做任何运动/机械臂控制。

职责边界：
  算力板（NVIDIA）→ 相机取流 + 巡检识别 → 返回结构化检测结果
  机器狗本地      → 接收结果 → 播放 speak_key 对应音频或交给上层决策

PerceptionGateway 的所有方法只返回结构化数据，不产生任何副作用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.types import (
    GaugeReading,
    InspectionReading,
    ZoneLetterResult,
)


class PerceptionGateway(ABC):
    """感知层抽象 —— 纯检测，不控制任何硬件。

    当前项目只保留巡检识别闭环：
    - RemotePerceptionGateway   (外接 NVIDIA 算力板，TCP 通信)
    """

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

    # ── 生命周期 ────────────────────────────────────────

    @abstractmethod
    def is_ready(self) -> bool:
        """感知层是否就绪（模型加载完成 / 远程连接建立）。"""
        ...
