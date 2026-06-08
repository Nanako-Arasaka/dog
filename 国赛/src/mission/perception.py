"""感知层接口 + JSON 仿真实现。

PerceptionGateway ABC 的主定义在 perception/gateway.py 中。
此文件保留：
- 向后兼容的 PerceptionGateway re-export
- JsonScenarioPerception（JSON 仿真感知实现）
- PerceptionConfig（仿真配置）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
from perception.gateway import PerceptionGateway

# ── 配置 ────────────────────────────────────────────────


@dataclass(frozen=True)
class PerceptionConfig:
    """仿真感知配置。"""
    scenario_file: str


# ── JSON 仿真实现 ────────────────────────────────────────


class JsonScenarioPerception(PerceptionGateway):
    """基于 JSON 场景文件的仿真感知。

    用于不连真机时的状态机联调。
    """

    def __init__(self, cfg: PerceptionConfig) -> None:
        scenario = json.loads(Path(cfg.scenario_file).read_text(encoding="utf-8"))

        # 避障配置
        obstacle = scenario.get("obstacle", {})
        self._obstacle_clear_after_ticks: int = int(obstacle.get("clear_after_ticks", 60))
        self._obstacle_ticks: int = 0

        # 巡检读数
        readings = scenario.get("inspection_readings", [])
        self._inspection_readings: list[InspectionReading] = []
        for item in readings:
            zone_str = str(item["zone"]).upper()
            if zone_str not in ("A", "B", "C", "D"):
                continue
            status = MeterStatus(str(item["status"]).lower())
            self._inspection_readings.append(
                InspectionReading(
                    zone=Zone(zone_str),
                    meter_status=status,
                    confidence=0.95,
                )
            )
        self._inspection_cursor: int = 0

        # 抓取结果序列
        pickup_seq: dict = scenario.get("pickup_outcomes", {})
        self._pickup_outcomes: dict[str, list[str]] = {}
        for zone_str, outcomes in pickup_seq.items():
            z = str(zone_str).upper()
            if z not in ("A", "B", "C", "D"):
                continue
            self._pickup_outcomes[z] = [str(x).lower() for x in outcomes]

        # 模拟锥桶数据
        cones_data = scenario.get("cones", [])
        self._mock_cones: list[ConeDetection] = []
        for c in cones_data:
            self._mock_cones.append(ConeDetection(
                bbox=BBox(c.get("x1", 0), c.get("y1", 0), c.get("x2", 100), c.get("y2", 200)),
                center_3d=(float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 1.5))),
                confidence=float(c.get("confidence", 0.9)),
            ))

        self._cones_returned: bool = False

    # ── 避障 ──────────────────────────────────────────

    def detect_cones(self, rgb: np.ndarray, depth: np.ndarray) -> list[ConeDetection]:
        if self._obstacle_ticks < self._obstacle_clear_after_ticks and not self._cones_returned:
            if self._mock_cones:
                self._cones_returned = True
                return list(self._mock_cones)
        return []

    def obstacle_cleared(self) -> bool:
        self._obstacle_ticks += 1
        return self._obstacle_ticks >= self._obstacle_clear_after_ticks

    # ── 巡检 ──────────────────────────────────────────

    def detect_equipment(self, rgb: np.ndarray) -> list[EquipmentDetection]:
        """仿真：返回固定设备检测结果。"""
        return [
            EquipmentDetection(bbox=BBox(100, 100, 300, 400), equipment_type="power_cabinet", zone_letter="A", zone_confidence=0.95),
            EquipmentDetection(bbox=BBox(350, 100, 550, 400), equipment_type="transformer", zone_letter="B", zone_confidence=0.93),
            EquipmentDetection(bbox=BBox(100, 420, 300, 700), equipment_type="power_cabinet", zone_letter="C", zone_confidence=0.94),
            EquipmentDetection(bbox=BBox(350, 420, 550, 700), equipment_type="transformer", zone_letter="D", zone_confidence=0.96),
        ]

    def read_zone_letter(self, rgb: np.ndarray, roi: BBox) -> tuple[str, float]:
        """仿真：返回 ROI 对应的区域字母（由 detect_equipment 预处理）。"""
        return ("A", 0.95)

    def read_meter(self, rgb: np.ndarray, roi: BBox) -> tuple[MeterStatus, float, float | None]:
        """仿真：返回仪表状态。"""
        return (MeterStatus.NORMAL, 0.9, 50.0)

    def poll_inspection(self) -> list[InspectionReading]:
        if self._inspection_cursor >= len(self._inspection_readings):
            return []
        item = self._inspection_readings[self._inspection_cursor]
        self._inspection_cursor += 1
        return [item]

    # ── 抓取 ──────────────────────────────────────────

    def detect_red_strip(self, rgb: np.ndarray, depth: np.ndarray) -> StripDetection | None:
        """仿真：假设长条在视野中心。"""
        h, w = rgb.shape[:2]
        return StripDetection(
            bbox=BBox(w // 3, h // 3, 2 * w // 3, 2 * h // 3),
            center_3d=(0.0, 0.0, 0.5),
            confidence=0.9,
        )

    def check_drop(self, rgb: np.ndarray) -> bool:
        """仿真：不检测掉落。"""
        return False

    def execute_pickup_for_zone(self, zone: str) -> str:
        z = zone.upper()
        queue = self._pickup_outcomes.get(z, [])
        if queue:
            return queue.pop(0)
        return "success"


__all__ = [
    "PerceptionGateway",
    "PerceptionConfig",
    "JsonScenarioPerception",
]
