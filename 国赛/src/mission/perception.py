"""感知层接口 + JSON 仿真实现。

PerceptionGateway ABC 主定义在 perception/gateway.py。
此文件保留 JsonScenarioPerception（JSON 仿真感知实现）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.types import (
    BBox,
    ConeDetection,
    GaugeReading,
    InspectionReading,
    MeterStatus,
    StripDetection,
    TargetPose,
    Zone,
    ZoneLetterResult,
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

    纯检测，不执行任何运动/机械臂操作。
    """

    def __init__(self, cfg: PerceptionConfig) -> None:
        scenario = json.loads(Path(cfg.scenario_file).read_text(encoding="utf-8"))

        # 避障配置
        obstacle = scenario.get("obstacle", {})
        self._obstacle_clear_after_ticks: int = int(obstacle.get("clear_after_ticks", 60))
        self._obstacle_ticks: int = 0

        # 锥桶 mock
        cones_data = scenario.get("cones", [])
        self._mock_cones: list[ConeDetection] = []
        for c in cones_data:
            self._mock_cones.append(ConeDetection(
                bbox=BBox(c.get("x1", 0), c.get("y1", 0), c.get("x2", 100), c.get("y2", 200)),
                center_3d=(float(c.get("x", 0)), float(c.get("y", 0)), float(c.get("z", 1.5))),
                confidence=float(c.get("confidence", 0.9)),
            ))
        self._cones_returned: bool = False

        # 巡检数据
        readings = scenario.get("inspection_readings", [])
        self._inspection_queue: list[InspectionReading] = []
        for item in readings:
            zone_str = str(item["zone"]).upper()
            if zone_str not in ("A", "B", "C", "D"):
                continue
            status = MeterStatus(str(item["status"]).lower())
            self._inspection_queue.append(InspectionReading(
                zone=Zone(zone_str),
                meter_status=status,
                confidence=float(item.get("confidence", 0.95)),
            ))
        self._inspection_cursor: int = 0

        # 抓取结果序列（供测试驱动 MockArm 行为用，不再由此类执行）
        pickup_seq: dict = scenario.get("pickup_outcomes", {})
        self._pickup_outcomes: dict[str, list[str]] = {}
        for zone_str, outcomes in pickup_seq.items():
            z = str(zone_str).upper()
            if z in ("A", "B", "C", "D"):
                self._pickup_outcomes[z] = [str(x).lower() for x in outcomes]
        self._pickup_cursors: dict[str, int] = {z: 0 for z in self._pickup_outcomes}

        # 红色长条 mock
        strips_data = scenario.get("red_strips", [])
        self._mock_strips: list[StripDetection] = []
        for s in strips_data:
            self._mock_strips.append(StripDetection(
                bbox=BBox(s.get("x1", 0), s.get("y1", 0), s.get("x2", 100), s.get("y2", 100)),
                center_3d=(
                    float(s.get("x", 0.05)), float(s.get("y", 0.0)), float(s.get("z", 0.3)),
                ),
                confidence=float(s.get("confidence", 0.9)),
                timestamp=time.time(),
            ))

    # ── 避障 ──────────────────────────────────────────

    def detect_obstacles(self, rgb: np.ndarray | None = None) -> list[ConeDetection]:
        if self._obstacle_ticks < self._obstacle_clear_after_ticks and not self._cones_returned:
            if self._mock_cones:
                self._cones_returned = True
                return list(self._mock_cones)
        return []

    def obstacle_cleared(self) -> bool:
        self._obstacle_ticks += 1
        return self._obstacle_ticks >= self._obstacle_clear_after_ticks

    # ── 巡检 ──────────────────────────────────────────

    def detect_zone_letters(self, rgb: np.ndarray | None = None) -> list[ZoneLetterResult]:
        return [
            ZoneLetterResult(zone=Zone("A"), confidence=0.95, timestamp=time.time()),
            ZoneLetterResult(zone=Zone("B"), confidence=0.93, timestamp=time.time()),
            ZoneLetterResult(zone=Zone("C"), confidence=0.94, timestamp=time.time()),
            ZoneLetterResult(zone=Zone("D"), confidence=0.96, timestamp=time.time()),
        ]

    def detect_gauges(self, rgb: np.ndarray | None = None) -> list[GaugeReading]:
        results: list[GaugeReading] = []
        for item in self._inspection_queue:
            results.append(GaugeReading(
                zone=item.zone,
                status=item.meter_status,
                confidence=item.confidence,
                raw_value=item.meter_raw_value,
                timestamp=time.time(),
            ))
        return results

    def poll_inspection(self) -> list[InspectionReading]:
        if self._inspection_cursor >= len(self._inspection_queue):
            return []
        item = self._inspection_queue[self._inspection_cursor]
        self._inspection_cursor += 1
        return [item]

    # ── 抓取 ──────────────────────────────────────────

    def detect_red_strips(self, rgb: np.ndarray | None = None) -> list[StripDetection]:
        return list(self._mock_strips) if self._mock_strips else [
            StripDetection(
                bbox=BBox(200, 150, 400, 300),
                center_3d=(0.05, 0.0, 0.25),
                confidence=0.9,
                timestamp=time.time(),
            )
        ]

    def estimate_target_pose(self, rgb: np.ndarray | None = None) -> TargetPose | None:
        strips = self.detect_red_strips()
        if not strips:
            return None
        s = strips[0]
        return TargetPose(
            x=s.center_3d[0], y=s.center_3d[1], z=s.center_3d[2],
            confidence=s.confidence,
            timestamp=time.time(),
        )

    # ── 生命周期 ──────────────────────────────────────

    def is_ready(self) -> bool:
        return True

    # ── 测试辅助 ──────────────────────────────────────

    def consume_pickup_outcome(self, zone: str) -> str:
        """供 Mission 读取 mock 抓取结果序列（用于测试）"""
        outcomes = self._pickup_outcomes.get(zone.upper(), [])
        cursor = self._pickup_cursors.get(zone.upper(), 0)
        if cursor < len(outcomes):
            outcome = outcomes[cursor]
            self._pickup_cursors[zone.upper()] = cursor + 1
            return outcome
        return "success"


__all__ = [
    "PerceptionGateway",
    "PerceptionConfig",
    "JsonScenarioPerception",
]
