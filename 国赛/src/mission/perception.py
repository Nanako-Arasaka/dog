from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mission.models import InspectionReading, MeterStatus, PickupOutcome, VALID_ZONES


@dataclass(frozen=True)
class PerceptionConfig:
    scenario_file: str


class PerceptionGateway:
    """
    感知层接口。后续可替换成真实视觉/点云/定位模块。
    """

    def obstacle_cleared(self) -> bool:
        raise NotImplementedError

    def poll_inspection(self) -> list[InspectionReading]:
        raise NotImplementedError

    def execute_pickup_for_zone(self, zone: str) -> PickupOutcome:
        raise NotImplementedError


class JsonScenarioPerception(PerceptionGateway):
    def __init__(self, cfg: PerceptionConfig) -> None:
        scenario = json.loads(Path(cfg.scenario_file).read_text(encoding="utf-8"))
        obstacle = scenario.get("obstacle", {})
        self._obstacle_clear_after_ticks = int(obstacle.get("clear_after_ticks", 60))
        self._obstacle_ticks = 0

        readings = scenario.get("inspection_readings", [])
        self._inspection_readings: list[InspectionReading] = []
        for item in readings:
            zone = str(item["zone"]).upper()
            if zone not in VALID_ZONES:
                continue
            status_raw = str(item["status"]).lower()
            status = MeterStatus(status_raw)
            self._inspection_readings.append(InspectionReading(zone=zone, status=status))

        self._inspection_cursor = 0
        pickup_seq: dict[str, list[str]] = scenario.get("pickup_outcomes", {})
        self._pickup_outcomes: dict[str, list[PickupOutcome]] = {}
        for zone, outcomes in pickup_seq.items():
            z = str(zone).upper()
            if z not in VALID_ZONES:
                continue
            self._pickup_outcomes[z] = [PickupOutcome(str(x).lower()) for x in outcomes]

    def obstacle_cleared(self) -> bool:
        self._obstacle_ticks += 1
        return self._obstacle_ticks >= self._obstacle_clear_after_ticks

    def poll_inspection(self) -> list[InspectionReading]:
        if self._inspection_cursor >= len(self._inspection_readings):
            return []
        item = self._inspection_readings[self._inspection_cursor]
        self._inspection_cursor += 1
        return [item]

    def execute_pickup_for_zone(self, zone: str) -> PickupOutcome:
        z = zone.upper()
        queue = self._pickup_outcomes.get(z, [])
        if queue:
            return queue.pop(0)
        return PickupOutcome.SUCCESS
