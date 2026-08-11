#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze final A/B/C/D inspection results before arm task handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .schemas import InspectionResult, ZONES, format_inspection_all, format_inspection_all_detailed


@dataclass
class ZoneCandidate:
    result: InspectionResult
    count: int = 1


class InspectionFreezer:
    """Collect stable zone states and freeze the final four-zone result.

    The live detector publishes continuously. This class prevents a partial
    result, such as only "A:abnormal", from triggering the arm before all four
    inspection zones have been observed with stable states.
    """

    def __init__(self, stable_count: int = 3, required_zones: Iterable[str] = ZONES):
        self.stable_count = max(1, int(stable_count))
        self.required_zones = tuple(required_zones)
        self._candidates: Dict[str, ZoneCandidate] = {}
        self._frozen: Dict[str, InspectionResult] = {}

    def reset(self) -> None:
        self._candidates.clear()
        self._frozen.clear()

    def update(self, result: InspectionResult) -> bool:
        """Update one zone result.

        Returns True when this update freezes a zone for the first time.
        """
        zone = result.zone
        if zone not in self.required_zones or zone in self._frozen:
            return False
        if result.zone_state == "unknown":
            return False

        key = self._key(result)
        candidate = self._candidates.get(zone)
        if candidate is None or self._key(candidate.result) != key:
            self._candidates[zone] = ZoneCandidate(result=result, count=1)
            if self.stable_count <= 1:
                self._frozen[zone] = result
                return True
            return False

        candidate.count += 1
        candidate.result = result
        if candidate.count >= self.stable_count:
            self._frozen[zone] = result
            return True
        return False

    def is_complete(self) -> bool:
        return all(zone in self._frozen for zone in self.required_zones)

    def frozen_text(self) -> str:
        return format_inspection_all(self._frozen.values())

    def frozen_text_detailed(self) -> str:
        """冻结结果（保留 low/high 区分），形如 'A:low,B:normal,C:high,D:normal'。

        供 /inspection/all_detailed 发布、语音播报节点消费。
        与 frozen_text() 共用同一份冻结数据，二者同时发布、同时更新。
        """
        return format_inspection_all_detailed(self._frozen.values())

    def progress_text(self) -> str:
        parts = []
        for zone in self.required_zones:
            if zone in self._frozen:
                parts.append(f"{zone}:frozen:{self._frozen[zone].zone_state}")
            elif zone in self._candidates:
                cand = self._candidates[zone]
                parts.append(f"{zone}:pending:{cand.result.zone_state}:{cand.count}/{self.stable_count}")
            else:
                parts.append(f"{zone}:missing")
        return ",".join(parts)

    @staticmethod
    def _key(result: InspectionResult) -> Tuple[str, str]:
        return result.zone_state, result.gauge_status
