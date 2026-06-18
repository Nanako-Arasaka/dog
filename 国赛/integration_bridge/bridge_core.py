#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core forwarding logic independent from ROS2."""

from __future__ import annotations

from typing import List, Optional, Protocol

from .event_logger import EventLogger
from .schemas import (
    InspectionResult,
    PlacementZoneResult,
    format_inspection_all,
    inspections_from_payload,
    placement_from_payload,
)


class BridgePublisher(Protocol):
    def publish_inspection_all(self, text: str) -> None:
        ...

    def publish_placement_zone(self, zone: str) -> None:
        ...


class IntegrationBridge:
    """Small bridge that normalizes, logs, and forwards competition events."""

    def __init__(
        self,
        publisher: Optional[BridgePublisher] = None,
        logger: Optional[EventLogger] = None,
    ):
        self.publisher = publisher
        self.logger = logger or EventLogger()
        self.inspection_memory = {}

    def handle_inspection_payload(self, payload: str) -> str:
        results = inspections_from_payload(payload)
        return self.handle_inspection_results(results)

    def handle_inspection_results(self, results: List[InspectionResult]) -> str:
        for result in results:
            self.inspection_memory[result.zone] = result
            self.logger.write(result.to_event())

        text = format_inspection_all(self.inspection_memory.values())
        self.logger.write({"type": "publish", "topic": "/inspection/all", "data": text})
        if self.publisher:
            self.publisher.publish_inspection_all(text)
        return text

    def handle_placement_payload(self, payload: str) -> str:
        result = placement_from_payload(payload)
        return self.handle_placement_zone(result)

    def handle_placement_zone(self, result: PlacementZoneResult) -> str:
        self.logger.write(result.to_event())
        self.logger.write(
            {"type": "publish", "topic": "/placement/recognized_zone", "data": result.zone}
        )
        if self.publisher:
            self.publisher.publish_placement_zone(result.zone)
        return result.zone
