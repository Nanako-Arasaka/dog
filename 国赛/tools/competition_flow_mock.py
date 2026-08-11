#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run competition flow for the national competition glue layer.

This script deliberately avoids ROS2, cameras, arm hardware, and robot motion.
It uses the same inspection and placement payload parsers as integration_bridge
so the mock flow stays close to the real topic formats:

  /inspection/all              "A:abnormal,B:normal,C:abnormal,D:normal"
  /inspection/target_zones     "A,C"
  /placement/recognized_zone   "A"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration_bridge.schemas import (  # noqa: E402
    InspectionResult,
    format_inspection_all,
    inspections_from_payload,
    placement_from_payload,
)


DEFAULT_INSPECTION = "A:abnormal,B:normal,C:abnormal,D:normal"
DEFAULT_PLACEMENTS = ("B", "A", "D", "C")
FINISHED_STATES = {"DONE", "FINISHED"}


@dataclass(frozen=True)
class FlowEvent:
    event: str
    state: str
    topic: Optional[str] = None
    data: Optional[str] = None
    target_zones: Optional[List[str]] = None
    current_target: Optional[str] = None
    placement_zone: Optional[str] = None
    accepted: Optional[bool] = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


class CompetitionFlowMock:
    """Small dry-run state machine for inspection -> grasp -> place."""

    WAITING_INSPECTION = "WAITING_INSPECTION"
    READY_TO_GRASP = "READY_TO_GRASP"
    GRASPING = "GRASPING"
    WAITING_PLACE_ZONE = "WAITING_PLACE_ZONE"
    FINISHED = "FINISHED"

    def __init__(self) -> None:
        self.state = self.WAITING_INSPECTION
        self.inspection_text = ""
        self.inspection_results: List[InspectionResult] = []
        self.target_zones: List[str] = []
        self.completed_zones: List[str] = []
        self.current_index = 0
        self.events: List[FlowEvent] = []

    @property
    def current_target(self) -> Optional[str]:
        if self.current_index < len(self.target_zones):
            return self.target_zones[self.current_index]
        return None

    def freeze_inspection(self, payload: str) -> List[str]:
        self.inspection_results = inspections_from_payload(payload)
        self.inspection_text = format_inspection_all(self.inspection_results)
        self.target_zones = [
            result.zone for result in sorted(self.inspection_results, key=lambda item: item.zone)
            if result.zone_state == "abnormal"
        ]
        self.current_index = 0
        self.completed_zones = []
        self.state = self.READY_TO_GRASP if self.target_zones else self.FINISHED
        self._log(
            "inspection_frozen",
            topic="/inspection/all",
            data=self.inspection_text,
            target_zones=list(self.target_zones),
            current_target=self.current_target,
        )
        self._log(
            "target_zones",
            topic="/inspection/target_zones",
            data=",".join(self.target_zones),
            target_zones=list(self.target_zones),
            current_target=self.current_target,
        )
        if not self.target_zones:
            self._log("task_done")
        return list(self.target_zones)

    def run_next_grasp(self) -> Optional[str]:
        target = self.current_target
        if target is None:
            self.state = self.FINISHED
            self._log("task_done")
            return None
        if self.state not in {self.READY_TO_GRASP, self.WAITING_PLACE_ZONE}:
            raise RuntimeError(f"cannot start grasp while state={self.state}")

        self.state = self.GRASPING
        self._log("current_target", current_target=target)
        self._log("grasp_started", topic="/task/direct_grasp", data="red", current_target=target)
        self._log("grasp_success", topic="/arm/feedback", data="direct_grasp|success|mock", current_target=target)
        self.state = self.WAITING_PLACE_ZONE
        self._log("waiting_place_zone", topic="/placement/recognized_zone", current_target=target)
        return target

    def observe_placement(self, payload: str) -> bool:
        result = placement_from_payload(payload)
        target = self.current_target
        if target is None:
            self.state = self.FINISHED
            self._log("task_done")
            return False

        matched = result.zone == target
        if not matched:
            self._log(
                "placement_ignored",
                topic="/placement/recognized_zone",
                data=result.zone,
                current_target=target,
                placement_zone=result.zone,
                accepted=False,
            )
            return False

        self._log(
            "placement_matched",
            topic="/placement/recognized_zone",
            data=result.zone,
            current_target=target,
            placement_zone=result.zone,
            accepted=True,
        )
        self._log("place_success", topic="/arm/feedback", data="place|success|mock", current_target=target)
        self.completed_zones.append(target)
        self.current_index += 1

        next_target = self.current_target
        if next_target is None:
            self.state = self.FINISHED
            self._log("task_done")
        else:
            self.state = self.READY_TO_GRASP
            self._log("next_target", current_target=next_target)
        return True

    def run(self, inspection_payload: str, placements: Iterable[str]) -> "CompetitionFlowMock":
        self.freeze_inspection(inspection_payload)
        while self.state != self.FINISHED:
            self.run_next_grasp()
            matched = False
            for placement in placements:
                if self.observe_placement(placement):
                    matched = True
                    break
            if not matched:
                raise RuntimeError(f"no placement matched current_target={self.current_target}")
        return self

    def event_dicts(self) -> List[dict]:
        return [event.to_dict() for event in self.events]

    def _log(
        self,
        event: str,
        *,
        topic: Optional[str] = None,
        data: Optional[str] = None,
        target_zones: Optional[List[str]] = None,
        current_target: Optional[str] = None,
        placement_zone: Optional[str] = None,
        accepted: Optional[bool] = None,
    ) -> None:
        self.events.append(
            FlowEvent(
                event=event,
                state=self.state,
                topic=topic,
                data=data,
                target_zones=target_zones,
                current_target=current_target,
                placement_zone=placement_zone,
                accepted=accepted,
            )
        )


def run_default_flow() -> CompetitionFlowMock:
    flow = CompetitionFlowMock()
    flow.run(DEFAULT_INSPECTION, DEFAULT_PLACEMENTS)
    return flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run national competition task flow.")
    parser.add_argument("--inspection", default=DEFAULT_INSPECTION)
    parser.add_argument(
        "--placements",
        default=",".join(DEFAULT_PLACEMENTS),
        help="Comma-separated placement observations, for example B,A,D,C.",
    )
    parser.add_argument("--jsonl", action="store_true", help="Print machine-readable JSON lines.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    placements = [item.strip() for item in args.placements.split(",") if item.strip()]
    flow = CompetitionFlowMock().run(args.inspection, placements)
    for event in flow.event_dicts():
        if args.jsonl:
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))
        else:
            details = " ".join(f"{key}={value}" for key, value in event.items() if key not in {"event", "state"})
            print(f"{event['event']} state={event['state']} {details}".rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
