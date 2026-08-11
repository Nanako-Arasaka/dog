#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data normalization for the integration bridge.

This layer intentionally keeps only light validation and format conversion.
Vision inference, navigation planning, arm motion, and grasp decisions belong
to their own modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional


ZONES = ("A", "B", "C", "D")
ZONE_CLASS_TO_LETTER = {
    "zone_A": "A",
    "zone_B": "B",
    "zone_C": "C",
    "zone_D": "D",
}
GAUGE_TO_ZONE_STATE = {
    "low": "abnormal",
    "normal": "normal",
    "high": "abnormal",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class InspectionResult:
    zone: str
    gauge_status: str = "unknown"
    abnormal: Optional[bool] = None
    confidence: Optional[float] = None
    timestamp: float = 0.0

    @property
    def zone_state(self) -> str:
        if self.abnormal is True:
            return "abnormal"
        if self.abnormal is False:
            return "normal"
        return GAUGE_TO_ZONE_STATE.get(self.gauge_status, "unknown")

    @property
    def zone_state_detailed(self) -> str:
        """保留 偏低/偏高 区分的状态词，供语音播报使用。

        与 zone_state 的区别：
          - zone_state        把 low/high 折叠成 "abnormal"（FSM 依赖此契约，勿改）
          - zone_state_detailed  保留 "low" / "high"（语音规则要求黄=偏低、红=偏高分别播报）

        退化情况（gauge_status 不是 low/normal/high，例如 compact 文本 "A:abnormal"）：
        回退到 abnormal/normal/unknown，此时无法区分偏低/偏高，会丢状态分。
        生产链路（live_detect_yolo_opencv.py 发 JSON 带 gauge_status）不会退化。
        """
        if self.gauge_status in ("low", "normal", "high"):
            return self.gauge_status
        if self.abnormal is True:
            return "abnormal"
        if self.abnormal is False:
            return "normal"
        return "unknown"

    def to_event(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "inspection_result"
        data["zone_state"] = self.zone_state
        data["zone_state_detailed"] = self.zone_state_detailed
        return data


@dataclass(frozen=True)
class PlacementZoneResult:
    zone: str
    confidence: Optional[float] = None
    timestamp: float = 0.0

    def to_event(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = "placement_zone"
        return data


def now_ts() -> float:
    return time.time()


def normalize_zone(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    if text in ZONE_CLASS_TO_LETTER:
        return ZONE_CLASS_TO_LETTER[text]
    upper = text.upper()
    if upper.startswith("ZONE_"):
        upper = upper.split("_", 1)[1]
    if upper in ZONES:
        return upper
    return None


def normalize_gauge_status(value: Any, abnormal: Optional[bool] = None) -> str:
    if abnormal is False and value in (None, ""):
        return "normal"
    text = str(value or "unknown").strip().lower()
    aliases = {
        "偏低": "low",
        "低": "low",
        "low": "low",
        "正常": "normal",
        "normal": "normal",
        "偏高": "high",
        "高": "high",
        "high": "high",
        "未知": "unknown",
        "unknown": "unknown",
    }
    return aliases.get(text, "unknown")


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "abnormal"):
        return True
    if text in ("false", "0", "no", "n", "normal"):
        return False
    return None


def parse_json_or_text(payload: str) -> Any:
    text = payload.strip()
    if not text:
        raise ValueError("empty payload")
    if text[0] in "[{":
        return json.loads(text)
    return text


def inspection_from_mapping(data: Mapping[str, Any]) -> InspectionResult:
    zone = normalize_zone(data.get("zone") or data.get("letter") or data.get("zone_id"))
    if zone is None:
        raise ValueError(f"invalid inspection zone: {data!r}")
    abnormal = parse_bool(data.get("abnormal"))
    gauge_status = normalize_gauge_status(
        data.get("gauge_status") or data.get("status"),
        abnormal=abnormal,
    )
    timestamp = float(data.get("timestamp") or now_ts())
    confidence = data.get("confidence")
    return InspectionResult(
        zone=zone,
        gauge_status=gauge_status,
        abnormal=abnormal,
        confidence=float(confidence) if confidence is not None else None,
        timestamp=timestamp,
    )


def inspections_from_payload(payload: str) -> List[InspectionResult]:
    data = parse_json_or_text(payload)
    if isinstance(data, list):
        return [inspection_from_mapping(item) for item in data]
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return [inspection_from_mapping(item) for item in data["results"]]
        return [inspection_from_mapping(data)]
    return inspections_from_compact(str(data))


def inspections_from_compact(text: str) -> List[InspectionResult]:
    results: List[InspectionResult] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid compact inspection item: {part!r}")
        zone_text, state_text = part.split(":", 1)
        zone = normalize_zone(zone_text)
        if zone is None:
            raise ValueError(f"invalid compact inspection zone: {zone_text!r}")
        state = state_text.strip().lower()
        if state in ("normal", "abnormal", "unknown"):
            abnormal = True if state == "abnormal" else False if state == "normal" else None
            gauge_status = "normal" if state == "normal" else "unknown"
        else:
            gauge_status = normalize_gauge_status(state)
            abnormal = None
        results.append(
            InspectionResult(
                zone=zone,
                gauge_status=gauge_status,
                abnormal=abnormal,
                timestamp=now_ts(),
            )
        )
    return results


def placement_from_payload(payload: str) -> PlacementZoneResult:
    data = parse_json_or_text(payload)
    if isinstance(data, dict):
        zone = normalize_zone(data.get("zone") or data.get("zone_id") or data.get("letter"))
        confidence = data.get("confidence")
        timestamp = float(data.get("timestamp") or now_ts())
    else:
        zone = normalize_zone(str(data))
        confidence = None
        timestamp = now_ts()
    if zone is None:
        raise ValueError(f"invalid placement zone payload: {payload!r}")
    return PlacementZoneResult(
        zone=zone,
        confidence=float(confidence) if confidence is not None else None,
        timestamp=timestamp,
    )


def format_inspection_all(results: Iterable[InspectionResult]) -> str:
    zone_states = {zone: "unknown" for zone in ZONES}
    for result in results:
        zone_states[result.zone] = result.zone_state
    return ",".join(f"{zone}:{zone_states[zone]}" for zone in ZONES)


def format_inspection_all_detailed(results: Iterable[InspectionResult]) -> str:
    """与 format_inspection_all 同构，但保留 low/high 区分。

    输出形如 "A:low,B:normal,C:high,D:normal"。
    供语音播报节点订阅 /inspection/all_detailed，避免被 zone_state 折叠成 abnormal。
    FSM 仍消费 /inspection/all（abnormal/normal），二者互不影响。
    """
    zone_states = {zone: "unknown" for zone in ZONES}
    for result in results:
        zone_states[result.zone] = result.zone_state_detailed
    return ",".join(f"{zone}:{zone_states[zone]}" for zone in ZONES)
