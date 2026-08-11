import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration_bridge.bridge_core import IntegrationBridge
from integration_bridge.inspection_freezer import InspectionFreezer
from integration_bridge.schemas import (
    format_inspection_all,
    format_inspection_all_detailed,
    inspection_from_mapping,
    inspections_from_payload,
    placement_from_payload,
)


class FakePublisher:
    def __init__(self):
        self.inspection_all = []
        self.placement_zones = []

    def publish_inspection_all(self, text):
        self.inspection_all.append(text)

    def publish_placement_zone(self, zone):
        self.placement_zones.append(zone)


def test_parse_single_inspection_json_to_abnormal():
    results = inspections_from_payload(
        json.dumps({"zone": "zone_A", "gauge_status": "high", "abnormal": True})
    )
    assert results[0].gauge_status == "high"
    assert format_inspection_all(results) == "A:abnormal,B:unknown,C:unknown,D:unknown"


def test_abnormal_does_not_overwrite_low_or_high_gauge_status():
    low_result = inspection_from_mapping({"zone": "A", "gauge_status": "low", "abnormal": True})
    high_result = inspection_from_mapping({"zone": "C", "gauge_status": "high", "abnormal": True})
    normal_result = inspection_from_mapping({"zone": "B", "gauge_status": "normal", "abnormal": False})

    assert low_result.gauge_status == "low"
    assert low_result.zone_state == "abnormal"
    assert high_result.gauge_status == "high"
    assert high_result.zone_state == "abnormal"
    assert normal_result.gauge_status == "normal"
    assert normal_result.zone_state == "normal"


def test_abnormal_without_gauge_status_keeps_status_unknown():
    result = inspection_from_mapping({"zone": "A", "abnormal": True})

    assert result.gauge_status == "unknown"
    assert result.zone_state == "abnormal"


def test_parse_compact_inspection_text():
    results = inspections_from_payload("A:abnormal,B:normal,C:unknown,D:normal")
    assert results[0].gauge_status == "unknown"
    assert results[0].zone_state == "abnormal"
    assert format_inspection_all(results) == "A:abnormal,B:normal,C:unknown,D:normal"


def test_parse_placement_zone_alias():
    result = placement_from_payload('{"zone":"zone_C","confidence":0.91}')
    assert result.zone == "C"
    assert result.confidence == 0.91


def test_bridge_forwards_to_expected_topics():
    publisher = FakePublisher()
    bridge = IntegrationBridge(publisher=publisher)

    bridge.handle_inspection_payload('{"zone":"A","gauge_status":"low"}')
    bridge.handle_placement_payload("zone_A")

    assert publisher.inspection_all == ["A:abnormal,B:unknown,C:unknown,D:unknown"]
    assert publisher.placement_zones == ["A"]


def test_inspection_freezer_waits_for_stable_all_zones():
    freezer = InspectionFreezer(stable_count=2)
    payloads = [
        {"zone": "A", "gauge_status": "high", "abnormal": True},
        {"zone": "B", "gauge_status": "normal", "abnormal": False},
        {"zone": "C", "gauge_status": "low", "abnormal": True},
        {"zone": "D", "gauge_status": "normal", "abnormal": False},
    ]

    for item in payloads:
        assert freezer.update(inspection_from_mapping(item)) is False
        assert freezer.is_complete() is False

    for index, item in enumerate(payloads):
        assert freezer.update(inspection_from_mapping(item)) is True
        if index < len(payloads) - 1:
            assert freezer.is_complete() is False

    assert freezer.is_complete() is True
    assert freezer.frozen_text() == "A:abnormal,B:normal,C:abnormal,D:normal"


def test_inspection_freezer_ignores_unknown_and_frozen_updates():
    freezer = InspectionFreezer(stable_count=1)
    assert freezer.update(inspection_from_mapping({"zone": "A", "gauge_status": "unknown"})) is False
    assert freezer.update(inspection_from_mapping({"zone": "A", "gauge_status": "high"})) is True
    assert freezer.update(inspection_from_mapping({"zone": "A", "gauge_status": "normal"})) is False
    assert freezer.frozen_text() == "A:abnormal,B:unknown,C:unknown,D:unknown"


def test_detailed_format_preserves_low_and_high():
    """detailed 格式保留 low/high 区分，供语音播报区分偏低/偏高。"""
    low = inspection_from_mapping({"zone": "A", "gauge_status": "low", "abnormal": True})
    high = inspection_from_mapping({"zone": "C", "gauge_status": "high", "abnormal": True})
    normal_b = inspection_from_mapping({"zone": "B", "gauge_status": "normal", "abnormal": False})
    normal_d = inspection_from_mapping({"zone": "D", "gauge_status": "normal", "abnormal": False})

    # 详细格式保留 low/high
    assert format_inspection_all_detailed([low, high, normal_b, normal_d]) == "A:low,B:normal,C:high,D:normal"
    # 同时 zone_state_detailed 也保留
    assert low.zone_state_detailed == "low"
    assert high.zone_state_detailed == "high"
    assert normal_b.zone_state_detailed == "normal"
    # 老契约不变：zone_state 仍折叠成 abnormal（FSM 依赖）
    assert low.zone_state == "abnormal"
    assert high.zone_state == "abnormal"
    assert format_inspection_all([low, high, normal_b, normal_d]) == "A:abnormal,B:normal,C:abnormal,D:normal"


def test_detailed_format_degrades_when_no_gauge_status():
    """compact 文本 'A:abnormal' 没有 gauge_status 时，detailed 退化为 abnormal（无 low/high 区分）。"""
    results = inspections_from_payload("A:abnormal,B:normal,C:abnormal,D:normal")
    assert format_inspection_all_detailed(results) == "A:abnormal,B:normal,C:abnormal,D:normal"
    assert results[0].zone_state_detailed == "abnormal"


def test_freezer_frozen_text_detailed_preserves_low_high():
    freezer = InspectionFreezer(stable_count=1)
    for item in [
        {"zone": "A", "gauge_status": "low", "abnormal": True},
        {"zone": "B", "gauge_status": "normal", "abnormal": False},
        {"zone": "C", "gauge_status": "high", "abnormal": True},
        {"zone": "D", "gauge_status": "normal", "abnormal": False},
    ]:
        freezer.update(inspection_from_mapping(item))
    assert freezer.is_complete() is True
    # 老 frozen_text 给 FSM（abnormal/normal）
    assert freezer.frozen_text() == "A:abnormal,B:normal,C:abnormal,D:normal"
    # 新 frozen_text_detailed 给语音播报（low/normal/high）
    assert freezer.frozen_text_detailed() == "A:low,B:normal,C:high,D:normal"
