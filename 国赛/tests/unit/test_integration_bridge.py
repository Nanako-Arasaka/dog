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
    assert format_inspection_all(results) == "A:abnormal,B:unknown,C:unknown,D:unknown"


def test_parse_compact_inspection_text():
    results = inspections_from_payload("A:abnormal,B:normal,C:unknown,D:normal")
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
