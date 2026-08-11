from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.inspection_pick_place_coordinator import (  # noqa: E402
    NO_ABNORMAL,
    READY_TO_START_PICK,
    WAITING_INSPECTION,
    WAITING_PICK_AREA,
    InspectionPickPlaceCoordinator,
)


def test_records_normal_and_abnormal_zones_before_pick_area():
    coordinator = InspectionPickPlaceCoordinator()

    snapshot = coordinator.update_inspection("A:abnormal,B:normal,C:abnormal,D:normal")

    assert snapshot.state == WAITING_PICK_AREA
    assert snapshot.zone_states == {
        "A": "abnormal",
        "B": "normal",
        "C": "abnormal",
        "D": "normal",
    }
    assert snapshot.abnormal_targets == ["A", "C"]
    assert snapshot.target_text == "A,C"


def test_waits_until_all_zones_are_known():
    coordinator = InspectionPickPlaceCoordinator()

    snapshot = coordinator.update_inspection("A:abnormal,B:normal,C:unknown,D:normal")

    assert snapshot.state == WAITING_INSPECTION
    assert snapshot.abnormal_targets == ["A"]
    assert "C" in snapshot.message


def test_pick_ready_starts_abnormal_target_queue():
    coordinator = InspectionPickPlaceCoordinator()
    coordinator.update_inspection("A:abnormal,B:normal,C:abnormal,D:normal")

    snapshot = coordinator.mark_pick_area_ready()

    assert snapshot.state == READY_TO_START_PICK
    assert snapshot.target_text == "A,C"


def test_all_normal_skips_red_bar_task():
    coordinator = InspectionPickPlaceCoordinator()

    snapshot = coordinator.update_inspection("A:normal,B:normal,C:normal,D:normal")
    ready = coordinator.mark_pick_area_ready()

    assert snapshot.state == NO_ABNORMAL
    assert ready.state == NO_ABNORMAL
    assert ready.target_text == ""
