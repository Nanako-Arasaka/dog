from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.competition_flow_mock import (  # noqa: E402
    DEFAULT_INSPECTION,
    CompetitionFlowMock,
    run_default_flow,
)


def event_names(flow):
    return [event.event for event in flow.events]


def test_mock_flow_generates_two_abnormal_targets():
    flow = CompetitionFlowMock()

    targets = flow.freeze_inspection(DEFAULT_INSPECTION)

    assert targets == ["A", "C"]
    assert flow.target_zones == ["A", "C"]
    assert flow.current_target == "A"
    assert flow.inspection_text == "A:abnormal,B:normal,C:abnormal,D:normal"


def test_mock_flow_ignores_mismatched_placement_zone():
    flow = CompetitionFlowMock()
    flow.freeze_inspection(DEFAULT_INSPECTION)
    flow.run_next_grasp()

    accepted = flow.observe_placement("B")

    assert accepted is False
    assert flow.current_target == "A"
    assert flow.completed_zones == []
    assert flow.state == flow.WAITING_PLACE_ZONE
    ignored = flow.events[-1]
    assert ignored.event == "placement_ignored"
    assert ignored.placement_zone == "B"
    assert ignored.accepted is False


def test_mock_flow_runs_two_grasp_and_place_rounds():
    flow = run_default_flow()

    assert flow.state == flow.FINISHED
    assert flow.completed_zones == ["A", "C"]
    assert flow.current_target is None

    names = event_names(flow)
    assert names.count("grasp_started") == 2
    assert names.count("grasp_success") == 2
    assert names.count("waiting_place_zone") == 2
    assert names.count("placement_matched") == 2
    assert names.count("place_success") == 2
    assert "next_target" in names
    assert names[-1] == "task_done"


def test_first_target_completion_advances_to_second_target():
    flow = CompetitionFlowMock()
    flow.freeze_inspection(DEFAULT_INSPECTION)
    flow.run_next_grasp()

    assert flow.observe_placement("A") is True

    assert flow.completed_zones == ["A"]
    assert flow.current_target == "C"
    assert flow.state == flow.READY_TO_GRASP
    assert flow.events[-1].event == "next_target"
    assert flow.events[-1].current_target == "C"


def test_second_target_completion_finishes_task():
    flow = CompetitionFlowMock()
    flow.freeze_inspection(DEFAULT_INSPECTION)
    flow.run_next_grasp()
    assert flow.observe_placement("A") is True
    flow.run_next_grasp()

    assert flow.observe_placement("D") is False
    assert flow.observe_placement("C") is True

    assert flow.completed_zones == ["A", "C"]
    assert flow.state == flow.FINISHED
    assert flow.events[-1].event == "task_done"
