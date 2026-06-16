from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_input import VisionFrame  # noqa: E402
from perception.detector.fixed_detector import (  # noqa: E402
    FixedDetectionConfig,
    FixedDetectionPipeline,
    fuse_inspection_results,
)
from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway  # noqa: E402


def _frame() -> VisionFrame:
    image = _gauge_image(210.0)
    _draw_letters(image)
    return VisionFrame(
        frame_id=1,
        timestamp=123.456,
        image=image,
        width=640,
        height=480,
        source_type="mock",
    )


def _gauge_frame(angle_deg: float, frame_id: int = 1) -> VisionFrame:
    image = _gauge_image(angle_deg)
    return VisionFrame(
        frame_id=frame_id,
        timestamp=123.456 + frame_id,
        image=image,
        width=640,
        height=480,
        source_type="mock",
    )


def _gauge_image(angle_deg: float) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cx, cy, radius = 320, 240, 120
    yy, xx = np.indices((480, 640))
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    image[disk] = (240, 240, 240)
    ring = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius) < 2.0
    image[ring] = (20, 20, 20)

    tip_x = int(cx + np.cos(np.deg2rad(angle_deg)) * radius * 0.78)
    tip_y = int(cy - np.sin(np.deg2rad(angle_deg)) * radius * 0.78)
    steps = max(abs(tip_x - cx), abs(tip_y - cy), 1)
    xs = np.linspace(cx, tip_x, steps).astype(np.int64)
    ys = np.linspace(cy, tip_y, steps).astype(np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            px = np.clip(xs + dx, 0, 639)
            py = np.clip(ys + dy, 0, 479)
            image[py, px] = (0, 0, 0)
    return image


def _draw_letters(image: np.ndarray) -> None:
    for letter, x, y in (("A", 28, 24), ("B", 120, 24), ("C", 28, 360), ("D", 120, 360)):
        mask = _letter_mask(letter, 56, 84)
        pad = 14
        image[y - pad:y + mask.shape[0] + pad, x - pad:x + mask.shape[1] + pad] = (245, 245, 245)
        image[y:y + mask.shape[0], x:x + mask.shape[1]][mask] = (0, 0, 0)


def _letter_mask(letter: str, width: int, height: int) -> np.ndarray:
    patterns = {
        "A": ["0011100", "0110110", "1100011", "1100011", "1111111", "1100011", "1100011", "1100011", "1100011"],
        "B": ["1111100", "1100110", "1100011", "1100110", "1111100", "1100110", "1100011", "1100110", "1111100"],
        "C": ["0011110", "0110011", "1100000", "1100000", "1100000", "1100000", "1100000", "0110011", "0011110"],
        "D": ["1111000", "1101100", "1100110", "1100011", "1100011", "1100011", "1100110", "1101100", "1111000"],
    }
    small = np.array([[ch == "1" for ch in row] for row in patterns[letter]], dtype=bool)
    y_idx = np.linspace(0, small.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, small.shape[1] - 1, width).astype(np.int64)
    return small[y_idx][:, x_idx]


def test_fixed_detector_returns_structured_data() -> None:
    detector = FixedDetectionPipeline()
    frame = _frame()

    letters = detector.detect_zone_letters(frame)
    assert letters["type"] == "zone_letters"
    assert [d["zone"] for d in letters["detections"]] == ["A", "B", "C", "D"]

    gauges = detector.detect_gauges(frame)
    assert gauges["type"] == "gauges"
    assert gauges["detections"][0]["status"] == "low"
    assert "bbox" in gauges["detections"][0]
    assert "timestamp" in gauges["detections"][0]


def test_fixed_detector_empty_when_no_input() -> None:
    detector = FixedDetectionPipeline()
    assert detector.detect_zone_letters(None)["detections"] == []
    assert detector.detect_gauges(None)["detections"] == []


def test_detect_gauges_classifies_low_normal_high() -> None:
    detector = FixedDetectionPipeline(FixedDetectionConfig(
        gauge_low_angle_range=(180.0, 250.0),
        gauge_normal_angle_range=(250.0, 310.0),
        gauge_high_angle_range=(310.0, 30.0),
    ))

    low = detector.detect_gauges(_gauge_frame(210.0))
    normal = detector.detect_gauges(_gauge_frame(280.0))
    high = detector.detect_gauges(_gauge_frame(340.0))

    assert low["detections"][0]["status"] == "low"
    assert normal["detections"][0]["status"] == "normal"
    assert high["detections"][0]["status"] == "high"
    assert low["detections"][0]["confidence"] >= 0.55


def test_detect_gauges_returns_empty_when_no_dial() -> None:
    detector = FixedDetectionPipeline()
    blank = VisionFrame(
        frame_id=2,
        timestamp=200.0,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        width=640,
        height=480,
        source_type="mock",
    )
    result = detector.detect_gauges(blank)
    assert result["type"] == "gauges"
    assert result["detections"] == []


def test_detect_zone_letters_recognizes_synthetic_abcd() -> None:
    detector = FixedDetectionPipeline()
    result = detector.detect_zone_letters(_frame())
    assert result["type"] == "zone_letters"
    assert [item["zone"] for item in result["detections"]] == ["A", "B", "C", "D"]
    for item in result["detections"]:
        assert item["letter"] == item["zone"]
        assert "bbox" in item
        assert item["confidence"] >= 0.55


def test_detect_zone_letters_returns_empty_without_letters() -> None:
    detector = FixedDetectionPipeline()
    blank = VisionFrame(
        frame_id=4,
        timestamp=400.0,
        image=np.full((480, 640, 3), 180, dtype=np.uint8),
        width=640,
        height=480,
        source_type="mock",
    )
    result = detector.detect_zone_letters(blank)
    assert result["type"] == "zone_letters"
    assert result["detections"] == []


def test_detect_zone_letters_generates_missing_templates() -> None:
    template_dir = ROOT / "output" / "test_letter_templates"
    shutil.rmtree(template_dir, ignore_errors=True)
    detector = FixedDetectionPipeline(FixedDetectionConfig(letter_template_dir=str(template_dir)))
    result = detector.detect_zone_letters(_frame())
    assert [item["zone"] for item in result["detections"]] == ["A", "B", "C", "D"]
    for letter in ("A", "B", "C", "D"):
        assert (template_dir / f"{letter}.png").exists()
    shutil.rmtree(template_dir, ignore_errors=True)


def test_detect_zone_letters_saves_debug_images() -> None:
    debug_dir = ROOT / "output" / "test_debug_letters"
    shutil.rmtree(debug_dir, ignore_errors=True)
    detector = FixedDetectionPipeline(FixedDetectionConfig(
        letter_debug_save_roi=True,
        letter_debug_dir=str(debug_dir),
    ))
    result = detector.detect_zone_letters(_frame())
    assert result["detections"]
    saved = list(debug_dir.glob("letter_*"))
    assert saved
    shutil.rmtree(debug_dir, ignore_errors=True)


def test_fuse_inspection_results_generates_speak_keys() -> None:
    letters = [
        {"zone": "A", "confidence": 0.9, "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
        {"zone": "B", "confidence": 0.9, "bbox": {"x1": 110, "y1": 10, "x2": 150, "y2": 50}},
        {"zone": "C", "confidence": 0.9, "bbox": {"x1": 210, "y1": 10, "x2": 250, "y2": 50}},
    ]
    gauges = [
        {"status": "low", "confidence": 0.9, "bbox": {"x1": 12, "y1": 70, "x2": 52, "y2": 110}},
        {"status": "normal", "confidence": 0.9, "bbox": {"x1": 112, "y1": 70, "x2": 152, "y2": 110}},
        {"status": "high", "confidence": 0.9, "bbox": {"x1": 212, "y1": 70, "x2": 252, "y2": 110}},
    ]
    result = fuse_inspection_results(letters, gauges, timestamp=1.0)
    assert [(item["zone"], item["gauge_status"], item["speak_key"], item["abnormal"]) for item in result] == [
        ("A", "low", "A_low", True),
        ("B", "normal", "B_normal", False),
        ("C", "high", "C_high", True),
    ]
    assert result[0]["text"] == "A区域仪表盘显示偏低，状态异常"
    assert result[1]["bbox"]["letter"] == letters[1]["bbox"]
    assert result[1]["bbox"]["gauge"] == gauges[1]["bbox"]


def test_fuse_inspection_results_filters_low_confidence() -> None:
    letters = [
        {"zone": "A", "confidence": 0.4, "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}},
        {"zone": "B", "confidence": 0.9, "bbox": {"x1": 40, "y1": 0, "x2": 60, "y2": 20}},
    ]
    gauges = [
        {"status": "low", "confidence": 0.4, "bbox": {"x1": 0, "y1": 30, "x2": 20, "y2": 50}},
        {"status": "normal", "confidence": 0.4, "bbox": {"x1": 40, "y1": 30, "x2": 60, "y2": 50}},
    ]
    assert fuse_inspection_results(letters, gauges, timestamp=1.0) == []


def test_fuse_inspection_results_empty_inputs() -> None:
    assert fuse_inspection_results([], [], timestamp=1.0) == []


def test_poll_inspection_returns_fused_results_and_debug() -> None:
    debug_dir = ROOT / "output" / "test_debug_inspection"
    shutil.rmtree(debug_dir, ignore_errors=True)
    detector = FixedDetectionPipeline(FixedDetectionConfig(
        inspection_debug_save=True,
        inspection_debug_dir=str(debug_dir),
        inspection_max_match_distance=10.0,
    ))
    result = detector.poll_inspection(_frame())
    assert result["type"] == "inspection_results"
    assert result["results"]
    assert result["results"][0]["speak_key"].endswith(("_low", "_normal", "_high"))
    assert list(debug_dir.glob("inspection_debug_*"))
    shutil.rmtree(debug_dir, ignore_errors=True)
    detector = FixedDetectionPipeline(FixedDetectionConfig(
        gauge_debug_save_roi=True,
        gauge_debug_dir=str(debug_dir),
    ))
    result = detector.detect_gauges(_gauge_frame(210.0, frame_id=3))
    assert result["detections"]
    saved = list(debug_dir.glob("gauge_*"))
    assert saved
    shutil.rmtree(debug_dir, ignore_errors=True)


def test_fixed_detector_empty_config() -> None:
    detector = FixedDetectionPipeline(FixedDetectionConfig(empty_results=True))
    frame = _frame()
    assert detector.detect_zone_letters(frame)["detections"] == []
    assert detector.detect_gauges(frame)["detections"] == []


def test_remote_gateway_parses_fixed_detector_json() -> None:
    detector = FixedDetectionPipeline()
    frame = _frame()
    gateway = RemotePerceptionGateway(RemotePerceptionConfig(host="127.0.0.1", port=1))

    letters = gateway._parse_zone_letters(detector.detect_zone_letters(frame))  # noqa: SLF001
    assert [item.zone.value for item in letters] == ["A", "B", "C", "D"]
    assert letters[0].bbox is not None

    gauges = gateway._parse_gauges(detector.detect_gauges(frame))  # noqa: SLF001
    assert gauges[0].zone.value == "A"
    assert gauges[0].status.value == "low"

    inspection = gateway._parse_inspection_results(detector.poll_inspection(frame))  # noqa: SLF001
    assert inspection
    assert inspection[0].zone.value in {"A", "B", "C", "D"}
