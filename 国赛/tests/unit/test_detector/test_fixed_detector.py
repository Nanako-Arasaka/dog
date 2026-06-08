from __future__ import annotations

import sys
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_input import VisionFrame  # noqa: E402
from perception.detector.fixed_detector import FixedDetectionConfig, FixedDetectionPipeline  # noqa: E402
from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway  # noqa: E402


def _frame() -> VisionFrame:
    image = _gauge_image(210.0)
    image[40:100, 430:590, 2] = 255
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

    obstacles = detector.detect_obstacles(frame)
    assert obstacles["type"] == "obstacles"
    assert obstacles["detections"][0]["object_type"] == "cone"
    assert "bbox" in obstacles["detections"][0]
    assert "pose" in obstacles["detections"][0]
    assert obstacles["detections"][0]["confidence"] > 0

    letters = detector.detect_zone_letters(frame)
    assert letters["type"] == "zone_letters"
    assert [d["zone"] for d in letters["detections"]] == ["A", "B", "C", "D"]

    gauges = detector.detect_gauges(frame)
    assert gauges["type"] == "gauges"
    assert gauges["detections"][0]["status"] == "low"
    assert "bbox" in gauges["detections"][0]
    assert "timestamp" in gauges["detections"][0]

    strips = detector.detect_red_strips(frame)
    assert strips["type"] == "red_strips"
    assert strips["detections"][0]["object_type"] == "red_strip"

    pose = detector.estimate_target_pose(frame)
    assert pose["type"] == "target_pose"
    assert pose["pose"]["z"] > 0
    assert pose["confidence"] > 0


def test_fixed_detector_empty_when_no_input() -> None:
    detector = FixedDetectionPipeline()
    assert detector.detect_obstacles(None)["detections"] == []
    assert detector.detect_zone_letters(None)["detections"] == []
    assert detector.detect_gauges(None)["detections"] == []
    assert detector.detect_red_strips(None)["detections"] == []
    assert detector.estimate_target_pose(None)["pose"] is None


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


def test_detect_gauges_saves_debug_images() -> None:
    debug_dir = ROOT / "output" / "test_debug_gauge"
    shutil.rmtree(debug_dir, ignore_errors=True)


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
    assert detector.detect_obstacles(frame)["detections"] == []
    assert detector.detect_zone_letters(frame)["detections"] == []
    assert detector.detect_gauges(frame)["detections"] == []
    assert detector.detect_red_strips(frame)["detections"] == []
    assert detector.estimate_target_pose(frame)["pose"] is None


def test_remote_gateway_parses_fixed_detector_json() -> None:
    detector = FixedDetectionPipeline()
    frame = _frame()
    gateway = RemotePerceptionGateway(RemotePerceptionConfig(host="127.0.0.1", port=1))

    obstacles = gateway._parse_obstacles(detector.detect_obstacles(frame))  # noqa: SLF001
    assert len(obstacles) == 1
    assert obstacles[0].confidence > 0

    letters = gateway._parse_zone_letters(detector.detect_zone_letters(frame))  # noqa: SLF001
    assert [item.zone.value for item in letters] == ["A", "B", "C", "D"]
    assert letters[0].bbox is not None

    gauges = gateway._parse_gauges(detector.detect_gauges(frame))  # noqa: SLF001
    assert gauges[0].zone.value == "A"
    assert gauges[0].status.value == "low"

    strips = gateway._parse_red_strips(detector.detect_red_strips(frame))  # noqa: SLF001
    assert len(strips) == 1
    assert strips[0].center_3d[2] > 0

    pose = gateway._parse_target_pose(detector.estimate_target_pose(frame))  # noqa: SLF001
    assert pose is not None
    assert pose.confidence > 0
