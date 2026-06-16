from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from dataset_builder import BuildConfig, build_dataset  # noqa: E402
from gauge_reader import GaugeReader, GaugeReaderConfig  # noqa: E402
from inspection_pipeline_demo import DemoConfig, run_demo  # noqa: E402
from label_check import CheckConfig, check_labels  # noqa: E402
from yolo_locator import normalize_class  # noqa: E402


def _case_dir(name: str) -> Path:
    path = ROOT / "output" / "test_inspection_tools" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def test_dataset_builder_creates_yolo_layout_and_stats() -> None:
    tmp_path = _case_dir("dataset_builder")
    raw = tmp_path / "raw"
    out = tmp_path / "dataset"
    for idx in range(5):
        image = raw / f"frame_{idx}.jpg"
        _write_fake_image(image)
        image.with_suffix(".txt").write_text("4 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    summary = build_dataset(BuildConfig(raw_dir=raw, output_dir=out, workers=2, seed=1))

    assert (out / "dataset.yaml").exists()
    assert (out / "stats.csv").exists()
    assert summary["total_images"] == 5
    assert sum(len(list((out / "images" / split).glob("*.jpg"))) for split in ("train", "val", "test")) == 5
    assert sum(len(list((out / "labels" / split).glob("*.txt"))) for split in ("train", "val", "test")) == 5


def test_label_check_reports_out_of_bounds_bbox() -> None:
    tmp_path = _case_dir("label_check")
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_fake_image(images / "bad.jpg")
    labels.mkdir(parents=True, exist_ok=True)
    (labels / "bad.txt").write_text("0 0.95 0.50 0.20 0.20\n", encoding="utf-8")

    result = check_labels(CheckConfig(images_dir=images, labels_dir=labels, class_count=5, workers=1))

    assert result["ok"] is False
    assert any("outside image" in issue["message"] for issue in result["issues"])


def test_yolo_locator_normalizes_competition_classes() -> None:
    assert normalize_class("zone_A") == ("zone_letter", "A")
    assert normalize_class("letter_b") == ("zone_letter", "B")
    assert normalize_class("gauge") == ("gauge", None)
    assert normalize_class("other") == ("other", None)


def _cv2_available() -> bool:
    try:
        import cv2  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _gauge_image(angle_deg: float) -> np.ndarray:
    import cv2  # type: ignore

    image = np.zeros((240, 240, 3), dtype=np.uint8)
    cx, cy, radius = 120, 120, 88
    cv2.circle(image, (cx, cy), radius, (235, 235, 235), -1)
    cv2.circle(image, (cx, cy), radius, (30, 30, 30), 2)
    tip_x = int(cx + np.cos(np.deg2rad(angle_deg)) * radius * 0.78)
    tip_y = int(cy - np.sin(np.deg2rad(angle_deg)) * radius * 0.78)
    cv2.line(image, (cx, cy), (tip_x, tip_y), (0, 0, 0), 3)
    return image


@pytest.mark.skipif(not _cv2_available(), reason="OpenCV is optional in the local test environment")
def test_gauge_reader_reads_status_and_text() -> None:
    import cv2  # type: ignore

    tmp_path = _case_dir("gauge_reader")
    image_path = tmp_path / "gauge.jpg"
    cv2.imwrite(str(image_path), _gauge_image(210.0))
    reader = GaugeReader(GaugeReaderConfig(debug_dir=tmp_path / "debug"))

    result = reader.read_image(image_path, {"x1": 0, "y1": 0, "x2": 240, "y2": 240}, zone="A")

    assert result["gauge_status"] == "low"
    assert result["abnormal"] is True
    assert result["speak_key"] == "A_low"
    assert result["text"] == "A区域仪表盘显示偏低，状态异常"
    assert list((tmp_path / "debug").glob("gauge_debug_*.jpg"))


def test_gauge_reader_parseable_output_shape() -> None:
    sample = {
        "zone": "B",
        "gauge_status": "normal",
        "abnormal": False,
        "speak_key": "B_normal",
        "text": "B区域仪表盘显示正常，状态正常",
        "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
        "timestamp": 1.0,
    }
    assert json.loads(json.dumps(sample, ensure_ascii=False))["speak_key"] == "B_normal"


def test_pipeline_demo_reports_missing_model_without_crashing() -> None:
    tmp_path = _case_dir("pipeline_missing_model")
    image = tmp_path / "sample.jpg"
    _write_fake_image(image)

    result = run_demo(DemoConfig(source=image, model=tmp_path / "missing_best.pt"))

    assert result["ok"] is False
    assert result["error"] == "model_not_found"
    assert "YOLO model not found" in result["message"]


def test_pipeline_demo_reports_missing_source() -> None:
    tmp_path = _case_dir("pipeline_missing_source")

    result = run_demo(DemoConfig(source=tmp_path / "missing.jpg", model=tmp_path / "best.pt"))

    assert result["ok"] is False
    assert result["error"] == "source_not_found"
    assert "does not exist" in result["message"]


def test_gauge_reader_empty_roi_returns_warning_without_cv2() -> None:
    reader = GaugeReader()

    result = reader.read(np.zeros((0, 0, 3), dtype=np.uint8), {"x1": 0, "y1": 0, "x2": 0, "y2": 0}, zone="A")

    assert result["gauge_status"] == "normal"
    assert result["confidence"] == 0.0
    assert result["warning"] == "empty image"
    assert result["text"] == "A区域仪表盘显示正常，状态正常"
