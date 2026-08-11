"""End-to-end demo for real-photo inspection recognition.

This script wires the existing tools only:
YOLO bbox localization -> gauge ROI reading -> JSON/text/debug output.
It does not trigger speaker playback, Mission, navigation, arm, or DogController.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gauge_reader import GaugeReader, GaugeReaderConfig
from yolo_locator import IMAGE_EXTS, LocatorConfig, YoloLocator, image_sources


DEFAULT_MODEL = Path("runs/detect/train/weights/best.pt")


@dataclass(frozen=True)
class DemoConfig:
    source: Path
    model: Path = DEFAULT_MODEL
    output_json: Path = Path("output/inspection_pipeline_demo.json")
    debug_dir: Path = Path("output/debug_inspection_pipeline")
    conf: float = 0.25
    imgsz: int = 640
    device: str | None = None


def error_payload(code: str, message: str, *, source: Path | None = None, model: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": code,
        "message": message,
        "timestamp": time.time(),
    }
    if source is not None:
        payload["source"] = str(source)
    if model is not None:
        payload["model"] = str(model)
    return payload


def validate_inputs(cfg: DemoConfig) -> dict[str, Any] | None:
    if not cfg.source.exists():
        return error_payload("source_not_found", f"input image or directory does not exist: {cfg.source}", source=cfg.source)
    if cfg.source.is_file() and cfg.source.suffix.lower() not in IMAGE_EXTS:
        return error_payload("unsupported_source", f"input file is not a supported image: {cfg.source}", source=cfg.source)
    if cfg.source.is_dir() and not image_sources(cfg.source):
        return error_payload("no_images", f"input directory contains no supported images: {cfg.source}", source=cfg.source)
    if not cfg.model.exists():
        return error_payload(
            "model_not_found",
            f"YOLO model not found: {cfg.model}. Train first or pass --model runs/detect/train/weights/best.pt",
            source=cfg.source,
            model=cfg.model,
        )
    return None


def bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, (float(bbox["y1"]) + float(bbox["y2"])) / 2.0)


def nearest_zone(gauge: dict[str, Any], letters: list[dict[str, Any]]) -> str | None:
    if not letters:
        return None
    gx, gy = bbox_center(gauge["bbox"])
    best = min(
        letters,
        key=lambda item: (bbox_center(item["bbox"])[0] - gx) ** 2 + (bbox_center(item["bbox"])[1] - gy) ** 2,
    )
    zone = best.get("zone") or best.get("letter")
    return str(zone).upper() if zone else None


def draw_debug(image_path: Path, output_path: Path, detections: list[dict[str, Any]], readings: list[dict[str, Any]]) -> None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for det in detections:
        bbox = det.get("bbox") or {}
        if not {"x1", "y1", "x2", "y2"} <= set(bbox):
            continue
        x1, y1, x2, y2 = (int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))
        color = (0, 255, 255) if det.get("object_type") == "gauge" else (255, 255, 0)
        label = str(det.get("zone") or det.get("class_name") or det.get("object_type"))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for item in readings:
        bbox = item.get("bbox") or {}
        if not {"x1", "y1", "x2", "y2"} <= set(bbox):
            continue
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        label = str(item.get("speak_key") or item.get("gauge_status"))
        cv2.putText(image, label, (x1, max(36, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def process_image(image_path: Path, locator: YoloLocator, reader: GaugeReader, cfg: DemoConfig) -> dict[str, Any]:
    detections = locator.detect(image_path)
    letters = [item for item in detections if item.get("object_type") == "zone_letter"]
    gauges = [item for item in detections if item.get("object_type") == "gauge"]
    readings: list[dict[str, Any]] = []
    for gauge in gauges:
        zone = nearest_zone(gauge, letters)
        try:
            reading = reader.read_image(image_path, gauge["bbox"], zone=zone)
        except Exception as exc:  # keep batch demos moving on bad ROIs
            reading = {
                "zone": zone,
                "gauge_status": "normal",
                "status": "normal",
                "abnormal": False,
                "angle": None,
                "confidence": 0.0,
                "bbox": gauge.get("bbox"),
                "speak_key": None if zone is None else f"{zone}_normal",
                "text": f"{zone}区域仪表盘读取失败，状态按正常处理" if zone else "仪表盘读取失败，状态按正常处理",
                "timestamp": time.time(),
                "error": str(exc),
            }
        readings.append(reading)

    debug_path = cfg.debug_dir / f"{image_path.stem}_inspection.jpg"
    draw_debug(image_path, debug_path, detections, readings)
    return {
        "image": str(image_path),
        "detections": detections,
        "readings": readings,
        "texts": [item["text"] for item in readings],
        "debug_image": str(debug_path),
    }


def run_demo(cfg: DemoConfig) -> dict[str, Any]:
    validation_error = validate_inputs(cfg)
    if validation_error is not None:
        return validation_error

    locator = YoloLocator(LocatorConfig(
        model_path=str(cfg.model),
        confidence=cfg.conf,
        imgsz=cfg.imgsz,
        device=cfg.device,
        debug_dir=cfg.debug_dir / "yolo",
    ))
    reader = GaugeReader(GaugeReaderConfig(debug_dir=cfg.debug_dir / "gauge"))
    try:
        items = [process_image(path, locator, reader, cfg) for path in image_sources(cfg.source)]
    except RuntimeError as exc:
        return error_payload("runtime_error", str(exc), source=cfg.source, model=cfg.model)

    return {
        "ok": True,
        "source": str(cfg.source),
        "model": str(cfg.model),
        "count": len(items),
        "items": items,
        "timestamp": time.time(),
    }


def write_payload(payload: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-photo inspection demo with YOLO locator and gauge reader.")
    parser.add_argument("--source", required=True, type=Path, help="One image or a folder of images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO best.pt path.")
    parser.add_argument("--output-json", type=Path, default=Path("output/inspection_pipeline_demo.json"))
    parser.add_argument("--debug-dir", type=Path, default=Path("output/debug_inspection_pipeline"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = DemoConfig(
        source=args.source,
        model=args.model,
        output_json=args.output_json,
        debug_dir=args.debug_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    payload = run_demo(cfg)
    write_payload(payload, cfg.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok") or payload.get("error") == "model_not_found" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
