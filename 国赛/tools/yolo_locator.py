"""YOLO locator for zone letters and gauge bounding boxes.

This module intentionally keeps the output contract close to fixed_detector:
letter detections use `object_type=zone_letter`; gauge detections use
`object_type=gauge`. The model dependency is optional until training starts.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LocatorConfig:
    model_path: str
    confidence: float = 0.25
    imgsz: int = 640
    device: str | None = None
    debug_dir: Path | None = None


def try_cv2() -> Any | None:
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None


def image_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def normalize_class(name: str) -> tuple[str, str | None]:
    raw = name.strip()
    lower = raw.lower()
    if lower in {"a", "zone_a", "letter_a"}:
        return "zone_letter", "A"
    if lower in {"b", "zone_b", "letter_b"}:
        return "zone_letter", "B"
    if lower in {"c", "zone_c", "letter_c"}:
        return "zone_letter", "C"
    if lower in {"d", "zone_d", "letter_d"}:
        return "zone_letter", "D"
    if lower in {"gauge", "meter", "dashboard", "dial"}:
        return "gauge", None
    return raw, None


class YoloLocator:
    """Thin ultralytics wrapper with a stable JSON-friendly output."""

    def __init__(self, cfg: LocatorConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for yolo_locator.py; install ultralytics on Jetson/Ubuntu") from exc
        self._model = YOLO(self.cfg.model_path)

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        if self._model is None:
            self.load()
        assert self._model is not None
        kwargs: dict[str, Any] = {
            "source": str(image_path),
            "conf": self.cfg.confidence,
            "imgsz": self.cfg.imgsz,
            "verbose": False,
        }
        if self.cfg.device:
            kwargs["device"] = self.cfg.device
        results = self._model.predict(**kwargs)
        if not results:
            return []
        detections = self._convert_result(results[0], image_path)
        if self.cfg.debug_dir is not None:
            self.save_debug(image_path, detections)
        return detections

    def _convert_result(self, result: Any, image_path: Path) -> list[dict[str, Any]]:
        names = getattr(result, "names", {}) or getattr(self._model, "names", {})
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
            conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(round(float(v))) for v in coords]
            class_name = str(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id])
            object_type, zone = normalize_class(class_name)
            item: dict[str, Any] = {
                "object_type": object_type,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "image": str(image_path),
                "timestamp": time.time(),
            }
            if zone is not None:
                item["zone"] = zone
                item["letter"] = zone
            detections.append(item)
        return detections

    def save_debug(self, image_path: Path, detections: list[dict[str, Any]]) -> None:
        cv2 = try_cv2()
        if cv2 is None or self.cfg.debug_dir is None:
            return
        image = cv2.imread(str(image_path))
        if image is None:
            return
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = (0, 255, 255) if det.get("object_type") == "gauge" else (255, 255, 0)
            label = f"{det.get('zone', det.get('class_name'))} {det.get('confidence', 0):.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out_path = self.cfg.debug_dir / image_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)


class ThreadLocalLocator:
    def __init__(self, cfg: LocatorConfig) -> None:
        self.cfg = cfg
        self.local = threading.local()

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        locator = getattr(self.local, "locator", None)
        if locator is None:
            locator = YoloLocator(self.cfg)
            locator.load()
            self.local.locator = locator
        return locator.detect(image_path)


def locate_images(cfg: LocatorConfig, source: Path, workers: int = 1) -> list[dict[str, Any]]:
    sources = image_sources(source)
    if workers <= 1:
        locator = YoloLocator(cfg)
        return [{"image": str(path), "detections": locator.detect(path)} for path in sources]
    locator = ThreadLocalLocator(cfg)
    output: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(locator.detect, path): path for path in sources}
        for future in as_completed(future_map):
            path = future_map[future]
            output.append({"image": str(path), "detections": future.result()})
    return sorted(output, key=lambda item: item["image"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO localization for zone letters and gauges.")
    parser.add_argument("--model", required=True, help="YOLO .pt/.onnx model path, e.g. runs/detect/train/weights/best.pt")
    parser.add_argument("--source", required=True, type=Path, help="Image file or image directory.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = LocatorConfig(
        model_path=args.model,
        confidence=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        debug_dir=args.debug_dir,
    )
    result = locate_images(cfg, args.source, workers=args.workers)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
