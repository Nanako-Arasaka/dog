"""Read an analog gauge status from a YOLO-provided gauge ROI."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _patch_cv2_unicode_io() -> None:
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        return
    if getattr(cv2, "_dog_repo_unicode_io_patch", False):
        return

    original_imread = cv2.imread
    original_imwrite = cv2.imwrite

    def imread(path: str, flags: int = cv2.IMREAD_COLOR):
        image = original_imread(path, flags)
        if image is not None:
            return image
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, flags)
        except OSError:
            return None

    def imwrite(path: str, image: Any, params: Any = None) -> bool:
        ok = original_imwrite(path, image, params or [])
        if ok:
            return True
        suffix = Path(path).suffix or ".jpg"
        success, encoded = cv2.imencode(suffix, image, params or [])
        if not success:
            return False
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            encoded.tofile(str(path))
            return True
        except OSError:
            return False

    cv2.imread = imread
    cv2.imwrite = imwrite
    cv2._dog_repo_unicode_io_patch = True


_patch_cv2_unicode_io()


@dataclass(frozen=True)
class GaugeReaderConfig:
    low_range: tuple[float, float] = (180.0, 250.0)
    normal_range: tuple[float, float] = (250.0, 310.0)
    high_range: tuple[float, float] = (310.0, 30.0)
    min_line_ratio: float = 0.25
    debug_dir: Path | None = None


def try_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for gauge_reader.py") from exc
    _patch_cv2_unicode_io()
    return cv2


def parse_bbox(raw: str) -> dict[str, int]:
    if raw.strip().startswith("{"):
        data = json.loads(raw)
        return {key: int(data[key]) for key in ("x1", "y1", "x2", "y2")}
    parts = [int(float(x)) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be x1,y1,x2,y2 or JSON with x1/y1/x2/y2")
    return {"x1": parts[0], "y1": parts[1], "x2": parts[2], "y2": parts[3]}


def clamp_bbox(bbox: dict[str, int], width: int, height: int) -> dict[str, int]:
    x1 = max(0, min(width - 1, int(bbox["x1"])))
    y1 = max(0, min(height - 1, int(bbox["y1"])))
    x2 = max(x1 + 1, min(width, int(bbox["x2"])))
    y2 = max(y1 + 1, min(height, int(bbox["y2"])))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def angle_from_center(cx: float, cy: float, tip_x: float, tip_y: float) -> float:
    dx = tip_x - cx
    dy = cy - tip_y
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def angle_in_range(angle: float, angle_range: tuple[float, float]) -> bool:
    start, end = angle_range
    angle = angle % 360.0
    start = start % 360.0
    end = end % 360.0
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def status_from_angle(angle: float, cfg: GaugeReaderConfig) -> str:
    if angle_in_range(angle, cfg.low_range):
        return "low"
    if angle_in_range(angle, cfg.normal_range):
        return "normal"
    if angle_in_range(angle, cfg.high_range):
        return "high"
    return "normal"


def chinese_text(zone: str | None, status: str) -> str:
    display = {"low": "偏低", "normal": "正常", "high": "偏高"}[status]
    health = "异常" if status in {"low", "high"} else "正常"
    if zone:
        return f"{zone}区域仪表盘显示{display}，状态{health}"
    return f"仪表盘显示{display}，状态{health}"


class GaugeReader:
    def __init__(self, cfg: GaugeReaderConfig | None = None) -> None:
        self.cfg = cfg or GaugeReaderConfig()

    def read_image(self, image_path: Path, bbox: dict[str, int], zone: str | None = None) -> dict[str, Any]:
        cv2 = try_cv2()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        return self.read(image, bbox, zone=zone, image_id=str(image_path))

    def read(self, image: Any, bbox: dict[str, int], zone: str | None = None, image_id: str | None = None) -> dict[str, Any]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return self._empty_result(bbox, zone, image_id, "empty image")
        bbox = clamp_bbox(bbox, w, h)
        roi = image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        if roi.size == 0:
            return self._empty_result(bbox, zone, image_id, "empty gauge ROI")

        cv2 = try_cv2()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 45, 140)
        rh, rw = gray.shape[:2]
        cx, cy = rw / 2.0, rh / 2.0
        radius = min(rw, rh) / 2.0
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=math.pi / 180.0,
            threshold=max(18, int(radius * 0.18)),
            minLineLength=max(12, int(radius * self.cfg.min_line_ratio)),
            maxLineGap=max(6, int(radius * 0.08)),
        )
        best = self._choose_pointer(lines, cx, cy, radius)
        method = "hough"
        if best is None:
            best = self._choose_dark_tip(enhanced, cx, cy, radius)
            method = "dark_tip"
        if best is None:
            status = "normal"
            angle = None
            confidence = 0.0
            line = None
        else:
            tip_x, tip_y, score = best
            angle = angle_from_center(cx, cy, tip_x, tip_y)
            status = status_from_angle(angle, self.cfg)
            confidence = round(max(0.0, min(0.99, score)), 4)
            line = (int(cx), int(cy), int(tip_x), int(tip_y))

        result = {
            "zone": zone,
            "gauge_status": status,
            "status": status,
            "abnormal": status in {"low", "high"},
            "angle": None if angle is None else round(float(angle), 2),
            "confidence": confidence,
            "bbox": bbox,
            "speak_key": None if zone is None else f"{zone}_{status}",
            "text": chinese_text(zone, status),
            "timestamp": time.time(),
            "method": method,
        }
        if image_id is not None:
            result["image"] = image_id
        if self.cfg.debug_dir is not None:
            self._save_debug(cv2, image, roi, bbox, result, line)
        return result

    def _empty_result(
        self,
        bbox: dict[str, int],
        zone: str | None,
        image_id: str | None,
        reason: str,
    ) -> dict[str, Any]:
        result = {
            "zone": zone,
            "gauge_status": "normal",
            "status": "normal",
            "abnormal": False,
            "angle": None,
            "confidence": 0.0,
            "bbox": bbox,
            "speak_key": None if zone is None else f"{zone}_normal",
            "text": chinese_text(zone, "normal"),
            "timestamp": time.time(),
            "method": "empty_roi",
            "warning": reason,
        }
        if image_id is not None:
            result["image"] = image_id
        return result

    def _choose_pointer(self, lines: Any, cx: float, cy: float, radius: float) -> tuple[float, float, float] | None:
        if lines is None:
            return None
        best: tuple[float, float, float] | None = None
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            near, far = min(d1, d2), max(d1, d2)
            if near > radius * 0.38 or far < radius * 0.28:
                continue
            tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
            score = min(0.99, max(0.1, (far / max(radius, 1.0)) - (near / max(radius, 1.0)) * 0.35))
            if best is None or score > best[2]:
                best = (tip_x, tip_y, score)
        return best

    def _choose_dark_tip(self, gray: Any, cx: float, cy: float, radius: float) -> tuple[float, float, float] | None:
        import numpy as np

        yy, xx = np.indices(gray.shape)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = (rr < radius * 0.9) & (rr > radius * 0.16) & (gray < np.percentile(gray, 18))
        ys, xs = np.where(mask)
        if xs.size < 8:
            return None
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        idx = int(np.argmax(dist))
        score = min(0.75, max(0.1, float(dist[idx]) / max(radius, 1.0)))
        return float(xs[idx]), float(ys[idx]), score

    def _save_debug(self, cv2: Any, image: Any, roi: Any, bbox: dict[str, int], result: dict[str, Any], line: tuple[int, int, int, int] | None) -> None:
        assert self.cfg.debug_dir is not None
        self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(result["timestamp"] * 1000)
        cv2.imwrite(str(self.cfg.debug_dir / f"gauge_roi_{stamp}.jpg"), roi)
        annotated = image.copy()
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if line is not None:
            cv2.line(
                annotated,
                (x1 + line[0], y1 + line[1]),
                (x1 + line[2], y1 + line[3]),
                (0, 0, 255),
                2,
            )
        label = f"{result['status']} {result['angle']}"
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(self.cfg.debug_dir / f"gauge_debug_{stamp}.jpg"), annotated)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read low/normal/high status from a gauge ROI.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--bbox", required=True, help="x1,y1,x2,y2 or JSON bbox.")
    parser.add_argument("--zone", default=None, help="Optional zone letter A/B/C/D.")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reader = GaugeReader(GaugeReaderConfig(debug_dir=args.debug_dir))
    result = reader.read_image(args.image, parse_bbox(args.bbox), zone=args.zone)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
