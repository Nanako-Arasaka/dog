"""Fixed first-stage detectors for the compute-board vision server.

These detectors are intentionally simple. They provide stable structured
outputs for the remote-perception contract before YOLO/OCR/meter models are
integrated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from camera_input import VisionFrame
except ImportError:  # pragma: no cover - only used when imported outside server root
    VisionFrame = Any  # type: ignore


@dataclass(frozen=True)
class FixedDetectionConfig:
    empty_results: bool = False
    gauge_low_angle_range: tuple[float, float] = (180.0, 250.0)
    gauge_normal_angle_range: tuple[float, float] = (250.0, 310.0)
    gauge_high_angle_range: tuple[float, float] = (310.0, 30.0)
    gauge_min_confidence: float = 0.55
    gauge_debug_save_roi: bool = False
    gauge_debug_dir: str = "output/debug_gauge"


class FixedDetectionPipeline:
    """Protocol-level fixed detector used by vision_server.py."""

    def __init__(self, cfg: FixedDetectionConfig | None = None) -> None:
        self._cfg = cfg or FixedDetectionConfig()

    def handle(self, request: dict[str, Any], frame: VisionFrame | None) -> dict[str, Any]:
        req = str(request.get("req", ""))
        if req == "detect_obstacles":
            return self.detect_obstacles(frame)
        if req == "detect_zone_letters":
            return self.detect_zone_letters(frame)
        if req == "detect_gauges":
            return self.detect_gauges(frame)
        if req == "detect_red_strips":
            return self.detect_red_strips(frame)
        if req == "estimate_target_pose":
            return self.estimate_target_pose(frame, str(request.get("target", "strip")))
        return {"type": "error", "message": f"unknown request: {req}", "timestamp": _timestamp(frame)}

    def detect_obstacles(self, frame: VisionFrame | None) -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("detect_obstacles")
        w, h = frame.width, frame.height
        return {
            "type": "obstacles",
            "detections": [
                {
                    "object_type": "cone",
                    "bbox": _bbox(w, h, 0.42, 0.42, 0.54, 0.82),
                    "center_3d": [0.15, 0.0, 1.2],
                    "pose": {"x": 0.15, "y": 0.0, "z": 1.2},
                    "confidence": 0.92,
                    "timestamp": frame.timestamp,
                }
            ],
            "timestamp": frame.timestamp,
        }

    def detect_zone_letters(self, frame: VisionFrame | None) -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("detect_zone_letters")
        w, h = frame.width, frame.height
        zones = [
            ("A", _bbox(w, h, 0.12, 0.16, 0.25, 0.30), 0.95),
            ("B", _bbox(w, h, 0.42, 0.16, 0.55, 0.30), 0.94),
            ("C", _bbox(w, h, 0.12, 0.54, 0.25, 0.68), 0.93),
            ("D", _bbox(w, h, 0.42, 0.54, 0.55, 0.68), 0.96),
        ]
        return {
            "type": "zone_letters",
            "detections": [
                {
                    "zone": zone,
                    "object_type": "zone_letter",
                    "bbox": bbox,
                    "confidence": conf,
                    "timestamp": frame.timestamp,
                }
                for zone, bbox, conf in zones
            ],
            "timestamp": frame.timestamp,
        }

    def detect_gauges(self, frame: VisionFrame | None) -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("detect_gauges")
        detected = _detect_gauge_from_frame(frame, self._cfg)
        if detected is None:
            return {
                "type": "gauges",
                "detections": [],
                "timestamp": frame.timestamp,
            }

        status, angle, confidence, bbox = detected
        if confidence < self._cfg.gauge_min_confidence:
            return {
                "type": "gauges",
                "detections": [],
                "timestamp": frame.timestamp,
            }

        return {
            "type": "gauges",
            "detections": [
                {
                    "zone": "A",
                    "object_type": "gauge",
                    "status": status,
                    "raw_value": round(angle, 2),
                    "angle": round(angle, 2),
                    "bbox": bbox,
                    "confidence": conf,
                    "timestamp": frame.timestamp,
                }
                for conf in [confidence]
            ],
            "timestamp": frame.timestamp,
        }

    def detect_red_strips(self, frame: VisionFrame | None) -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("detect_red_strips")

        strip = _detect_red_region(frame)
        if strip is None:
            strip = {
                "bbox": _bbox(frame.width, frame.height, 0.36, 0.42, 0.66, 0.58),
                "center_3d": [0.05, 0.0, 0.28],
                "confidence": 0.82,
            }

        return {
            "type": "red_strips",
            "detections": [
                {
                    "object_type": "red_strip",
                    "bbox": strip["bbox"],
                    "center_3d": strip["center_3d"],
                    "pose": {
                        "x": strip["center_3d"][0],
                        "y": strip["center_3d"][1],
                        "z": strip["center_3d"][2],
                    },
                    "confidence": strip["confidence"],
                    "timestamp": frame.timestamp,
                }
            ],
            "timestamp": frame.timestamp,
        }

    def estimate_target_pose(self, frame: VisionFrame | None, target: str = "strip") -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("estimate_target_pose")

        strip_resp = self.detect_red_strips(frame)
        detections = strip_resp.get("detections", [])
        if target == "strip" and detections:
            center = detections[0].get("center_3d", [0.05, 0.0, 0.28])
            conf = float(detections[0].get("confidence", 0.0))
        else:
            center = [0.0, 0.0, 0.1]
            conf = 0.6
        return {
            "type": "target_pose",
            "pose": {
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "object_type": target,
            "confidence": conf,
            "timestamp": frame.timestamp,
        }


def empty_response(req: str) -> dict[str, Any]:
    ts = time.time()
    if req == "detect_obstacles":
        return {"type": "obstacles", "detections": [], "timestamp": ts}
    if req == "detect_zone_letters":
        return {"type": "zone_letters", "detections": [], "timestamp": ts}
    if req == "detect_gauges":
        return {"type": "gauges", "detections": [], "timestamp": ts}
    if req == "detect_red_strips":
        return {"type": "red_strips", "detections": [], "timestamp": ts}
    if req == "estimate_target_pose":
        return {"type": "target_pose", "pose": None, "confidence": 0.0, "timestamp": ts}
    return {"type": "error", "message": f"unknown request: {req}", "timestamp": ts}


def _timestamp(frame: VisionFrame | None) -> float:
    return float(frame.timestamp) if frame is not None else time.time()


def _bbox(width: int, height: int, x1: float, y1: float, x2: float, y2: float) -> dict[str, int]:
    return {
        "x1": int(width * x1),
        "y1": int(height * y1),
        "x2": int(width * x2),
        "y2": int(height * y2),
    }


def _detect_red_region(frame: VisionFrame) -> dict[str, Any] | None:
    image = frame.image
    if image.size == 0 or image.ndim != 3 or image.shape[2] < 3:
        return None

    # BGR heuristic: red channel high, blue/green lower. This is intentionally
    # simple and deterministic until a learned red-strip detector is added.
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    mask = (red > 120) & (red > green + 40) & (red > blue + 40)
    ys, xs = np.where(mask)
    if xs.size < 25:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = ((x1 + x2) / 2.0 - frame.width / 2.0) / max(frame.width, 1)
    cy = ((y1 + y2) / 2.0 - frame.height / 2.0) / max(frame.height, 1)
    area_ratio = float(xs.size) / float(max(frame.width * frame.height, 1))
    confidence = min(0.98, max(0.55, 0.55 + area_ratio * 4.0))
    return {
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "center_3d": [round(cx, 4), round(cy, 4), 0.28],
        "confidence": round(confidence, 4),
    }


def _detect_gauge_from_frame(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
) -> tuple[str, float, float, dict[str, int]] | None:
    cv2 = _try_cv2()
    if cv2 is not None:
        result = _detect_gauge_with_cv2(frame, cfg, cv2)
        if result is not None:
            return result
    return _detect_gauge_with_numpy(frame, cfg)


def _detect_gauge_with_cv2(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
    cv2: Any,
) -> tuple[str, float, float, dict[str, int]] | None:
    image = frame.image
    if image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min(frame.width, frame.height) // 3, 20),
        param1=80,
        param2=24,
        minRadius=max(min(frame.width, frame.height) // 10, 20),
        maxRadius=max(min(frame.width, frame.height) // 2, 30),
    )
    if circles is not None and len(circles) > 0:
        c = np.round(circles[0][0]).astype(int)
        cx, cy, radius = int(c[0]), int(c[1]), int(c[2])
    else:
        located = _locate_gauge_by_bright_region(image)
        if located is None:
            return None
        cx, cy, radius = located

    bbox = _circle_bbox(cx, cy, radius, frame.width, frame.height)
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    roi_gray = gray[y1:y2, x1:x2]
    if roi_gray.size == 0:
        return None
    edges = cv2.Canny(roi_gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=25,
        minLineLength=max(radius // 3, 15),
        maxLineGap=10,
    )

    best: tuple[int, int, int, int, float] | None = None
    local_cx, local_cy = cx - x1, cy - y1
    if lines is not None:
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(v) for v in line]
            d1 = ((lx1 - local_cx) ** 2 + (ly1 - local_cy) ** 2) ** 0.5
            d2 = ((lx2 - local_cx) ** 2 + (ly2 - local_cy) ** 2) ** 0.5
            near = min(d1, d2)
            far = max(d1, d2)
            score = far - near * 0.5
            if near <= radius * 0.35 and (best is None or score > best[4]):
                best = (lx1, ly1, lx2, ly2, score)

    if best is None:
        return _detect_gauge_with_numpy(frame, cfg)

    lx1, ly1, lx2, ly2, score = best
    d1 = ((lx1 - local_cx) ** 2 + (ly1 - local_cy) ** 2) ** 0.5
    d2 = ((lx2 - local_cx) ** 2 + (ly2 - local_cy) ** 2) ** 0.5
    tip_x, tip_y = (lx1, ly1) if d1 > d2 else (lx2, ly2)
    angle = _angle_from_center(local_cx, local_cy, tip_x, tip_y)
    status = _status_from_angle(angle, cfg)
    confidence = min(0.98, max(0.55, float(score) / max(radius, 1)))
    _save_gauge_debug(frame, cfg, bbox, angle, status, (x1 + local_cx, y1 + local_cy, x1 + tip_x, y1 + tip_y), cv2)
    return status, angle, confidence, bbox


def _detect_gauge_with_numpy(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
) -> tuple[str, float, float, dict[str, int]] | None:
    image = frame.image
    if image.size == 0 or image.ndim != 3:
        return None
    located = _locate_gauge_by_bright_region(image)
    if located is None:
        return None
    cx, cy, radius = located
    bbox = _circle_bbox(cx, cy, radius, frame.width, frame.height)
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = _gray_numpy(roi)
    local_cx, local_cy = cx - x1, cy - y1
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - local_cx) ** 2 + (yy - local_cy) ** 2)
    inner = rr < radius * 0.85
    center_gap = rr > radius * 0.12
    dark = gray < 80
    candidates = np.where(inner & center_gap & dark)
    if candidates[0].size < 8:
        return None

    ys, xs = candidates
    dist = np.sqrt((xs - local_cx) ** 2 + (ys - local_cy) ** 2)
    idx = int(np.argmax(dist))
    tip_x, tip_y = int(xs[idx]), int(ys[idx])
    angle = _angle_from_center(local_cx, local_cy, tip_x, tip_y)
    status = _status_from_angle(angle, cfg)
    confidence = min(0.95, max(0.55, float(dist[idx]) / max(radius, 1)))
    _save_gauge_debug(frame, cfg, bbox, angle, status, (cx, cy, x1 + tip_x, y1 + tip_y), None)
    return status, angle, confidence, bbox


def _locate_gauge_by_bright_region(image: np.ndarray) -> tuple[int, int, int] | None:
    gray = _gray_numpy(image)
    bright = gray > 120
    ys, xs = np.where(bright)
    if xs.size < 80:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = x2 - x1
    height = y2 - y1
    if width < 20 or height < 20:
        return None
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = max(10, min(width, height) // 2)
    return cx, cy, radius


def _circle_bbox(cx: int, cy: int, radius: int, width: int, height: int) -> dict[str, int]:
    return {
        "x1": max(0, cx - radius),
        "y1": max(0, cy - radius),
        "x2": min(width, cx + radius),
        "y2": min(height, cy + radius),
    }


def _angle_from_center(cx: float, cy: float, tip_x: float, tip_y: float) -> float:
    dx = tip_x - cx
    dy = cy - tip_y
    return (float(np.degrees(np.arctan2(dy, dx))) + 360.0) % 360.0


def _status_from_angle(angle: float, cfg: FixedDetectionConfig) -> str:
    if _angle_in_range(angle, cfg.gauge_low_angle_range):
        return "low"
    if _angle_in_range(angle, cfg.gauge_normal_angle_range):
        return "normal"
    if _angle_in_range(angle, cfg.gauge_high_angle_range):
        return "high"
    return "normal"


def _angle_in_range(angle: float, angle_range: tuple[float, float]) -> bool:
    start, end = angle_range
    angle = angle % 360.0
    start = start % 360.0
    end = end % 360.0
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def _gray_numpy(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    return np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)


def _save_gauge_debug(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
    bbox: dict[str, int],
    angle: float,
    status: str,
    line: tuple[int, int, int, int],
    cv2: Any | None,
) -> None:
    if not cfg.gauge_debug_save_roi:
        return
    out_dir = Path(cfg.gauge_debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    millis = int(frame.timestamp * 1000)
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    roi = frame.image[y1:y2, x1:x2]
    if cv2 is not None:
        cv2.imwrite(str(out_dir / f"gauge_roi_{frame.frame_id:06d}_{millis}.jpg"), roi)
        debug = frame.image.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.line(debug, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
        cv2.putText(
            debug,
            f"{status} {angle:.1f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(str(out_dir / f"gauge_debug_{frame.frame_id:06d}_{millis}.jpg"), debug)
        return
    _write_ppm(out_dir / f"gauge_roi_{frame.frame_id:06d}_{millis}.ppm", roi)
    _write_ppm(out_dir / f"gauge_debug_{frame.frame_id:06d}_{millis}.ppm", frame.image)


def _write_ppm(path: Path, image: np.ndarray) -> None:
    if image.size == 0:
        return
    rgb = image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=2)
    header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + np.ascontiguousarray(rgb).astype(np.uint8).tobytes())


def _try_cv2() -> Any | None:
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None
