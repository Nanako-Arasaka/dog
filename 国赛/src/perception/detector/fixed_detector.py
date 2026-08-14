"""Fixed first-stage detectors for the compute-board vision server.

These detectors are intentionally simple. They provide stable structured
outputs for the remote-perception contract before full OCR/meter models are
integrated. This module is intentionally limited to inspection recognition:
zone letters, gauges, and fused inspection results.
"""

from __future__ import annotations

import struct
import time
import zlib
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
    letter_min_confidence: float = 0.55
    letter_template_dir: str = "assets/templates/letters"
    letter_debug_save_roi: bool = False
    letter_debug_dir: str = "output/debug_letters"
    inspection_debug_save: bool = False
    inspection_debug_dir: str = "output/debug_inspection"
    inspection_max_match_distance: float = 180.0
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
        if req == "detect_zone_letters":
            return self.detect_zone_letters(frame)
        if req == "detect_gauges":
            return self.detect_gauges(frame)
        if req == "poll_inspection":
            return self.poll_inspection(frame)
        return {"type": "error", "message": f"unknown request: {req}", "timestamp": _timestamp(frame)}

    def detect_zone_letters(self, frame: VisionFrame | None) -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("detect_zone_letters")
        detections = _detect_letters_from_frame(frame, self._cfg)
        return {
            "type": "zone_letters",
            "detections": detections,
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

    def poll_inspection(self, frame: VisionFrame | None) -> dict[str, Any]:
        if frame is None or self._cfg.empty_results:
            return empty_response("poll_inspection")
        letters_resp = self.detect_zone_letters(frame)
        gauges_resp = self.detect_gauges(frame)
        results = fuse_inspection_results(
            letters_resp.get("detections", []),
            gauges_resp.get("detections", []),
            timestamp=frame.timestamp,
            letter_min_confidence=self._cfg.letter_min_confidence,
            gauge_min_confidence=self._cfg.gauge_min_confidence,
            max_match_distance=self._cfg.inspection_max_match_distance,
        )
        _save_inspection_debug(frame, self._cfg, results, _try_cv2())
        return {
            "type": "inspection_results",
            "results": results,
            "detections": results,
            "timestamp": frame.timestamp,
        }



def empty_response(req: str) -> dict[str, Any]:
    ts = time.time()
    if req == "detect_zone_letters":
        return {"type": "zone_letters", "detections": [], "timestamp": ts}
    if req == "detect_gauges":
        return {"type": "gauges", "detections": [], "timestamp": ts}
    if req == "poll_inspection":
        return {"type": "inspection_results", "results": [], "detections": [], "timestamp": ts}
    return {"type": "error", "message": f"unknown request: {req}", "timestamp": ts}


def fuse_inspection_results(
    zone_letters: list[dict[str, Any]],
    gauges: list[dict[str, Any]],
    *,
    timestamp: float | None = None,
    letter_min_confidence: float = 0.55,
    gauge_min_confidence: float = 0.55,
    max_match_distance: float = 180.0,
) -> list[dict[str, Any]]:
    ts = time.time() if timestamp is None else timestamp
    valid_letters = [
        item for item in zone_letters
        if str(item.get("zone", "")).upper() in {"A", "B", "C", "D"}
        and float(item.get("confidence", 0.0)) >= letter_min_confidence
    ]
    valid_gauges = [
        item for item in gauges
        if str(item.get("status", "")).lower() in {"low", "normal", "high"}
        and float(item.get("confidence", 0.0)) >= gauge_min_confidence
    ]
    if not valid_letters or not valid_gauges:
        return []

    pairs = _match_letters_to_gauges(valid_letters, valid_gauges, max_match_distance)
    results: list[dict[str, Any]] = []
    for letter, gauge in pairs:
        zone = str(letter.get("zone", "")).upper()
        status = str(gauge.get("status", "normal")).lower()
        confidence = min(float(letter.get("confidence", 0.0)), float(gauge.get("confidence", 0.0)))
        abnormal = status in {"low", "high"}
        text = _inspection_text(zone, status, abnormal)
        result = {
            "zone": zone,
            "gauge_status": status,
            "status": status,
            "abnormal": abnormal,
            "confidence": round(confidence, 4),
            "letter_bbox": letter.get("bbox"),
            "gauge_bbox": gauge.get("bbox"),
            "bbox": {
                "letter": letter.get("bbox"),
                "gauge": gauge.get("bbox"),
            },
            "speak_key": f"{zone}_{status}",
            "text": text,
            "timestamp": ts,
        }
        results.append(result)
    results.sort(key=lambda item: item["zone"])
    return results


def _match_letters_to_gauges(
    letters: list[dict[str, Any]],
    gauges: list[dict[str, Any]],
    max_match_distance: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[float, int, int]] = []
    for li, letter in enumerate(letters):
        lb = letter.get("bbox")
        if not lb:
            continue
        lc = _bbox_center(lb)
        for gi, gauge in enumerate(gauges):
            gb = gauge.get("bbox")
            if not gb:
                continue
            gc = _bbox_center(gb)
            dist = ((lc[0] - gc[0]) ** 2 + (lc[1] - gc[1]) ** 2) ** 0.5
            pairs.append((dist, li, gi))

    matched_letters: set[int] = set()
    matched_gauges: set[int] = set()
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for dist, li, gi in sorted(pairs, key=lambda item: item[0]):
        if dist > max_match_distance:
            continue
        if li in matched_letters or gi in matched_gauges:
            continue
        matched_letters.add(li)
        matched_gauges.add(gi)
        result.append((letters[li], gauges[gi]))

    # Fallback: stable order matching for anything not paired spatially.
    remaining_letters = [item for i, item in enumerate(letters) if i not in matched_letters]
    remaining_gauges = [item for i, item in enumerate(gauges) if i not in matched_gauges]
    remaining_letters.sort(key=lambda item: str(item.get("zone", "")))
    for letter, gauge in zip(remaining_letters, remaining_gauges):
        result.append((letter, gauge))
    return result


def _inspection_text(zone: str, status: str, abnormal: bool) -> str:
    status_cn = {
        "low": "偏低",
        "normal": "正常",
        "high": "偏高",
    }.get(status, status)
    health = "异常" if abnormal else "正常"
    return f"{zone}区域仪表盘显示{status_cn}，状态{health}"


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox.get("x1", 0)) + float(bbox.get("x2", 0))) / 2.0,
        (float(bbox.get("y1", 0)) + float(bbox.get("y2", 0))) / 2.0,
    )


def _timestamp(frame: VisionFrame | None) -> float:
    return float(frame.timestamp) if frame is not None else time.time()


def _detect_letters_from_frame(frame: VisionFrame, cfg: FixedDetectionConfig) -> list[dict[str, Any]]:
    templates = _ensure_letter_templates(cfg.letter_template_dir)
    image = frame.image
    if image.size == 0 or image.ndim != 3:
        return []

    cv2 = _try_cv2()
    if cv2 is not None:
        candidates = _letter_candidates_cv2(image, cv2)
        detections = _detections_from_letter_candidates(frame, cfg, candidates, templates, cv2)
        if detections:
            return detections
        # OpenCV thresholding can merge the synthetic/field white dial region
        # into one large contour. The NumPy connected-component path is more
        # conservative for black-on-light zone labels, so use it as a fallback.
        candidates = _letter_candidates_numpy(image)
    else:
        candidates = _letter_candidates_numpy(image)

    return _detections_from_letter_candidates(frame, cfg, candidates, templates, cv2)


def _detections_from_letter_candidates(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
    candidates: list[tuple[dict[str, int], np.ndarray]],
    templates: dict[str, np.ndarray],
    cv2: Any | None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bbox, binary_roi in candidates:
        match = _match_letter_template(binary_roi, templates)
        if match is None:
            continue
        letter, confidence = match
        if confidence < cfg.letter_min_confidence or letter in seen:
            continue
        seen.add(letter)
        detection = {
            "zone": letter,
            "letter": letter,
            "object_type": "zone_letter",
            "bbox": bbox,
            "confidence": round(float(confidence), 4),
            "timestamp": frame.timestamp,
        }
        detections.append(detection)
        _save_letter_debug(frame, cfg, bbox, letter, confidence, cv2)

    detections.sort(key=lambda item: item["zone"])
    return detections


def _letter_candidates_cv2(image: np.ndarray, cv2: Any) -> list[tuple[dict[str, int], np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[dict[str, int], np.ndarray]] = []
    h, w = gray.shape
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = int(cv2.contourArea(contour))
        candidate = _normalize_letter_candidate(binary, x, y, bw, bh, area, w, h)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _letter_candidates_numpy(image: np.ndarray) -> list[tuple[dict[str, int], np.ndarray]]:
    gray = _gray_numpy(image)
    mask = gray < 90
    return _connected_components(mask)


def _connected_components(mask: np.ndarray) -> list[tuple[dict[str, int], np.ndarray]]:
    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    candidates: list[tuple[dict[str, int], np.ndarray]] = []

    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True
        points_x: list[int] = []
        points_y: list[int] = []
        while stack:
            x, y = stack.pop()
            points_x.append(x)
            points_y.append(y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))

        x1, x2 = min(points_x), max(points_x) + 1
        y1, y2 = min(points_y), max(points_y) + 1
        bw, bh = x2 - x1, y2 - y1
        candidate = _normalize_letter_candidate(mask.astype(np.uint8) * 255, x1, y1, bw, bh, len(points_x), w, h)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _normalize_letter_candidate(
    binary_image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    area: int,
    image_width: int,
    image_height: int,
) -> tuple[dict[str, int], np.ndarray] | None:
    if width < 18 or height < 28 or area < 60:
        return None
    if width > image_width * 0.45 or height > image_height * 0.65:
        return None
    aspect = width / max(height, 1)
    if not 0.25 <= aspect <= 1.4:
        return None
    pad = max(4, int(max(width, height) * 0.08))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_width, x + width + pad)
    y2 = min(image_height, y + height + pad)
    roi = binary_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    roi_binary = roi > 0
    bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return bbox, roi_binary


def _match_letter_template(
    roi_binary: np.ndarray,
    templates: dict[str, np.ndarray],
) -> tuple[str, float] | None:
    if roi_binary.size == 0:
        return None
    target = _resize_binary(roi_binary, (96, 64))
    best_letter = ""
    best_score = -1.0
    for letter, template in templates.items():
        tmpl = _resize_binary(template, target.shape)
        intersection = np.logical_and(target, tmpl).sum()
        union = np.logical_or(target, tmpl).sum()
        iou = float(intersection) / float(max(union, 1))
        same = float((target == tmpl).sum()) / float(target.size)
        score = 0.7 * iou + 0.3 * same
        if score > best_score:
            best_score = score
            best_letter = letter
    if not best_letter:
        return None
    return best_letter, best_score


def _ensure_letter_templates(template_dir: str) -> dict[str, np.ndarray]:
    root = Path(template_dir)
    root.mkdir(parents=True, exist_ok=True)
    templates: dict[str, np.ndarray] = {}
    for letter in ("A", "B", "C", "D"):
        mask = _letter_template_mask(letter, width=80, height=120)
        path = root / f"{letter}.png"
        if not path.exists():
            _write_png_gray(path, np.where(mask, 0, 255).astype(np.uint8))
        templates[letter] = mask
    return templates


def _letter_template_mask(letter: str, width: int = 80, height: int = 120) -> np.ndarray:
    patterns = {
        "A": [
            "0011100",
            "0110110",
            "1100011",
            "1100011",
            "1111111",
            "1100011",
            "1100011",
            "1100011",
            "1100011",
        ],
        "B": [
            "1111100",
            "1100110",
            "1100011",
            "1100110",
            "1111100",
            "1100110",
            "1100011",
            "1100110",
            "1111100",
        ],
        "C": [
            "0011110",
            "0110011",
            "1100000",
            "1100000",
            "1100000",
            "1100000",
            "1100000",
            "0110011",
            "0011110",
        ],
        "D": [
            "1111000",
            "1101100",
            "1100110",
            "1100011",
            "1100011",
            "1100011",
            "1100110",
            "1101100",
            "1111000",
        ],
    }
    rows = patterns[letter]
    small = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    return _resize_binary(small, (height, width))


def _resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    in_h, in_w = mask.shape
    y_idx = np.linspace(0, in_h - 1, out_h).astype(np.int64)
    x_idx = np.linspace(0, in_w - 1, out_w).astype(np.int64)
    return mask[y_idx][:, x_idx].astype(bool)


def _save_letter_debug(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
    bbox: dict[str, int],
    letter: str,
    confidence: float,
    cv2: Any | None,
) -> None:
    if not cfg.letter_debug_save_roi:
        return
    out_dir = Path(cfg.letter_debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    millis = int(frame.timestamp * 1000)
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    roi = frame.image[y1:y2, x1:x2]
    safe_conf = int(confidence * 1000)
    if cv2 is not None:
        _cv2_imwrite(cv2, out_dir / f"letter_roi_{letter}_{safe_conf}_{frame.frame_id:06d}_{millis}.jpg", roi)
        debug = frame.image.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            debug,
            f"{letter} {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        _cv2_imwrite(cv2, out_dir / f"letter_debug_{letter}_{safe_conf}_{frame.frame_id:06d}_{millis}.jpg", debug)
        return
    debug = frame.image.copy()
    _draw_rect_numpy(debug, bbox, color=(255, 255, 0))
    _write_ppm(out_dir / f"letter_roi_{letter}_{safe_conf}_{frame.frame_id:06d}_{millis}.ppm", roi)
    _write_ppm(out_dir / f"letter_debug_{letter}_{safe_conf}_{frame.frame_id:06d}_{millis}.ppm", debug)


def _draw_rect_numpy(image: np.ndarray, bbox: dict[str, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    x1 = max(0, min(image.shape[1] - 1, x1))
    x2 = max(0, min(image.shape[1] - 1, x2))
    y1 = max(0, min(image.shape[0] - 1, y1))
    y2 = max(0, min(image.shape[0] - 1, y2))
    image[y1:y1 + 2, x1:x2] = color
    image[max(y2 - 2, y1):y2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, max(x2 - 2, x1):x2] = color


def _save_inspection_debug(
    frame: VisionFrame,
    cfg: FixedDetectionConfig,
    results: list[dict[str, Any]],
    cv2: Any | None,
) -> None:
    if not cfg.inspection_debug_save or not results:
        return
    out_dir = Path(cfg.inspection_debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    millis = int(frame.timestamp * 1000)
    debug = frame.image.copy()
    for item in results:
        letter_bbox = item.get("letter_bbox")
        gauge_bbox = item.get("gauge_bbox")
        label = f"{item.get('zone', '?')}_{item.get('gauge_status', '?')} {float(item.get('confidence', 0.0)):.2f}"
        if cv2 is not None:
            if letter_bbox:
                cv2.rectangle(
                    debug,
                    (int(letter_bbox["x1"]), int(letter_bbox["y1"])),
                    (int(letter_bbox["x2"]), int(letter_bbox["y2"])),
                    (255, 255, 0),
                    2,
                )
            if gauge_bbox:
                cv2.rectangle(
                    debug,
                    (int(gauge_bbox["x1"]), int(gauge_bbox["y1"])),
                    (int(gauge_bbox["x2"]), int(gauge_bbox["y2"])),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    debug,
                    label,
                    (int(gauge_bbox["x1"]), max(20, int(gauge_bbox["y1"]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
        else:
            if letter_bbox:
                _draw_rect_numpy(debug, letter_bbox, color=(255, 255, 0))
            if gauge_bbox:
                _draw_rect_numpy(debug, gauge_bbox, color=(0, 255, 255))
    if cv2 is not None:
        _cv2_imwrite(cv2, out_dir / f"inspection_debug_{frame.frame_id:06d}_{millis}.jpg", debug)
    else:
        _write_ppm(out_dir / f"inspection_debug_{frame.frame_id:06d}_{millis}.ppm", debug)


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
        # OpenCV 4.x 返回 (N,1,4), 5.x 返回 (N,4); 统一 reshape 兼容
        lines = np.atleast_2d(np.asarray(lines)).reshape(-1, 4)
        for line in lines:
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
    components = _bright_components(bright)
    if not components:
        return None

    best: tuple[int, int, int, int, int] | None = None
    best_score = -1.0
    for x1, y1, x2, y2, area in components:
        width = x2 - x1
        height = y2 - y1
        if width < 30 or height < 30:
            continue
        aspect = width / max(height, 1)
        if not 0.65 <= aspect <= 1.35:
            continue
        box_area = max(width * height, 1)
        fill = area / box_area
        # A filled circular dial has a high but not rectangular fill ratio.
        if not 0.45 <= fill <= 0.95:
            continue
        score = area * (1.0 - abs(1.0 - aspect))
        if score > best_score:
            best = (x1, y1, x2, y2, area)
            best_score = score
    if best is None:
        return None

    x1, y1, x2, y2, _area = best
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = max(10, min(width, height) // 2)
    return cx, cy, radius


def _bright_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        stack = [(start_x, start_y)]
        visited[start_y, start_x] = True
        min_x = max_x = start_x
        min_y = max_y = start_y
        area = 0
        while stack:
            x, y = stack.pop()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))
        components.append((min_x, min_y, max_x + 1, max_y + 1, area))
    return components


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
        _cv2_imwrite(cv2, out_dir / f"gauge_roi_{frame.frame_id:06d}_{millis}.jpg", roi)
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
        _cv2_imwrite(cv2, out_dir / f"gauge_debug_{frame.frame_id:06d}_{millis}.jpg", debug)
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


def _write_png_gray(path: Path, image: np.ndarray) -> None:
    img = np.ascontiguousarray(image.astype(np.uint8))
    height, width = img.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _cv2_imwrite(cv2: Any, path: Path, image: np.ndarray) -> bool:
    if cv2.imwrite(str(path), image):
        return True
    suffix = path.suffix or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return True


def _try_cv2() -> Any | None:
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None
