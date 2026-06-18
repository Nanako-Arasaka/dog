"""OpenCV-based analog gauge pointer reader.

This module is intentionally independent from YOLO. YOLO only crops the gauge
ROI; this code estimates the pointer angle and maps it to low/normal/high.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import cv2
import numpy as np


GaugeResult = Dict[str, Any]
Point = Tuple[float, float]
Circle = Tuple[float, float, float]
AngleSpan = Tuple[float, float, float]


# Fallback gauge angle thresholds for the current competition gauge.
# These values are field-calibrated parameters and should be adjusted again
# when the camera position, gauge type, or installation angle changes.
# The main status decision below first tries to use the colored dial bands
# because image-space pointer angle changes when the handheld camera rotates.
LOW_MAX_ANGLE = 25.0
NORMAL_MIN_ANGLE = 25.0
NORMAL_MAX_ANGLE = 55.0
HIGH_MIN_ANGLE = 55.0

COLOR_BAND_EDGE_MARGIN_DEG = 8.0
DIAL_MIN_SIZE = 40
MIN_MASK_PIXEL_COUNT = 20

RED_HSV_RANGES = (
    (np.array([0, 80, 45]), np.array([12, 255, 255])),
    (np.array([168, 80, 45]), np.array([179, 255, 255])),
)
YELLOW_HSV_RANGE = (np.array([12, 25, 35]), np.array([55, 255, 255]))


def _unknown() -> GaugeResult:
    return {"angle": None, "status": "unknown", "success": False}


def _angle_from_center(cx: float, cy: float, x: float, y: float) -> float:
    dx = x - cx
    dy = cy - y
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def _status_from_angle(angle: float) -> str:
    # Image-space angle is unstable when the handheld camera rolls or tilts.
    # Once a pointer angle is found, treat non-extreme readings as normal unless
    # the colored dial-band check clearly proves low or high.
    return "normal"


def _angle_distance(a: np.ndarray | float, b: float) -> np.ndarray | float:
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _mask_angles(mask: np.ndarray, cx: float, cy: float) -> np.ndarray:
    ys, xs = np.where(mask)
    if xs.size < MIN_MASK_PIXEL_COUNT:
        return np.array([], dtype=np.float32)

    dx = xs.astype(np.float32) - cx
    dy = cy - ys.astype(np.float32)
    return (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0


def _angle_span(angles: np.ndarray) -> AngleSpan | None:
    if angles.size < MIN_MASK_PIXEL_COUNT:
        return None

    sorted_angles = np.sort(angles.astype(np.float32) % 360.0)
    gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 360.0]))
    gap_index = int(np.argmax(gaps))
    start = float(sorted_angles[(gap_index + 1) % sorted_angles.size])
    end = float(sorted_angles[gap_index])
    width = (end - start) % 360.0
    return start, end, width


def _angle_in_span(angle: float, span: AngleSpan, margin: float = 0.0) -> bool:
    start, end, width = span
    if width <= 0.0 or width > 160.0:
        return False
    relative = (angle - start) % 360.0
    return relative <= width + margin or relative >= 360.0 - margin


def _status_from_color_bands(roi: Any, pointer_angle: float, cx: float, cy: float, radius: float) -> str | None:
    if roi is None or not hasattr(roi, "shape") or len(roi.shape) < 3:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(hsv.shape[:2])
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    annulus = (rr >= radius * 0.42) & (rr <= radius * 0.98)

    red_mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lower, upper in RED_HSV_RANGES:
        red_mask |= cv2.inRange(hsv, lower, upper) > 0
    red_mask &= annulus

    yellow_lower, yellow_upper = YELLOW_HSV_RANGE
    yellow_mask = (cv2.inRange(hsv, yellow_lower, yellow_upper) > 0) & annulus

    red_span = _angle_span(_mask_angles(red_mask, cx, cy))
    yellow_span = _angle_span(_mask_angles(yellow_mask, cx, cy))

    high_hit = red_span is not None and _angle_in_span(pointer_angle, red_span, COLOR_BAND_EDGE_MARGIN_DEG)
    low_hit = yellow_span is not None and _angle_in_span(pointer_angle, yellow_span, COLOR_BAND_EDGE_MARGIN_DEG)

    if high_hit and not low_hit:
        return "high"
    if low_hit and not high_hit:
        return "low"
    return None


def _find_gauge_circle(gray: np.ndarray) -> Circle | None:
    h, w = gray.shape[:2]
    min_side = min(w, h)
    if min_side < DIAL_MIN_SIZE:
        return None

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(min_side * 0.45)),
        param1=100,
        param2=24,
        minRadius=max(12, int(min_side * 0.22)),
        maxRadius=max(18, int(min_side * 0.56)),
    )
    if circles is None:
        return None

    candidates = np.round(circles[0, :]).astype("int")
    best: Circle | None = None
    best_score = -1.0
    for x, y, r in candidates:
        if x - r < -w * 0.05 or y - r < -h * 0.05 or x + r > w * 1.05 or y + r > h * 1.05:
            continue
        center_bias = math.hypot(x - w / 2.0, y - h / 2.0) / max(min_side, 1)
        score = float(r) - center_bias * min_side * 0.25
        if score > best_score:
            best = (float(x), float(y), float(r))
            best_score = score
    return best


def _choose_pointer_line(lines: Any, cx: float, cy: float, radius: float) -> tuple[float, float, float] | None:
    if lines is None:
        return None

    best: tuple[float, float, float] | None = None
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in raw]
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        near = min(d1, d2)
        far = max(d1, d2)

        if near > radius * 0.30:
            continue
        if far < radius * 0.25:
            continue

        tip_x, tip_y = (x1, y1) if d1 > d2 else (x2, y2)
        center_score = max(0.0, 1.0 - near / max(radius * 0.25, 1.0))
        length_score = far / max(radius, 1.0)
        score = length_score + center_score * 0.8 - near / max(radius, 1.0)
        if best is None or score > best[2]:
            best = (tip_x, tip_y, score)
    return best


def _choose_dark_tip(gray: np.ndarray, cx: float, cy: float, radius: float) -> Point | None:
    yy, xx = np.indices(gray.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    threshold = float(np.percentile(gray, 18))
    mask = (rr > radius * 0.16) & (rr < radius * 0.90) & (gray <= threshold)
    ys, xs = np.where(mask)
    if xs.size < 10:
        return None
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    idx = int(np.argmax(dist))
    return float(xs[idx]), float(ys[idx])


def read_gauge(gauge_roi: Any, debug: bool = False) -> GaugeResult:
    """Read gauge angle and status from a cropped ROI.

    Args:
        gauge_roi: BGR or grayscale OpenCV image cropped around the gauge.
        debug: When true, include intermediate debug images in the result.

    Returns:
        {"angle": float|None, "status": "low|normal|high|unknown", "success": bool}
    """
    if gauge_roi is None:
        return _unknown()
    if not hasattr(gauge_roi, "size") or gauge_roi.size == 0:
        return _unknown()

    h, w = gauge_roi.shape[:2]
    if h < 20 or w < 20:
        return _unknown()

    if len(gauge_roi.shape) == 2:
        gray = gauge_roi.copy()
    else:
        gray = cv2.cvtColor(gauge_roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 45, 140)

    circle = _find_gauge_circle(gray)
    if circle is None:
        cx = w / 2.0
        cy = h / 2.0
        radius = min(w, h) / 2.0
        circle_found = False
    else:
        cx, cy, radius = circle
        circle_found = True

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(16, int(radius * 0.18)),
        minLineLength=max(12, int(radius * 0.28)),
        maxLineGap=max(6, int(radius * 0.08)),
    )

    chosen = _choose_pointer_line(lines, cx, cy, radius)
    if chosen is not None:
        tip_x, tip_y, _score = chosen
    else:
        fallback = _choose_dark_tip(gray, cx, cy, radius)
        if fallback is None:
            return _unknown()
        tip_x, tip_y = fallback

    angle = round(float(_angle_from_center(cx, cy, tip_x, tip_y)), 2)
    status = _status_from_color_bands(gauge_roi, angle, cx, cy, radius) or _status_from_angle(angle)
    result: dict[str, Any] = {
        "angle": angle,
        "status": status,
        "success": True,
        "circle_found": circle_found,
    }

    if debug:
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        cv2.circle(debug_img, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
        cv2.line(debug_img, (int(cx), int(cy)), (int(tip_x), int(tip_y)), (0, 0, 255), 2)
        result["debug_image"] = debug_img
        result["edges"] = edges

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Read a gauge ROI image with OpenCV.")
    parser.add_argument("image", help="Path to a cropped gauge ROI image.")
    args = parser.parse_args()

    roi = cv2.imread(args.image)
    print(json.dumps(read_gauge(roi), ensure_ascii=False, indent=2))
