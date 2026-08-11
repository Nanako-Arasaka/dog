"""OpenCV-based analog gauge pointer reader.

This module is intentionally independent from YOLO. YOLO only crops the gauge
ROI; this code estimates the pointer angle and maps it to low/normal/high.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


# Fallback gauge angle thresholds for the current competition gauge.
# These values are field-calibrated parameters and should be adjusted again
# when the camera position, gauge type, or installation angle changes.
# The main status decision below first tries to use the colored dial bands
# because image-space pointer angle changes when the handheld camera rotates.
#
# Angle ranges (degrees, from _angle_from_center: 0°=right, 90°=up, 180°=left):
HIGH_MAX_ANGLE = 15.0          # angle <= this → high (pointer far right)
LOW_MIN_ANGLE = 120.0          # angle >= this → low (pointer far left)
ANGLE_VALID_MAX = 180.0        # beyond this, angle is unreliable → unknown
# Normal = everything between HIGH_MAX_ANGLE and LOW_MIN_ANGLE

COLOR_BAND_EDGE_MARGIN_DEG = 8.0
COLOR_BAND_MAX_SPAN_DEG = 200.0  # reject spans wider than this (noise, not a real band)


def _unknown(pointer_detected: bool = False) -> dict[str, Any]:
    return {
        "angle": None,
        "status": "unknown",
        "success": False,
        "circle_found": False,
        "status_source": "unknown",
        "color_band_detected": False,
        "pointer_detected": pointer_detected,
    }


def _angle_from_center(cx: float, cy: float, x: float, y: float) -> float:
    dx = x - cx
    dy = cy - y
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def _status_from_angle(angle: float) -> str:
    """Fallback: determine gauge status from pointer angle alone.

    Only used when color-band detection fails.  Returns "unknown" when
    the angle falls outside the calibrated valid range so that we never
    silently misclassify an abnormal reading as normal.
    """
    if angle > ANGLE_VALID_MAX:
        return "unknown"
    if angle <= HIGH_MAX_ANGLE:
        return "high"
    if angle >= LOW_MIN_ANGLE:
        return "low"
    return "normal"


def _angle_distance(a: np.ndarray | float, b: float) -> np.ndarray | float:
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _mask_angles(mask: np.ndarray, cx: float, cy: float) -> np.ndarray:
    ys, xs = np.where(mask)
    if xs.size < 20:
        return np.array([], dtype=np.float32)

    dx = xs.astype(np.float32) - cx
    dy = cy - ys.astype(np.float32)
    return (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0


def _angle_span(angles: np.ndarray) -> tuple[float, float, float] | None:
    if angles.size < 20:
        return None

    sorted_angles = np.sort(angles.astype(np.float32) % 360.0)
    gaps = np.diff(np.concatenate([sorted_angles, sorted_angles[:1] + 360.0]))
    gap_index = int(np.argmax(gaps))
    start = float(sorted_angles[(gap_index + 1) % sorted_angles.size])
    end = float(sorted_angles[gap_index])
    width = (end - start) % 360.0
    return start, end, width


def _angle_in_span(angle: float, span: tuple[float, float, float], margin: float = 0.0) -> bool:
    start, end, width = span
    if width <= 0.0 or width > COLOR_BAND_MAX_SPAN_DEG:
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

    red1 = cv2.inRange(hsv, np.array([0, 65, 40]), np.array([12, 255, 255])) > 0
    red2 = cv2.inRange(hsv, np.array([168, 65, 40]), np.array([179, 255, 255])) > 0
    red_mask = (red1 | red2) & annulus
    yellow_mask = (cv2.inRange(hsv, np.array([12, 30, 30]), np.array([55, 255, 255])) > 0) & annulus

    red_pixels = int(np.sum(red_mask))
    yellow_pixels = int(np.sum(yellow_mask))
    annulus_pixels = int(np.sum(annulus))

    red_angles = _mask_angles(red_mask, cx, cy)
    yellow_angles = _mask_angles(yellow_mask, cx, cy)
    red_span = _angle_span(red_angles)
    yellow_span = _angle_span(yellow_angles)

    print(f"[DEBUG color_bands] annulus_px={annulus_pixels} red_px={red_pixels} yellow_px={yellow_pixels} "
          f"red_angles_n={red_angles.size} yellow_angles_n={yellow_angles.size} "
          f"pointer_angle={pointer_angle:.1f} circle=({cx:.0f},{cy:.0f}) r={radius:.0f} "
          f"red_span={red_span} yellow_span={yellow_span}")

    high_hit = red_span is not None and _angle_in_span(pointer_angle, red_span, COLOR_BAND_EDGE_MARGIN_DEG)
    low_hit = yellow_span is not None and _angle_in_span(pointer_angle, yellow_span, COLOR_BAND_EDGE_MARGIN_DEG)

    print(f"[DEBUG color_bands] high_hit={high_hit} low_hit={low_hit}")

    if high_hit and not low_hit:
        return "high"
    if low_hit and not high_hit:
        return "low"
    return None


def _detect_gauge_circle(gray: np.ndarray) -> tuple[float, float, float] | None:
    h, w = gray.shape[:2]
    min_side = min(w, h)
    if min_side < 40:
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
    best: tuple[float, float, float] | None = None
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


def _choose_dark_tip(gray: np.ndarray, cx: float, cy: float, radius: float) -> tuple[float, float] | None:
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


def read_gauge(gauge_roi: Any, debug: bool = False) -> dict[str, Any]:
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

    circle = _detect_gauge_circle(gray)
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
    print(f"[DEBUG read_gauge] circle_found={circle_found} cx={cx:.1f} cy={cy:.1f} radius={radius:.1f} "
          f"roi_shape=({w},{h}) pointer_angle={angle}")

    # ── status decision: color-band first, then angle fallback ──
    color_result = _status_from_color_bands(gauge_roi, angle, cx, cy, radius)
    if color_result is not None:
        status = color_result
        status_source = "color_band"
        color_band_detected = True
    else:
        status = _status_from_angle(angle)
        status_source = "angle"
        color_band_detected = False

    print(f"[DEBUG read_gauge] status={status} source={status_source}")

    result: dict[str, Any] = {
        "angle": angle,
        "status": status,
        "success": True,
        "circle_found": circle_found,
        "status_source": status_source,
        "color_band_detected": color_band_detected,
        "pointer_detected": True,
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
