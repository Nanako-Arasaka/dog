from __future__ import annotations

from collections import Counter, deque
import os
import subprocess
import time
from typing import Any, Dict, Tuple

import cv2
from ultralytics import YOLO

from gauge_reader import read_gauge


GaugeResult = Dict[str, Any]
Box = Tuple[int, int, int, int]
UnknownCounter = Dict[str, int]

MODEL_PATH = "/home/jetson/yolo_deploy/best.pt"
CAMERA_ID = 0
CAMERA_PATH = "/dev/video0"
CONF_THRES = 0.25
IMG_SIZE = 416
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PRINT_INTERVAL = 1.0
STATUS_HISTORY_SIZE = 7
STATUS_STABLE_MIN_COUNT = 4
UNKNOWN_RESET_COUNT = 5

WINDOW_NAME = "YOLO Gauge Location + OpenCV Reader"
ROI_WINDOW_NAME = "Gauge ROI"

CLASS_NAMES = {
    0: "zone_A",
    1: "zone_B",
    2: "zone_C",
    3: "zone_D",
    4: "gauge",
}
ZONE_CLASSES = {"zone_A", "zone_B", "zone_C", "zone_D"}
STATUS_CN = {
    "low": "偏低",
    "normal": "正常",
    "high": "偏高",
    "unknown": "未知",
}
DEFAULT_GAUGE_RESULT: GaugeResult = {"angle": None, "status": "unknown", "success": False}


def status_name(status: str) -> str:
    return STATUS_CN.get(status, STATUS_CN["unknown"])


def format_angle(angle: Any) -> str:
    return "-" if angle is None else f"{float(angle):.2f}"


def stabilize_gauge_result(
    raw_result: GaugeResult,
    status_history: deque[str],
    last_result: GaugeResult,
    unknown_counter: UnknownCounter,
) -> GaugeResult:
    status = raw_result.get("status", "unknown")
    if raw_result.get("success") and status != "unknown":
        unknown_counter["count"] = 0
        status_history.append(status)
    else:
        unknown_counter["count"] = unknown_counter.get("count", 0) + 1
        if unknown_counter["count"] >= UNKNOWN_RESET_COUNT:
            status_history.clear()
            result: GaugeResult = dict(raw_result)
            result["status"] = "unknown"
            result["stable_count"] = 0
            return result

    if status_history:
        stable_status, count = Counter(status_history).most_common(1)[0]
        if count >= STATUS_STABLE_MIN_COUNT:
            result = dict(raw_result)
            result["status"] = stable_status
            result["stable_count"] = count
            return result

    if last_result.get("status") != "unknown":
        result = dict(raw_result)
        result["status"] = last_result["status"]
        result["stable_count"] = last_result.get("stable_count", 0)
        return result
    return raw_result


def check_environment() -> bool:
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model file not found:")
        print(MODEL_PATH)
        return False
    if not os.path.exists(CAMERA_PATH):
        print("[ERROR] Camera device not found:")
        print(CAMERA_PATH)
        return False
    return True


def run_v4l2_command(command: str) -> None:
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            msg = result.stderr.strip() or result.stdout.strip()
            print(f"[WARNING] v4l2 control failed: {command}")
            if msg:
                print(f"[WARNING] {msg}")
    except Exception as exc:
        print(f"[WARNING] v4l2 control skipped: {command}")
        print(f"[WARNING] {exc}")


def configure_camera_exposure() -> None:
    run_v4l2_command("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_auto=1")
    run_v4l2_command("v4l2-ctl -d /dev/video0 --set-ctrl=exposure_absolute=120")


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    return cap


def draw_text_with_background(frame: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(int(x), 0)
    y = max(int(y), th + padding * 2)
    cv2.rectangle(frame, (x, y - th - padding * 2), (x + tw + padding * 2, y), color, -1)
    cv2.putText(frame, text, (x + padding, y - padding), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def predict_with_fallback(model: YOLO, frame: Any, use_half: bool) -> tuple[Any, bool]:
    try:
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=0, half=use_half, verbose=False)
        return results, use_half
    except Exception as exc:
        if use_half:
            print("[WARNING] YOLO half=True failed, fallback to half=False")
            print(f"[WARNING] {exc}")
            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=0, half=False, verbose=False)
            return results, False
        raise


def draw_overlay(frame: Any, fps: float, object_count: int, zone_name: str | None, gauge_result: GaugeResult) -> None:
    angle = gauge_result.get("angle")
    status = gauge_result.get("status", "unknown")
    status_cn = status_name(status)
    stable_count = gauge_result.get("stable_count", 0)
    circle_found = gauge_result.get("circle_found", False)
    lines = [
        f"FPS: {fps:.1f}",
        f"Objects: {object_count}",
        f"Zone: {zone_name or '-'}",
        f"Gauge angle: {format_angle(angle)}",
        f"Gauge status: {status} / {status_cn} ({stable_count}/{STATUS_HISTORY_SIZE})",
        f"Circle: {'ok' if circle_found else 'fallback'}",
    ]
    y = 32
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        y += 32


def print_inspection_result(zone_name: str | None, gauge_result: GaugeResult) -> None:
    angle = gauge_result.get("angle")
    status = gauge_result.get("status", "unknown")
    status_cn = status_name(status)
    stable_count = gauge_result.get("stable_count", 0)
    print(f"当前区域：{zone_name or 'unknown'}")
    print(f"仪表盘角度：{'unknown' if angle is None else format_angle(angle) + '°'}")
    print(f"仪表盘状态：{status} / {status_cn} ({stable_count}/{STATUS_HISTORY_SIZE})")
    if zone_name:
        print(f"巡检结果：{zone_name} 区域仪表盘{status_cn}")


def main() -> None:
    if not check_environment():
        return

    configure_camera_exposure()
    model = YOLO(MODEL_PATH)
    if getattr(model, "names", None) != CLASS_NAMES:
        print(f"[WARNING] Model class names differ from expected 5-class mapping: {model.names}")

    cap = open_camera()
    if not cap.isOpened():
        print("[ERROR] Failed to open camera:")
        print(CAMERA_PATH)
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    prev_time = time.time()
    last_print_time = 0.0
    use_half = True
    last_gauge_result = dict(DEFAULT_GAUGE_RESULT)
    status_history: deque[str] = deque(maxlen=STATUS_HISTORY_SIZE)
    unknown_counter = {"count": 0}
    last_zone = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame from camera")
                break

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            h, w = frame.shape[:2]
            results, use_half = predict_with_fallback(model, frame, use_half)
            result = results[0]

            object_count = 0
            best_zone = None
            best_zone_conf = -1.0
            best_gauge_box: Box | None = None
            best_gauge_conf = -1.0

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf < CONF_THRES:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = clamp_box((int(x1), int(y1), int(x2), int(y2)), w, h)
                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                    object_count += 1
                    if class_name in ZONE_CLASSES and conf > best_zone_conf:
                        best_zone = class_name
                        best_zone_conf = conf
                    elif class_name == "gauge" and conf > best_gauge_conf:
                        best_gauge_box = (x1, y1, x2, y2)
                        best_gauge_conf = conf

                    color = (0, 255, 0) if class_name in ZONE_CLASSES else (0, 180, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, f"{class_name} {conf:.2f}", x1, y1, color)

            if best_zone is not None:
                last_zone = best_zone

            if best_gauge_box is not None:
                x1, y1, x2, y2 = best_gauge_box
                gauge_roi = frame[y1:y2, x1:x2]
                raw_gauge_result = read_gauge(gauge_roi)
                last_gauge_result = stabilize_gauge_result(raw_gauge_result, status_history, last_gauge_result, unknown_counter)
                if gauge_roi.size > 0:
                    cv2.imshow(ROI_WINDOW_NAME, gauge_roi)

            if now - last_print_time >= PRINT_INTERVAL:
                print_inspection_result(last_zone, last_gauge_result)
                last_print_time = now

            draw_overlay(frame, fps, object_count, last_zone, last_gauge_result)
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
