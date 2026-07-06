"""YOLO 7-class gauge detection with MJPEG streaming.

Uses the 7-class model: zone_A/B/C/D + gauge_low/normal/high.
Status primarily comes directly from YOLO, then a short frame window stabilizes
the displayed result.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
from ultralytics import YOLO

from gauge_reader import read_gauge


# ───────────────── 默认配置 ─────────────────
MODEL_PATH = "/home/jetson/yolo_deploy/best_7class.pt"
CAMERA_ID = 4
CAMERA_PATH = "/dev/video4"
CONF_THRES = 0.25
IMG_SIZE = 416
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PRINT_INTERVAL = 1.0
STREAM_PORT = 8080
STREAM_JPEG_QUALITY = 70
STATUS_HISTORY_SIZE = 7
STATUS_STABLE_MIN_COUNT = 4

WINDOW_NAME = "Jetson YOLO 7-Class Live Detect"

ZONE_CLASSES = {"zone_A", "zone_B", "zone_C", "zone_D"}
GAUGE_STATUS_MAP = {
    "gauge_low": ("low", "偏低"),
    "gauge_normal": ("normal", "正常"),
    "gauge_high": ("high", "偏高"),
}
# cv2.putText cannot render Chinese with Hershey fonts, so on-frame text stays ASCII.
GAUGE_STATUS_OVERLAY = {
    "gauge_low": "LOW",
    "gauge_normal": "NORMAL",
    "gauge_high": "HIGH",
}


# ───────────────── 线程安全的帧缓冲区 ─────────────────
class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def update(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


# ───────────────── MJPEG HTTP 串流服务 ─────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    frame_buffer: FrameBuffer | None = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    frame = MJPEGHandler.frame_buffer.get()
                    if frame is not None:
                        ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
                        if ret:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(buf.tobytes())
                            self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"""<!DOCTYPE html>
<html><head><meta charset=utf-8><title>YOLO Gauge Stream</title>
<style>body{display:flex;justify-content:center;align-items:center;min-height:100vh;
margin:0;background:#111}img{max-width:100vw;max-height:100vh}</style></head>
<body><img src="/stream" alt="live stream"></body></html>"""
            )
        else:
            self.send_response(302)
            self.send_header("Location", "/status")
            self.end_headers()


def start_stream_server(port: int) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ───────────────── 工具函数 ─────────────────
def _get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def check_environment(model_path: str, camera_path: str) -> bool:
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    if not os.path.exists(camera_path):
        print(f"[ERROR] Camera device not found: {camera_path}")
        return False
    return True


def configure_camera_exposure(camera_path: str) -> None:
    try:
        subprocess.run(f"v4l2-ctl -d {camera_path} --set-ctrl=auto_exposure=3", shell=True, timeout=2,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def open_camera(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    return cap


# ───────────────── 绘制工具 ─────────────────
def draw_text_with_background(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(int(x), 0)
    y = max(int(y), th + padding * 2)
    cv2.rectangle(frame, (x, y - th - padding * 2), (x + tw + padding * 2, y), color, -1)
    cv2.putText(frame, text, (x + padding, y - padding), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_overlay(frame, fps: float, object_count: int, zone_name: str | None,
                 gauge_class: str | None, stream_url: str = "",
                 gauge_override: str | None = None,
                 raw_gauge_class: str | None = None,
                 stable_count: int = 0) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Objects: {object_count}",
        f"imgsz: {IMG_SIZE}",
    ]
    if zone_name:
        lines.append(f"Zone: {zone_name}")
    if gauge_class:
        status_en, _status_cn = GAUGE_STATUS_MAP.get(gauge_class, (gauge_class, gauge_class))
        status_overlay = GAUGE_STATUS_OVERLAY.get(gauge_class, status_en.upper())
        src_label = "yolo+color" if gauge_override else "yolo"
        lines.append(f"Gauge: {status_overlay} ({status_en})  src={src_label} stable {stable_count}/{STATUS_HISTORY_SIZE}")
        if raw_gauge_class and raw_gauge_class != gauge_class:
            raw_en, _raw_cn = GAUGE_STATUS_MAP.get(raw_gauge_class, (raw_gauge_class, raw_gauge_class))
            raw_overlay = GAUGE_STATUS_OVERLAY.get(raw_gauge_class, raw_en.upper())
            lines.append(f"Raw gauge: {raw_overlay}")
        if zone_name:
            lines.append(f"Result: {zone_name} {status_overlay}")
    if stream_url:
        lines.append(f"Stream: {stream_url}")

    y = 32
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        y += 34


def predict_with_fallback(model: YOLO, frame, use_half: bool):
    try:
        results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=0, half=use_half, verbose=False)
        return results, use_half
    except Exception as exc:
        if use_half:
            print(f"[WARNING] YOLO half=True failed, fallback to half=False: {exc}")
            results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=0, half=False, verbose=False)
            return results, False
        raise


def stabilize_gauge_result(raw_gauge_class: str | None, status_history: deque[str],
                           last_stable_gauge: str | None) -> tuple[str | None, int]:
    """Use a short majority-vote window to suppress single-frame YOLO jumps."""
    if raw_gauge_class in GAUGE_STATUS_MAP:
        status_history.append(raw_gauge_class)

    if not status_history:
        return last_stable_gauge, 0

    stable_gauge, count = Counter(status_history).most_common(1)[0]
    if count >= STATUS_STABLE_MIN_COUNT:
        return stable_gauge, count

    return last_stable_gauge or raw_gauge_class, count


# ───────────────── 主循环 ─────────────────
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="YOLO 7-class gauge detection with MJPEG streaming")
    parser.add_argument("--model", default=MODEL_PATH, help=f"YOLO model path (default: {MODEL_PATH})")
    parser.add_argument("--camera-id", type=int, default=CAMERA_ID, help=f"Camera index (default: {CAMERA_ID})")
    parser.add_argument("--camera-path", default=CAMERA_PATH, help=f"V4L2 device path (default: {CAMERA_PATH})")
    parser.add_argument("--port", type=int, default=STREAM_PORT, help=f"Streaming port (default: {STREAM_PORT})")
    parser.add_argument("--no-gui", action="store_true", help="Disable local OpenCV windows")
    parser.add_argument("--no-stream", action="store_true", help="Disable MJPEG streaming")
    args = parser.parse_args()

    if not check_environment(args.model, args.camera_path):
        sys.exit(1)

    configure_camera_exposure(args.camera_path)
    model = YOLO(args.model)

    class_names = getattr(model, "names", None) or {}
    expected = {0: "zone_A", 1: "zone_B", 2: "zone_C", 3: "zone_D",
                4: "gauge_low", 5: "gauge_normal", 6: "gauge_high"}
    if class_names != expected:
        print(f"[WARNING] Model classes differ from expected 7-class mapping")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {class_names}")
    print(f"[INFO] Model classes: {class_names}")

    cap = open_camera(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera: {args.camera_path}")
        sys.exit(1)

    frame_buffer = FrameBuffer()
    MJPEGHandler.frame_buffer = frame_buffer
    stream_server = None
    stream_url = ""
    if not args.no_stream:
        stream_server = start_stream_server(args.port)
        local_ip = _get_lan_ip()
        stream_url = f"http://{local_ip}:{args.port}"
        print(f"[INFO] MJPEG stream: {stream_url}")
        print(f"[INFO] Open {stream_url} on your Mac/phone to watch")

    if not args.no_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    prev_time = time.time()
    last_print_time = 0.0
    use_half = True
    gauge_history: deque[str] = deque(maxlen=STATUS_HISTORY_SIZE)
    last_stable_gauge: str | None = None
    last_stable_zone: str | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to read frame from camera")
                break

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            results, use_half = predict_with_fallback(model, frame, use_half)
            result = results[0]

            object_count = 0
            best_zone = None
            best_zone_conf = -1.0
            best_gauge = None
            best_gauge_conf = -1.0
            best_gauge_box = None
            detected_classes: set[str] = set()

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if conf < CONF_THRES:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    class_name = class_names.get(cls_id, f"class_{cls_id}")
                    label = f"{class_name} {conf:.2f}"

                    object_count += 1
                    detected_classes.add(class_name)

                    if class_name in ZONE_CLASSES and conf > best_zone_conf:
                        best_zone = class_name
                        best_zone_conf = conf
                    elif class_name in GAUGE_STATUS_MAP and conf > best_gauge_conf:
                        best_gauge = class_name
                        best_gauge_conf = conf
                        best_gauge_box = (x1, y1, x2, y2)

                    color = (0, 255, 0) if class_name in ZONE_CLASSES else (0, 180, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, x1, y1, color)

            # ── 色带兜底：YOLO 未识别为 high 时，用 gauge_reader 红色带检测纠正 ──
            gauge_override = None
            if best_gauge_box is not None and best_gauge != "gauge_high":
                x1, y1, x2, y2 = best_gauge_box
                gauge_roi = frame[y1:y2, x1:x2]
                gr = read_gauge(gauge_roi)
                if gr.get("status") == "high" and gr.get("status_source") == "color_band":
                    gauge_override = "gauge_high"
                    best_gauge = "gauge_high"

            if best_zone is not None and best_zone != last_stable_zone:
                gauge_history.clear()
                last_stable_gauge = None
                last_stable_zone = best_zone

            stable_gauge, stable_count = stabilize_gauge_result(best_gauge, gauge_history, last_stable_gauge)
            if stable_gauge is not None:
                last_stable_gauge = stable_gauge

            if now - last_print_time >= PRINT_INTERVAL:
                if detected_classes:
                    names = ", ".join(sorted(detected_classes))
                    print(f"Detected: {names}")
                if best_zone:
                    status_en, status_cn = GAUGE_STATUS_MAP.get(stable_gauge or "", ("unknown", "未知"))
                    override_info = " (color-band corrected)" if gauge_override else ""
                    print(f"当前区域：{best_zone}")
                    print(f"仪表盘状态：{status_cn} ({status_en})  src=yolo{override_info} stable={stable_count}/{STATUS_HISTORY_SIZE}")
                    if best_gauge and best_gauge != stable_gauge:
                        _raw_en, raw_cn = GAUGE_STATUS_MAP.get(best_gauge, (best_gauge, best_gauge))
                        print(f"原始单帧状态：{raw_cn}")
                    print(f"巡检结果：{best_zone} 区域仪表盘{status_cn}")
                last_print_time = now

            draw_overlay(frame, fps, object_count, best_zone, stable_gauge, stream_url, gauge_override, best_gauge, stable_count)

            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            frame_buffer.update(display_frame)

            if not args.no_gui:
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if stream_server:
            stream_server.shutdown()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
