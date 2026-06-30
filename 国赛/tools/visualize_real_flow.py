#!/usr/bin/env python3
"""Visual dashboard for camera, red-bar detection, and task-flow ROS topics."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except ImportError:
    PILImage = None
    ImageDraw = None
    ImageFont = None


Box = Tuple[int, int, int, int]


DEFAULT_CONFIG = "arm_grasp/config/grasp_config.yaml"
DEFAULT_COLOR_TOPIC = "/rgbd_cam/color/image_rect_color"
DEFAULT_DEPTH_TOPIC = "/rgbd_cam/depth/image_raw"
DEFAULT_INFO_TOPIC = "/rgbd_cam/color/camera_info"
_FONT_CACHE = {}


def find_unicode_font(size: int = 18):
    if ImageFont is None:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    _FONT_CACHE[size] = font
    return font


def load_hsv(config_path: Path, color: str) -> Dict[str, List[int]]:
    fallback = {
        "red": {
            "lower": [0, 120, 100],
            "upper": [10, 255, 255],
            "lower2": [170, 120, 100],
            "upper2": [180, 255, 255],
        },
        "green": {"lower": [40, 80, 80], "upper": [85, 255, 255]},
    }
    if not config_path.exists():
        return fallback[color]
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("object_config", {}).get("hsv_ranges", fallback).get(color, fallback[color])


def detect_color_bars(frame, hsv_range: Dict[str, List[int]], min_area: float):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_range["lower"]), np.array(hsv_range["upper"]))
    if "lower2" in hsv_range and "upper2" in hsv_range:
        mask2 = cv2.inRange(hsv, np.array(hsv_range["lower2"]), np.array(hsv_range["upper2"]))
        mask = cv2.bitwise_or(mask, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(int)
        cx, cy = rect[0]
        w, h = rect[1]
        angle = rect[2]
        if w < h:
            w, h = h, w
            angle += 90
        ratio = w / h if h > 0 else 0.0
        if 1.3 < ratio < 3.5:
            objects.append(
                {
                    "area": area,
                    "box": box,
                    "cx": int(cx),
                    "cy": int(cy),
                    "w": float(w),
                    "h": float(h),
                    "ratio": float(ratio),
                    "angle": float(angle),
                }
            )
    objects.sort(key=lambda item: item["area"], reverse=True)
    return mask, objects


def sample_depth(depth_img, cx: int, cy: int, radius: int = 5) -> Optional[float]:
    if depth_img is None:
        return None
    h, w = depth_img.shape[:2]
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    roi = depth_img[y1:y2, x1:x2]
    valid = roi[(roi > 0) & (roi < 2500)]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0


def put_line(image, text: str, x: int, y: int, color=(255, 255, 255)) -> int:
    if PILImage is not None and ImageDraw is not None:
        font = find_unicode_font(18)
        if font is not None:
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            rgb = (int(color[2]), int(color[1]), int(color[0]))
            draw.text((x, max(0, y - 18)), text, font=font, fill=rgb)
            image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
            return y + 24

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return y + 24


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show camera image, red-bar detection, and ROS task topics.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--color", default="red", choices=("red", "green"))
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument("--window", default="Competition Real Flow Visualizer")
    parser.add_argument("--web", action="store_true", help="Serve the visualizer as a browser page instead of using cv2.imshow.")
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8080)
    return parser


class WebStreamer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._jpeg = None
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<title>Competition Real Flow</title>"
                        "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                        "header{padding:10px 14px;background:#222}img{width:100%;height:auto;display:block}</style>"
                        "</head><body><header>Competition Real Flow Visualizer</header>"
                        "<img id='frame' src='/frame.jpg'>"
                        "<script>setInterval(()=>{document.getElementById('frame').src='/frame.jpg?t='+Date.now()},120)</script>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path.startswith("/frame.jpg"):
                    with streamer._lock:
                        jpeg = streamer._jpeg
                    if jpeg is None:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return

                self.send_response(404)
                self.end_headers()

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, image) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self._lock:
            self._jpeg = encoded.tobytes()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    hsv_range = load_hsv(config_path, args.color)

    try:
        import rclpy
        from rclpy.node import Node
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
    except ImportError as exc:
        print(f"ERROR: ROS2/cv_bridge dependencies are unavailable: {exc}", file=sys.stderr)
        return 2

    class VisualizerNode(Node):
        def __init__(self):
            super().__init__("real_flow_visualizer")
            self.bridge = CvBridge()
            self.color_img = None
            self.depth_img = None
            self.cam_k = None
            self.last_image_time = 0.0
            self.topic_state = {
                "/competition/state": "-",
                "/inspection/all": "-",
                "/inspection/target_zones": "-",
                "/vision/detect_request": "-",
                "/vision/grasp_pose": "-",
                "/arm/command": "-",
                "/arm/feedback": "-",
                "/placement/recognized_zone": "-",
                "/task/status": "-",
            }
            self.create_subscription(Image, args.color_topic, self._on_color, 10)
            self.create_subscription(Image, args.depth_topic, self._on_depth, 10)
            self.create_subscription(CameraInfo, args.info_topic, self._on_info, 10)
            for topic in self.topic_state:
                self.create_subscription(String, topic, self._remember(topic), 10)

        def _on_color(self, msg):
            try:
                self.color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.last_image_time = time.monotonic()
            except Exception as exc:
                self.get_logger().warn(f"color image conversion failed: {exc}")

        def _on_depth(self, msg):
            try:
                self.depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            except Exception as exc:
                self.get_logger().warn(f"depth image conversion failed: {exc}")

        def _on_info(self, msg):
            self.cam_k = np.array(msg.k).reshape(3, 3)

        def _remember(self, topic: str):
            def callback(msg):
                self.topic_state[topic] = msg.data

            return callback

    rclpy.init(args=None)
    node = VisualizerNode()
    web_streamer = None
    use_window = not args.web
    if use_window:
        try:
            cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(args.window, 1280, 720)
        except cv2.error as exc:
            print(f"[WARN] OpenCV window unavailable, falling back to web mode: {exc}")
            use_window = False

    if not use_window:
        web_streamer = WebStreamer(args.web_host, args.web_port)
        web_streamer.start()
        print(f"Open this in a browser: http://<jetson-ip>:{args.web_port}")
        print(f"Local URL on Jetson: http://127.0.0.1:{args.web_port}")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if node.color_img is None:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                y = 60
                y = put_line(canvas, "Waiting for camera image...", 40, y, (0, 255, 255))
                y = put_line(canvas, f"color topic: {args.color_topic}", 40, y)
                y = put_line(canvas, f"depth topic: {args.depth_topic}", 40, y)
                y = put_line(canvas, "If this stays blank, start camera node or check ros2 topic list | grep rgbd", 40, y)
                draw_topic_panel(canvas, node.topic_state, 40, y + 20)
            else:
                canvas = render_dashboard(node, args, hsv_range)

            if use_window:
                cv2.imshow(args.window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            elif web_streamer is not None:
                web_streamer.update(canvas)
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        if web_streamer is not None:
            web_streamer.stop()
        if use_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


def render_dashboard(node, args, hsv_range):
    frame = node.color_img.copy()
    mask, objects = detect_color_bars(frame, hsv_range, args.min_area)
    best = objects[0] if objects else None

    for index, obj in enumerate(objects[:5]):
        color = (0, 0, 255) if index == 0 else (0, 180, 255)
        cv2.drawContours(frame, [obj["box"]], 0, color, 2)
        cv2.circle(frame, (obj["cx"], obj["cy"]), 5, (255, 0, 0), -1)
        label = f'{args.color} area={obj["area"]:.0f} ratio={obj["ratio"]:.2f}'
        cv2.putText(frame, label, (obj["cx"] + 8, max(24, obj["cy"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    depth_text = "depth: -"
    if best is not None:
        depth_m = sample_depth(node.depth_img, best["cx"], best["cy"])
        depth_text = "depth: invalid" if depth_m is None else f"depth: {depth_m:.3f}m"

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    panel_w = max(420, w)
    canvas = np.zeros((max(h, 720), w + panel_w, 3), dtype=np.uint8)
    canvas[:h, :w] = frame

    y = 28
    y = put_line(canvas, f"camera: {w}x{h}  objects: {len(objects)}  {depth_text}", 16, y, (0, 255, 255))
    if best is not None:
        y = put_line(
            canvas,
            f'best: cx={best["cx"]} cy={best["cy"]} angle={best["angle"]:.1f} area={best["area"]:.0f}',
            16,
            y,
            (0, 255, 255),
        )
    else:
        y = put_line(canvas, f"no {args.color} bar above area={args.min_area}", 16, y, (0, 180, 255))

    panel_x = w + 16
    next_y = draw_topic_panel(canvas, node.topic_state, panel_x, 28)
    draw_mask_preview(canvas, mask_bgr, panel_x, max(next_y + 20, 330), panel_w - 32)
    put_line(canvas, "Press q or Esc to quit", panel_x, canvas.shape[0] - 24, (180, 180, 180))
    return canvas


def draw_topic_panel(canvas, topic_state, x: int, y: int) -> int:
    y = put_line(canvas, "ROS flow topics", x, y, (0, 255, 255))
    for topic, value in topic_state.items():
        clipped = value if len(value) <= 78 else value[:75] + "..."
        y = put_line(canvas, f"{topic}: {clipped}", x, y)
    return y


def draw_mask_preview(canvas, mask_bgr, x: int, y: int, max_width: int) -> None:
    if y >= canvas.shape[0] - 40:
        return

    h, w = mask_bgr.shape[:2]
    preview_w = max(160, min(max_width, 420))
    preview_h = int(preview_w * h / max(w, 1))
    max_h = canvas.shape[0] - y - 52
    if preview_h > max_h:
        preview_h = max(80, max_h)
        preview_w = int(preview_h * w / max(h, 1))

    preview = cv2.resize(mask_bgr, (preview_w, preview_h))
    put_line(canvas, "HSV mask preview", x, y, (0, 255, 255))
    y0 = y + 12
    x2 = min(x + preview_w, canvas.shape[1])
    y2 = min(y0 + preview_h, canvas.shape[0])
    canvas[y0:y2, x:x2] = preview[: y2 - y0, : x2 - x]
    cv2.rectangle(canvas, (x, y0), (x2 - 1, y2 - 1), (80, 80, 80), 1)


if __name__ == "__main__":
    raise SystemExit(main())
