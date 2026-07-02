#!/usr/bin/env python3
"""Browser probe for RealSense D435i V4L2 color/depth devices.

This is a fallback when pyrealsense2 and realsense-ros are not installed.
It streams /dev/video4 color and /dev/video0 depth-like frames to a browser.

Important: this does NOT align depth to color. Use it only to verify that the
Jetson can see both streams. For bbox + depth localization, use aligned depth
from pyrealsense2 or realsense-ros later.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream D435i V4L2 color/depth preview to browser.")
    parser.add_argument("--color", default="/dev/video4", help="Color V4L2 device.")
    parser.add_argument("--depth", default="/dev/video0", help="Depth V4L2 device.")
    parser.add_argument("--width", type=int, default=640, help="Requested stream width.")
    parser.add_argument("--height", type=int, default=480, help="Requested stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested stream fps.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality.")
    return parser.parse_args()


class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.color_jpg: bytes | None = None
        self.depth_jpg: bytes | None = None
        self.stats: dict[str, Any] = {"status": "starting"}

    def update(self, color: np.ndarray | None, depth_vis: np.ndarray | None, stats: dict[str, Any], quality: int) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, quality)))]
        color_jpg = None
        depth_jpg = None
        if color is not None:
            ok, buf = cv2.imencode(".jpg", color, params)
            if ok:
                color_jpg = buf.tobytes()
        if depth_vis is not None:
            ok, buf = cv2.imencode(".jpg", depth_vis, params)
            if ok:
                depth_jpg = buf.tobytes()
        with self.lock:
            if color_jpg is not None:
                self.color_jpg = color_jpg
            if depth_jpg is not None:
                self.depth_jpg = depth_jpg
            self.stats = stats

    def get_jpg(self, name: str) -> bytes | None:
        with self.lock:
            return self.color_jpg if name == "color" else self.depth_jpg

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.stats)


def open_capture(path: str, fourcc: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return cap


def depth_to_vis(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    stats: dict[str, Any] = {
        "depth_shape": list(frame.shape),
        "depth_dtype": str(frame.dtype),
        "note": "V4L2 depth preview is not aligned to color.",
    }
    if frame.ndim == 2 and frame.dtype == np.uint16:
        depth_m = frame.astype(np.float32) / 1000.0
        valid = depth_m[(depth_m > 0.2) & (depth_m < 5.0)]
        if valid.size:
            stats["median_m"] = round(float(np.median(valid)), 4)
            stats["valid_ratio"] = round(float(valid.size / frame.size), 4)
        vis = np.clip(depth_m, 0.0, 3.0)
        vis_u8 = (vis / 3.0 * 255.0).astype(np.uint8)
        return cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET), stats

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stats["warning"] = "Depth was not received as uint16 Z16 by OpenCV; this is only a visual preview."
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET), stats


def capture_loop(args: argparse.Namespace, shared: Shared) -> None:
    color_cap = open_capture(args.color, "YUYV", args.width, args.height, args.fps)
    depth_cap = open_capture(args.depth, "Z16 ", args.width, args.height, args.fps)

    while True:
        color_ok, color = color_cap.read()
        depth_ok, depth = depth_cap.read()
        stats: dict[str, Any] = {
            "status": "ok" if color_ok or depth_ok else "no_frames",
            "color_device": args.color,
            "depth_device": args.depth,
            "timestamp": time.time(),
            "aligned": False,
            "aligned_warning": "This V4L2 preview does not provide aligned_depth_to_color.",
        }

        if color_ok and color is not None:
            stats["color_shape"] = list(color.shape)
            cv2.putText(
                color,
                "RGB /dev/video4 - NOT depth-aligned",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            color = None
            stats["color_error"] = "failed_to_read_color"

        if depth_ok and depth is not None:
            depth_vis, depth_stats = depth_to_vis(depth)
            stats.update(depth_stats)
            cv2.putText(
                depth_vis,
                "Depth /dev/video0 - NOT aligned",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            depth_vis = None
            stats["depth_error"] = "failed_to_read_depth"

        shared.update(color, depth_vis, stats, args.jpeg_quality)
        time.sleep(0.02)


def make_handler(shared: Shared):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                html = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>D435i V4L2 Probe</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}img{max-width:48%;border:1px solid #444;margin-right:1%;vertical-align:top}pre{background:#222;padding:12px;white-space:pre-wrap}</style>
</head><body>
<h2>D435i V4L2 Probe</h2>
<p>This verifies /dev/video4 RGB and /dev/video0 depth. It is <b>not</b> aligned depth.</p>
<img src="/color.mjpg"><img src="/depth.mjpg">
<h3>Status</h3><pre id="stats">loading...</pre>
<script>
async function tick(){const r=await fetch('/stats.json');document.getElementById('stats').textContent=JSON.stringify(await r.json(), null, 2);}
setInterval(tick,1000);tick();
</script></body></html>"""
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/stats.json":
                payload = json.dumps(shared.get_stats(), ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            name = None
            if self.path == "/color.mjpg":
                name = "color"
            elif self.path == "/depth.mjpg":
                name = "depth"
            if name is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = shared.get_jpg(name)
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def main() -> None:
    args = parse_args()
    shared = Shared()
    threading.Thread(target=capture_loop, args=(args, shared), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    print(f"Open http://<jetson-ip>:{args.port}/ in your browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
