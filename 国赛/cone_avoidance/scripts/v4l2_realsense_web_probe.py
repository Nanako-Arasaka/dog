#!/usr/bin/env python3
"""Browser probe for RealSense D435i V4L2 color/depth devices.

This is a fallback when pyrealsense2 and realsense-ros are not installed.
It streams RealSense RGB and Z16 depth-like frames to a browser.

Important: this does NOT align depth to color. Use it only to verify that the
Jetson can see both streams. For bbox + depth localization, use aligned depth
from pyrealsense2 or realsense-ros later.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream D435i V4L2 color/depth preview to browser.")
    parser.add_argument("--color", default="/dev/video6", help="RealSense RGB V4L2 device.")
    parser.add_argument("--depth", default="/dev/video2", help="RealSense Z16 depth V4L2 device.")
    parser.add_argument("--width", type=int, default=640, help="Requested stream width.")
    parser.add_argument("--height", type=int, default=480, help="Requested stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested stream fps.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality.")
    parser.add_argument("--depth-max-m", type=float, default=0.0, help="Fixed depth visualization max in meters. 0 uses dynamic min/max.")
    parser.add_argument("--no-auto-depth-mode", action="store_true", help="Disable automatic depth mode probing.")
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


def opencv_source(device: str) -> str | int:
    match = re.fullmatch(r"/dev/video(\d+)", device)
    if match:
        return int(match.group(1))
    return device


def open_capture(path: str, fourcc: str | None, width: int, height: int, fps: int, convert_rgb: bool = True) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(opencv_source(path), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1 if convert_rgb else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    return cap


def capture_mode_summary(cap: cv2.VideoCapture) -> dict[str, Any]:
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))
    return {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc,
    }


def read_one_frame(cap: cv2.VideoCapture, tries: int = 5) -> tuple[bool, np.ndarray | None]:
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.05)
    return False, None


def open_depth_capture(args: argparse.Namespace, shared: Shared | None = None) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    default_cap = open_capture(args.depth, "Z16 ", args.width, args.height, args.fps, convert_rgb=False)
    ok, frame = read_one_frame(default_cap, tries=5)
    if ok and frame is not None:
        info = {
            "depth_open_mode": "requested_default",
            "requested": {"width": args.width, "height": args.height, "fps": args.fps, "fourcc": "Z16 ", "convert_rgb": False},
            "actual": capture_mode_summary(default_cap),
            "first_frame_shape": list(frame.shape),
            "first_frame_dtype": str(frame.dtype),
        }
        return default_cap, info

    default_cap.release()
    if args.no_auto_depth_mode:
        info = {
            "depth_open_mode": "default_failed_auto_disabled",
            "requested": {"width": args.width, "height": args.height, "fps": args.fps, "fourcc": "Z16 ", "convert_rgb": False},
        }
        return open_capture(args.depth, "Z16 ", args.width, args.height, args.fps, convert_rgb=False), info

    sizes = [(args.width, args.height), (640, 480), (848, 480), (424, 240)]
    fps_values = [args.fps, 30, 15]
    fourcc_values: list[str | None] = ["Z16 ", "Y16 ", None]
    convert_values = [False, True]
    tried: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str | None, bool]] = set()

    print(f"Default depth mode failed for {args.depth}; probing V4L2 depth modes...")
    for width, height in sizes:
        for fps in fps_values:
            for fourcc in fourcc_values:
                for convert_rgb in convert_values:
                    key = (width, height, fps, fourcc, convert_rgb)
                    if key in seen:
                        continue
                    seen.add(key)
                    if shared is not None:
                        shared.stats = {
                            "status": "probing_depth_mode",
                            "color_device": args.color,
                            "depth_device": args.depth,
                            "trying": {"width": width, "height": height, "fps": fps, "fourcc": fourcc, "convert_rgb": convert_rgb},
                            "note": "Color preview will start after this short depth probe. If this takes too long, restart with --no-auto-depth-mode.",
                        }
                    cap = open_capture(args.depth, fourcc, width, height, fps, convert_rgb=convert_rgb)
                    ok, frame = read_one_frame(cap, tries=3)
                    actual = capture_mode_summary(cap)
                    entry = {
                        "requested": {"width": width, "height": height, "fps": fps, "fourcc": fourcc, "convert_rgb": convert_rgb},
                        "opened": cap.isOpened(),
                        "ok": ok,
                        "actual": actual,
                    }
                    if ok and frame is not None:
                        entry["first_frame_shape"] = list(frame.shape)
                        entry["first_frame_dtype"] = str(frame.dtype)
                        print(f"Found readable depth mode: {entry}")
                        return cap, {"depth_open_mode": "auto_probe_success", "selected": entry, "tried_count": len(tried) + 1}
                    cap.release()
                    tried.append(entry)

    print("No readable OpenCV/V4L2 depth mode found.")
    cap = open_capture(args.depth, "Z16 ", args.width, args.height, args.fps, convert_rgb=False)
    return cap, {"depth_open_mode": "auto_probe_failed", "tried_count": len(tried), "tried_preview": tried[:10]}


def depth_m_to_vis(depth_m: np.ndarray, stats: dict[str, Any], fixed_max_m: float) -> np.ndarray:
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
    if finite.size == 0:
        return np.zeros(depth_m.shape[:2], dtype=np.uint8)

    if fixed_max_m > 0.0:
        lo = 0.0
        hi = fixed_max_m
        mode = f"fixed_0_{fixed_max_m:.2f}m"
    else:
        lo = float(np.percentile(finite, 2))
        hi = float(np.percentile(finite, 98))
        if hi <= lo:
            hi = lo + 0.001
        mode = "dynamic_p02_p98"

    stats["vis_min_m"] = round(lo, 4)
    stats["vis_max_m"] = round(hi, 4)
    stats["vis_mode"] = mode
    normalized = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    vis_u8 = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)


def make_message_image(width: int, height: int, lines: list[str]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    y = 48
    for line in lines:
        cv2.putText(
            image,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return image


def add_depth_stats(depth_m: np.ndarray, depth_units: np.ndarray, stats: dict[str, Any]) -> None:
    valid_m = depth_m[(depth_m > 0.2) & (depth_m < 5.0)]
    nonzero_units = depth_units[depth_units > 0]
    if nonzero_units.size:
        stats["raw_min_units"] = int(np.min(nonzero_units))
        stats["raw_median_units"] = int(np.median(nonzero_units))
        stats["raw_max_units"] = int(np.max(nonzero_units))
    if valid_m.size:
        stats["median_m"] = round(float(np.median(valid_m)), 4)
        stats["valid_ratio"] = round(float(valid_m.size / depth_m.size), 4)
        stats["min_m"] = round(float(np.min(valid_m)), 4)
        stats["max_m"] = round(float(np.max(valid_m)), 4)
    else:
        stats["valid_ratio"] = 0.0


def depth_to_vis(frame: np.ndarray, fixed_max_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    stats: dict[str, Any] = {
        "depth_shape": list(frame.shape),
        "depth_dtype": str(frame.dtype),
        "note": "V4L2 depth preview is not aligned to color.",
    }
    if frame.ndim == 3 and frame.shape[2] == 2 and frame.dtype == np.uint8:
        depth_u16 = frame.view(np.uint16).reshape(frame.shape[0], frame.shape[1])
        depth_m = depth_u16.astype(np.float32) / 1000.0
        stats["decoded_as"] = "two_uint8_channels_to_z16"
        add_depth_stats(depth_m, depth_u16, stats)
        return depth_m_to_vis(depth_m, stats, fixed_max_m), stats

    if frame.ndim == 2 and frame.dtype == np.uint16:
        depth_m = frame.astype(np.float32) / 1000.0
        stats["decoded_as"] = "uint16_z16"
        add_depth_stats(depth_m, frame, stats)
        return depth_m_to_vis(depth_m, stats, fixed_max_m), stats

    if frame.ndim == 3:
        if frame.shape[2] in (3, 4):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.normalize(frame[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stats["warning"] = "Depth was not received as uint16 Z16 by OpenCV; this is only a visual preview."
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET), stats


def capture_loop(args: argparse.Namespace, shared: Shared) -> None:
    color_cap = open_capture(args.color, "YUYV", args.width, args.height, args.fps, convert_rgb=True)
    shared.stats = {"status": "opening_color", "color_device": args.color, "depth_device": args.depth}
    color_ok_first, color_first = read_one_frame(color_cap, tries=5)
    if color_ok_first and color_first is not None:
        color_first = color_first.copy()
        cv2.putText(
            color_first,
            f"RGB {args.color} - probing depth...",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        shared.update(
            color_first,
            None,
            {"status": "probing_depth_mode", "color_device": args.color, "depth_device": args.depth, "color_shape": list(color_first.shape)},
            args.jpeg_quality,
        )
    depth_cap, depth_open_info = open_depth_capture(args, shared)
    depth_unavailable = depth_open_info.get("depth_open_mode") == "auto_probe_failed"
    if depth_unavailable:
        placeholder = make_message_image(
            args.width,
            args.height,
            [
                "Depth unavailable",
                f"device: {args.depth}",
                "OpenCV/V4L2 cannot read Z16 depth",
                "Use pyrealsense2 or realsense-ros",
                "for aligned_depth_to_color",
            ],
        )
        shared.update(
            color_first if color_ok_first and color_first is not None else None,
            placeholder,
            {
                "status": "depth_unavailable",
                "color_device": args.color,
                "depth_device": args.depth,
                "depth_open_info": depth_open_info,
                "next_step": "Install/use pyrealsense2 or realsense-ros for aligned depth.",
            },
            args.jpeg_quality,
        )
    frame_id = 0
    last_depth: np.ndarray | None = None

    while True:
        color_ok, color = color_cap.read()
        depth_ok, depth = depth_cap.read()
        stats: dict[str, Any] = {
            "status": "ok" if color_ok or depth_ok else "no_frames",
            "color_device": args.color,
            "depth_device": args.depth,
            "timestamp": time.time(),
            "frame_id": frame_id,
            "aligned": False,
            "aligned_warning": "This V4L2 preview does not provide aligned_depth_to_color.",
            "depth_open_info": depth_open_info,
        }

        if color_ok and color is not None:
            stats["color_shape"] = list(color.shape)
            cv2.putText(
                color,
                f"RGB {args.color} - NOT depth-aligned",
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
            depth_changed = True
            depth_delta_mean = None
            if last_depth is not None and last_depth.shape == depth.shape:
                delta = cv2.absdiff(depth, last_depth)
                depth_delta_mean = float(np.mean(delta))
                depth_changed = depth_delta_mean > 0.01
            last_depth = depth.copy()
            depth_vis, depth_stats = depth_to_vis(depth, args.depth_max_m)
            stats.update(depth_stats)
            stats["depth_changed"] = depth_changed
            stats["depth_delta_mean"] = depth_delta_mean
            cv2.putText(
                depth_vis,
                f"Depth {args.depth} - NOT aligned",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            depth_vis = make_message_image(
                args.width,
                args.height,
                [
                    "Depth read failed",
                    f"device: {args.depth}",
                    "No Z16 frame from OpenCV/V4L2",
                    "RGB preview is still valid",
                    "Aligned depth needs RealSense SDK/ROS",
                ],
            )
            stats["depth_error"] = "failed_to_read_depth"

        shared.update(color, depth_vis, stats, args.jpeg_quality)
        frame_id += 1
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
<p>This verifies RealSense RGB and Z16 depth V4L2 streams. It is <b>not</b> aligned depth.</p>
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
