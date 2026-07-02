#!/usr/bin/env python3
"""Browser MJPEG viewer for ROS2 aligned RGB-D topics.

Run on the Jetson after starting realsense2_camera with aligned depth enabled.
Open http://<jetson-ip>:8080 from another computer on the same network.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream ROS2 aligned depth probe to a browser.")
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw", help="Color Image topic.")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw", help="Aligned depth Image topic.")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info", help="Color CameraInfo topic.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--roi", type=int, default=40, help="Center ROI size in pixels.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100.")
    parser.add_argument("--save-dir", type=Path, default=Path("debug/ros2_aligned_depth_web"), help="Snapshot output directory.")
    parser.add_argument("--save-every", type=int, default=0, help="Save one debug frame every N processed pairs. 0 disables saving.")
    return parser.parse_args()


class SharedFrames:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.overlay_jpg: bytes | None = None
        self.depth_jpg: bytes | None = None
        self.color_jpg: bytes | None = None
        self.stats: dict[str, Any] = {"status": "waiting_for_frames"}

    def update(self, overlay: np.ndarray, depth_vis: np.ndarray, color: np.ndarray, stats: dict[str, Any], quality: int) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, quality)))]
        ok_overlay, overlay_buf = cv2.imencode(".jpg", overlay, encode_params)
        ok_depth, depth_buf = cv2.imencode(".jpg", depth_vis, encode_params)
        ok_color, color_buf = cv2.imencode(".jpg", color, encode_params)
        if not (ok_overlay and ok_depth and ok_color):
            return
        with self.lock:
            self.overlay_jpg = overlay_buf.tobytes()
            self.depth_jpg = depth_buf.tobytes()
            self.color_jpg = color_buf.tobytes()
            self.stats = stats

    def get(self, name: str) -> bytes | None:
        with self.lock:
            if name == "overlay":
                return self.overlay_jpg
            if name == "depth":
                return self.depth_jpg
            if name == "color":
                return self.color_jpg
            return None

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.stats)


def convert_depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "16UC1":
        return depth.astype(np.float32) / 1000.0
    if encoding == "32FC1":
        return depth.astype(np.float32)
    raise ValueError(f"Unsupported depth encoding: {encoding}")


def center_depth_stats(depth_m: np.ndarray, roi_size: int) -> dict[str, Any]:
    h, w = depth_m.shape[:2]
    half = max(1, roi_size // 2)
    cx, cy = w // 2, h // 2
    roi = depth_m[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    valid = valid[(valid >= 0.20) & (valid <= 5.0)]
    if valid.size == 0:
        return {"valid_count": 0, "valid_ratio": 0.0, "min_m": None, "median_m": None, "max_m": None}
    return {
        "valid_count": int(valid.size),
        "valid_ratio": round(float(valid.size / max(1, roi.size)), 4),
        "min_m": round(float(np.min(valid)), 4),
        "median_m": round(float(np.median(valid)), 4),
        "max_m": round(float(np.max(valid)), 4),
    }


def make_depth_vis(depth_m: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, 3.0)
    depth_u8 = (clipped / 3.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def draw_overlay(color_bgr: np.ndarray, roi_size: int, stats: dict[str, Any]) -> np.ndarray:
    overlay = color_bgr.copy()
    h, w = overlay.shape[:2]
    half = max(1, roi_size // 2)
    cx, cy = w // 2, h // 2
    cv2.rectangle(
        overlay,
        (max(0, cx - half), max(0, cy - half)),
        (min(w - 1, cx + half), min(h - 1, cy + half)),
        (0, 255, 255),
        2,
    )
    median = stats.get("median_m")
    depth_text = "center depth: invalid" if median is None else f"center depth: {median:.3f} m"
    lines = [
        depth_text,
        f"valid_ratio: {stats.get('valid_ratio', 0.0):.2f}",
        "Open /depth.mjpg for depth view",
    ]
    for idx, line in enumerate(lines):
        y = 32 + idx * 30
        cv2.putText(overlay, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return overlay


class Ros2AlignedDepthWeb(Node):
    def __init__(self, args: argparse.Namespace, shared: SharedFrames) -> None:
        super().__init__("ros2_aligned_depth_web")
        self.args = args
        self.shared = shared
        self.bridge = CvBridge()
        self.latest_color: Image | None = None
        self.latest_depth: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.frame_count = 0
        self.last_print = 0.0
        self.args.save_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(Image, args.rgb_topic, self.on_color, 10)
        self.create_subscription(Image, args.depth_topic, self.on_depth, 10)
        self.create_subscription(CameraInfo, args.info_topic, self.on_info, 10)
        self.create_timer(0.03, self.process_latest)

        self.get_logger().info(f"RGB topic: {args.rgb_topic}")
        self.get_logger().info(f"Depth topic: {args.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {args.info_topic}")

    def on_color(self, msg: Image) -> None:
        self.latest_color = msg

    def on_depth(self, msg: Image) -> None:
        self.latest_depth = msg

    def on_info(self, msg: CameraInfo) -> None:
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(
                "CameraInfo K="
                + json.dumps(
                    {
                        "fx": msg.k[0],
                        "fy": msg.k[4],
                        "cx": msg.k[2],
                        "cy": msg.k[5],
                        "width": msg.width,
                        "height": msg.height,
                    },
                    ensure_ascii=False,
                )
            )

    def process_latest(self) -> None:
        if self.latest_color is None or self.latest_depth is None:
            return
        color_msg = self.latest_color
        depth_msg = self.latest_depth
        self.latest_color = None
        self.latest_depth = None

        try:
            color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_m = convert_depth_to_meters(depth, depth_msg.encoding)
        except Exception as exc:  # Keep stream alive during driver changes.
            self.shared.stats = {"status": "frame_convert_error", "error": str(exc)}
            self.get_logger().warn(str(exc))
            return

        center_stats = center_depth_stats(depth_m, self.args.roi)
        aligned = color_bgr.shape[:2] == depth_m.shape[:2]
        stats: dict[str, Any] = {
            "status": "ok",
            "frame": self.frame_count,
            "timestamp": time.time(),
            "rgb_shape": list(color_bgr.shape),
            "depth_shape": list(depth_m.shape),
            "depth_encoding": depth_msg.encoding,
            "aligned_shape_match": aligned,
            "center_roi": center_stats,
        }
        if self.camera_info is not None:
            stats["camera_info"] = {
                "fx": self.camera_info.k[0],
                "fy": self.camera_info.k[4],
                "cx": self.camera_info.k[2],
                "cy": self.camera_info.k[5],
            }

        overlay = draw_overlay(color_bgr, self.args.roi, center_stats)
        depth_color = make_depth_vis(depth_m)
        self.shared.update(overlay, depth_color, color_bgr, stats, self.args.jpeg_quality)

        now = time.time()
        if now - self.last_print >= 1.0:
            print(json.dumps(stats, ensure_ascii=False))
            self.last_print = now

        if self.args.save_every > 0 and self.frame_count % self.args.save_every == 0:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(self.args.save_dir / f"color_{stamp}_{self.frame_count:06d}.jpg"), color_bgr)
            cv2.imwrite(str(self.args.save_dir / f"aligned_depth_vis_{stamp}_{self.frame_count:06d}.jpg"), depth_color)
            cv2.imwrite(str(self.args.save_dir / f"overlay_{stamp}_{self.frame_count:06d}.jpg"), overlay)

        self.frame_count += 1


def make_handler(shared: SharedFrames):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"""<!doctype html>
<html><head><meta charset="utf-8"><title>RealSense Aligned Depth Probe</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:20px}img{max-width:48%;border:1px solid #444;margin-right:1%;vertical-align:top}pre{background:#222;padding:12px;white-space:pre-wrap}</style>
</head><body>
<h2>RealSense Aligned Depth Probe</h2>
<p>Overlay checks the center ROI depth. Put the cone at 0.5m, 1.0m, 1.5m and read median_m.</p>
<img src="/overlay.mjpg"><img src="/depth.mjpg">
<h3>Status</h3><pre id="stats">loading...</pre>
<script>
async function tick(){const r=await fetch('/stats.json');document.getElementById('stats').textContent=JSON.stringify(await r.json(), null, 2);}
setInterval(tick,1000);tick();
</script></body></html>"""
                )
                return

            if self.path == "/stats.json":
                payload = json.dumps(shared.get_stats(), ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            stream_name = None
            if self.path == "/overlay.mjpg":
                stream_name = "overlay"
            elif self.path == "/depth.mjpg":
                stream_name = "depth"
            elif self.path == "/color.mjpg":
                stream_name = "color"

            if stream_name is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = shared.get(stream_name)
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


def start_http_server(host: str, port: int, shared: SharedFrames) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(shared))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    args = parse_args()
    shared = SharedFrames()
    server = start_http_server(args.host, args.port, shared)
    print(f"Open http://<jetson-ip>:{args.port}/ in your browser.")

    rclpy.init()
    node = Ros2AlignedDepthWeb(args, shared)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
