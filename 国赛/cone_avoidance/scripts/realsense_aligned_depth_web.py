#!/usr/bin/env python3
"""Browser MJPEG viewer for RealSense aligned depth + YOLO cone localization."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL = Path(__file__).resolve().with_name("cone_yolo_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream pyrealsense2 aligned depth + YOLO cone localization to a browser.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--roi", type=int, default=40, help="Center ROI size in pixels.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO model path.")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default="", help="YOLO device: '', 'cpu', '0'.")
    parser.add_argument("--output-class-name", default="cone", help="Class name written to JSON output. Empty keeps model names.")
    parser.add_argument("--min-depth", type=float, default=0.20, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=5.00, help="Maximum valid depth in meters.")
    parser.add_argument("--min-valid-ratio", type=float, default=0.08, help="Minimum valid depth ratio inside bbox ROI.")
    parser.add_argument("--control-jsonl", action="store_true", help="Print compact control JSONL to stdout for main_avoidance_run.py.")
    parser.add_argument("--control-rate-hz", type=float, default=10.0, help="Maximum control JSONL output rate. <=0 prints every frame.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality.")
    parser.add_argument("--save-dir", type=Path, default=Path("debug/realsense_aligned_depth_web"), help="Snapshot output directory.")
    parser.add_argument("--save-every", type=int, default=0, help="Save one debug frame every N processed pairs. 0 disables saving.")
    return parser.parse_args()


class SharedFrames:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.overlay_jpg: bytes | None = None
        self.depth_jpg: bytes | None = None
        self.stats: dict[str, Any] = {"status": "starting"}
        self.motion: dict[str, Any] = {"status": "waiting_for_control"}

    def update(self, overlay: np.ndarray, depth_vis: np.ndarray, stats: dict[str, Any], quality: int) -> None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, quality)))]
        ok_overlay, overlay_buf = cv2.imencode(".jpg", overlay, params)
        ok_depth, depth_buf = cv2.imencode(".jpg", depth_vis, params)
        if not (ok_overlay and ok_depth):
            return
        with self.lock:
            self.overlay_jpg = overlay_buf.tobytes()
            self.depth_jpg = depth_buf.tobytes()
            self.stats = stats

    def get_jpg(self, name: str) -> bytes | None:
        with self.lock:
            if name == "overlay":
                return self.overlay_jpg
            if name == "depth":
                return self.depth_jpg
            return None

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.stats)

    def update_motion(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.motion = dict(payload)

    def get_motion(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.motion)


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


def depth_stats_for_roi(depth_m: np.ndarray, roi: tuple[int, int, int, int], min_depth: float, max_depth: float) -> dict[str, Any]:
    x1, y1, x2, y2 = roi
    h, w = depth_m.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return {"valid": False, "valid_count": 0, "valid_ratio": 0.0, "median_m": None}

    patch = depth_m[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    valid = valid[(valid >= min_depth) & (valid <= max_depth)]
    if valid.size == 0:
        return {"valid": False, "valid_count": 0, "valid_ratio": 0.0, "median_m": None}
    return {
        "valid": True,
        "valid_count": int(valid.size),
        "valid_ratio": round(float(valid.size / max(1, patch.size)), 4),
        "min_m": round(float(np.min(valid)), 4),
        "median_m": round(float(np.median(valid)), 4),
        "max_m": round(float(np.max(valid)), 4),
    }


def camera_fps(frame_times: deque[float]) -> float | None:
    if len(frame_times) < 2:
        return None
    elapsed = frame_times[-1] - frame_times[0]
    if elapsed <= 0.0:
        return None
    return round(float((len(frame_times) - 1) / elapsed), 2)


def make_control_payload(stats: dict[str, Any], fps: float | None) -> dict[str, Any]:
    center_roi = stats.get("center_roi") if isinstance(stats.get("center_roi"), dict) else {}
    front_depth = center_roi.get("min_m")
    if front_depth is None:
        front_depth = center_roi.get("median_m")

    obstacles = []
    timestamp = stats.get("timestamp", time.time())
    for obj in stats.get("obstacles", []):
        if not isinstance(obj, dict):
            continue
        obstacles.append(
            {
                "x": obj.get("x"),
                "z": obj.get("z"),
                "conf": obj.get("conf", obj.get("yolo_conf", 0.0)),
                "bbox": obj.get("bbox", []),
                "age": obj.get("age", 0),
                "last_seen": timestamp,
            }
        )

    payload: dict[str, Any] = {
        "status": stats.get("status", "unknown"),
        "frame": stats.get("frame"),
        "timestamp": timestamp,
        "obstacles": obstacles,
        "front_depth": front_depth,
        "depth_valid_ratio": center_roi.get("valid_ratio"),
        "aligned_depth_ok": bool(stats.get("aligned", False)),
        "realsense_ok": stats.get("status") == "ok",
        "source": "realsense_aligned_depth_web",
    }
    if fps is not None:
        payload["realsense_fps"] = fps
    return payload


def emit_control_payload(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.control_jsonl:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def bbox_depth_roi(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    # Cone tip and bbox edges are noisy; use the lower-middle body/bottom area.
    return (
        int(x1 + 0.25 * bw),
        int(y1 + 0.45 * bh),
        int(x1 + 0.75 * bw),
        int(y1 + 0.90 * bh),
    )


def localize_bbox(
    bbox: tuple[int, int, int, int],
    yolo_conf: float,
    class_id: int,
    class_name: str,
    depth_m: np.ndarray,
    intr: dict[str, float],
    min_depth: float,
    max_depth: float,
    min_valid_ratio: float,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    roi = bbox_depth_roi(bbox)
    depth_stats = depth_stats_for_roi(depth_m, roi, min_depth, max_depth)
    u = 0.5 * (x1 + x2)
    z = depth_stats.get("median_m")
    if z is None or depth_stats.get("valid_ratio", 0.0) < min_valid_ratio:
        return {
            "class_id": class_id,
            "class_name": class_name,
            "bbox": [x1, y1, x2, y2],
            "depth_roi": list(roi),
            "yolo_conf": round(float(yolo_conf), 4),
            "conf": 0.0,
            "x": None,
            "z": None,
            "depth": depth_stats,
            "valid": False,
        }

    # Camera projection gives x positive to image right. Project convention for
    # control is x > 0 on robot-left, so invert the sign.
    x_left_positive = -((u - intr["cx"]) * float(z) / intr["fx"])
    fused_conf = round(float(yolo_conf) * min(1.0, float(depth_stats["valid_ratio"]) / max(min_valid_ratio, 1e-6)), 4)
    return {
        "class_id": class_id,
        "class_name": class_name,
        "bbox": [x1, y1, x2, y2],
        "depth_roi": list(roi),
        "yolo_conf": round(float(yolo_conf), 4),
        "conf": fused_conf,
        "x": round(float(x_left_positive), 4),
        "z": round(float(z), 4),
        "depth": depth_stats,
        "valid": True,
    }


def make_depth_vis(depth_m: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, 3.0)
    depth_u8 = (clipped / 3.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def draw_overlay(
    color_bgr: np.ndarray,
    roi_size: int,
    center_stats: dict[str, Any],
    intr: dict[str, float],
    obstacles: list[dict[str, Any]],
) -> np.ndarray:
    overlay = color_bgr.copy()
    h, w = overlay.shape[:2]
    half = max(1, roi_size // 2)
    cx_pix, cy_pix = w // 2, h // 2
    x1, y1 = max(0, cx_pix - half), max(0, cy_pix - half)
    x2, y2 = min(w - 1, cx_pix + half), min(h - 1, cy_pix + half)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    median = center_stats.get("median_m")
    if median is None:
        depth_line = "center depth: invalid"
        x_line = "center x: invalid"
    else:
        x_m = (cx_pix - intr["cx"]) * float(median) / intr["fx"]
        depth_line = f"center depth: {float(median):.3f} m"
        x_line = f"center x: {x_m:.3f} m"

    lines = [
        "YOLO cone + aligned depth",
        depth_line,
        x_line,
        f"valid_ratio: {center_stats.get('valid_ratio', 0.0):.2f}",
        f"cones: {len([obj for obj in obstacles if obj.get('valid')])}/{len(obstacles)}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(overlay, line, (16, 32 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    for obj in obstacles:
        bx1, by1, bx2, by2 = obj["bbox"]
        rx1, ry1, rx2, ry2 = obj["depth_roi"]
        valid = bool(obj.get("valid"))
        color = (0, 220, 0) if valid else (0, 0, 255)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 2)
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        if valid:
            label = f"cone x={obj['x']:.2f} z={obj['z']:.2f} c={obj['conf']:.2f}"
        else:
            label = f"cone depth invalid yc={obj['yolo_conf']:.2f}"
        cv2.putText(overlay, label, (bx1, max(24, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return overlay


def realsense_loop(args: argparse.Namespace, shared: SharedFrames) -> None:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        shared.stats = {"status": "missing_pyrealsense2", "error": str(exc), "aligned": False}
        emit_control_payload(args, make_control_payload(shared.stats, None))
        return
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        shared.stats = {"status": "missing_ultralytics", "error": str(exc), "hint": "pip install ultralytics", "aligned": False}
        emit_control_payload(args, make_control_payload(shared.stats, None))
        return

    model_path = args.model.resolve()
    if not model_path.exists():
        shared.stats = {"status": "missing_model", "model": str(model_path), "aligned": False}
        emit_control_payload(args, make_control_payload(shared.stats, None))
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        shared.stats = {"status": "model_load_failed", "model": str(model_path), "error": str(exc), "aligned": False}
        emit_control_payload(args, make_control_payload(shared.stats, None))
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        shared.stats = {"status": "pipeline_start_failed", "error": str(exc), "aligned": False}
        emit_control_payload(args, make_control_payload(shared.stats, None))
        return

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    intr = {"fx": color_intr.fx, "fy": color_intr.fy, "cx": color_intr.ppx, "cy": color_intr.ppy}
    frame_id = 0
    last_print = 0.0
    last_control_print = 0.0
    frame_times: deque[float] = deque(maxlen=max(5, int(args.fps * 2)))

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                shared.stats = {"status": "missing_frame", "frame": frame_id, "aligned": False}
                emit_control_payload(args, make_control_payload(shared.stats, camera_fps(frame_times)))
                continue

            now = time.time()
            frame_times.append(now)
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            center_stats = center_depth_stats(depth_m, args.roi)
            obstacles: list[dict[str, Any]] = []
            try:
                result = model.predict(
                    color_bgr,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.yolo_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]
                names = result.names
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                        bbox = tuple(int(round(v)) for v in xyxy)
                        class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
                        yolo_conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
                        model_class_name = str(names.get(class_id, "cone")) if isinstance(names, dict) else "cone"
                        class_name = args.output_class_name.strip() or model_class_name
                        obstacles.append(
                            localize_bbox(
                                bbox=bbox,
                                yolo_conf=yolo_conf,
                                class_id=class_id,
                                class_name=class_name,
                                depth_m=depth_m,
                                intr=intr,
                                min_depth=args.min_depth,
                                max_depth=args.max_depth,
                                min_valid_ratio=args.min_valid_ratio,
                            )
                        )
                        obstacles[-1]["model_class_name"] = model_class_name
            except Exception as exc:
                shared.stats = {"status": "yolo_inference_failed", "error": str(exc), "aligned": True}
                emit_control_payload(args, make_control_payload(shared.stats, camera_fps(frame_times)))
                continue

            obstacles.sort(key=lambda obj: obj["z"] if obj.get("z") is not None else 999.0)
            valid_obstacles = [obj for obj in obstacles if obj.get("valid")]
            overlay = draw_overlay(color_bgr, args.roi, center_stats, intr, obstacles)
            depth_vis = make_depth_vis(depth_m)
            stats: dict[str, Any] = {
                "status": "ok",
                "frame": frame_id,
                "timestamp": time.time(),
                "aligned": True,
                "color_shape": list(color_bgr.shape),
                "depth_shape": list(depth_m.shape),
                "depth_scale": depth_scale,
                "center_roi": center_stats,
                "intrinsics": intr,
                "model": str(model_path),
                "obstacles": valid_obstacles,
                "raw_detections": obstacles,
            }

            shared.update(overlay, depth_vis, stats, args.jpeg_quality)

            fps = camera_fps(frame_times)
            control_interval = 0.0 if args.control_rate_hz <= 0.0 else 1.0 / args.control_rate_hz
            if args.control_jsonl and now - last_control_print >= control_interval:
                emit_control_payload(args, make_control_payload(stats, fps))
                last_control_print = now

            if (not args.control_jsonl) and now - last_print >= 1.0:
                print(json.dumps(stats, ensure_ascii=False))
                last_print = now

            if args.save_every > 0 and frame_id % args.save_every == 0:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(args.save_dir / f"overlay_{stamp}_{frame_id:06d}.jpg"), overlay)
                cv2.imwrite(str(args.save_dir / f"aligned_depth_vis_{stamp}_{frame_id:06d}.jpg"), depth_vis)

            frame_id += 1
    finally:
        pipeline.stop()


def make_handler(shared: SharedFrames):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, payload_obj: dict[str, Any]) -> None:
            payload = json.dumps(payload_obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                payload = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>YOLO Cone + RealSense Aligned Depth</title>
<style>
body{font-family:sans-serif;background:#111;color:#eee;margin:20px}
img{max-width:48%;border:1px solid #444;margin-right:1%;vertical-align:top}
pre{background:#222;padding:12px;white-space:pre-wrap}
.motion{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.box{background:#1e1e1e;border:1px solid #444;padding:12px}
.label{color:#aaa;font-size:13px}
.value{font-size:28px;font-weight:700;margin-top:4px}
.stop{color:#ff6666}.go{color:#66ff99}.turn{color:#ffd966}
</style>
</head><body>
<h2>YOLO Cone + RealSense Aligned Depth</h2>
<p>This uses YOLO bbox + pyrealsense2 rs.align(rs.stream.color) to output cone x/z.</p>
<img src="/overlay.mjpg"><img src="/depth.mjpg">
<h3>Motion Command</h3>
<div class="motion">
  <div class="box"><div class="label">state</div><div id="motion-state" class="value">waiting</div></div>
  <div class="box"><div class="label">reason</div><div id="motion-reason" class="value">waiting</div></div>
  <div class="box"><div class="label">vx</div><div id="motion-vx" class="value">0.00</div></div>
  <div class="box"><div class="label">wz</div><div id="motion-wz" class="value">0.00</div></div>
</div>
<pre id="motion-json">waiting...</pre>
<h3>Status</h3><pre id="stats">loading...</pre>
<script>
function fmt(v){return (typeof v === 'number') ? v.toFixed(3) : String(v ?? 'null');}
function setMotion(m){
  const state = m.state ?? m.status ?? 'waiting';
  const reason = m.reason ?? 'waiting';
  const vx = Number(m.vx ?? 0);
  const wz = Number(m.wz ?? 0);
  document.getElementById('motion-state').textContent = state;
  document.getElementById('motion-reason').textContent = reason;
  document.getElementById('motion-vx').textContent = fmt(vx);
  document.getElementById('motion-wz').textContent = fmt(wz);
  document.getElementById('motion-json').textContent = JSON.stringify(m, null, 2);
  const cls = (Math.abs(vx) < 0.001 && Math.abs(wz) < 0.001) ? 'stop' : (Math.abs(wz) > 0.001 ? 'turn' : 'go');
  for (const id of ['motion-state','motion-reason','motion-vx','motion-wz']) {
    document.getElementById(id).className = 'value ' + cls;
  }
}
async function tick(){
  const r=await fetch('/stats.json');
  document.getElementById('stats').textContent=JSON.stringify(await r.json(), null, 2);
  const m=await fetch('/motion.json');
  setMotion(await m.json());
}
setInterval(tick,1000);tick();
</script></body></html>"""
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path == "/stats.json":
                self._send_json(shared.get_stats())
                return

            if self.path == "/motion.json":
                self._send_json(shared.get_motion())
                return

            name = None
            if self.path == "/overlay.mjpg":
                name = "overlay"
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

        def do_POST(self) -> None:
            if self.path != "/motion.json":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("motion payload must be a JSON object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            shared.update_motion(payload)
            self._send_json({"status": "ok"})

    return Handler


def main() -> None:
    args = parse_args()
    shared = SharedFrames()
    threading.Thread(target=realsense_loop, args=(args, shared), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    print(f"Open http://<jetson-ip>:{args.port}/ in your browser.", file=sys.stderr if args.control_jsonl else sys.stdout)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
