#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLO cone detection and send obstacle-zone velocity commands."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from typing import Iterable

from .cone_detector_yolo import ConeYoloDetector
from .cone_strategy import AvoidanceConfig, ConeDetection, plan_cone_avoidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cone obstacle-zone runner")
    parser.add_argument("--model", default="/home/jetson/yolo_deploy/cone_best.pt")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--send-hz", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--web-host", default="0.0.0.0", help="HTTP bind host for browser preview.")
    parser.add_argument("--web-port", type=int, default=0, help="Enable browser preview on this port, e.g. 8080. 0 disables it.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="Browser preview JPEG quality.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for obstacle_zone_runner") from exc

    detector = ConeYoloDetector(args.model, conf=args.conf)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {args.camera}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.udp_host, int(args.udp_port))
    config = AvoidanceConfig(min_confidence=args.conf)
    send_interval = 1.0 / max(args.send_hz, 0.5)
    last_send_time = 0.0
    preview = BrowserPreview(args.web_host, args.web_port, args.jpeg_quality) if args.web_port > 0 else None
    if preview is not None:
        preview.start()

    print(f"[obstacle] model={args.model}")
    print(f"[obstacle] camera={args.camera}")
    print(f"[obstacle] UDP target={target[0]}:{target[1]} dry_run={args.dry_run}")
    if preview is not None:
        print(f"[obstacle] browser preview: http://<jetson-ip>:{args.web_port}/")
    if not args.no_display:
        print("[obstacle] press q to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[obstacle] failed to read frame")
                time.sleep(0.05)
                continue

            detections = detector.detect(frame)
            decision = plan_cone_avoidance(detections, frame.shape, config)
            now = time.time()
            payload = decision.to_payload()
            payload["source"] = "cone_obstacle"
            if now - last_send_time >= send_interval:
                send_payload(sock, target, payload, args.dry_run)
                last_send_time = now

            debug_frame = frame
            if preview is not None or not args.no_display:
                debug_frame = frame.copy()
                draw_debug(debug_frame, detections, decision)
            if preview is not None:
                preview.update(debug_frame, detections, payload, args)
            if not args.no_display:
                cv2.imshow("Cone Obstacle Avoidance", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        send_payload(sock, target, {"vx": 0.0, "vy": 0.0, "wz": 0.0, "source": "cone_obstacle", "state": "stop"}, args.dry_run)
        cap.release()
        sock.close()
        if preview is not None:
            preview.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


def send_payload(sock: socket.socket, target, payload: dict, dry_run: bool) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if dry_run:
        print(f"[dry-run] {payload}")
        return
    sock.sendto(data, target)


def draw_debug(frame, detections: Iterable[ConeDetection], decision) -> None:
    import cv2

    for det in detections:
        x1, y1, x2, y2 = [int(value) for value in det.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        label = f"cone {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    lines = [
        f"state: {decision.state}",
        f"vx: {decision.vx:.2f}",
        f"wz: {decision.wz:.2f}",
        decision.reason,
    ]
    for index, text in enumerate(lines):
        y = 32 + index * 30
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


class BrowserPreview:
    def __init__(self, host: str, port: int, jpeg_quality: int) -> None:
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.lock = threading.Lock()
        self.jpg: bytes | None = None
        self.status: dict[str, Any] = {"status": "starting"}
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.server = ThreadingHTTPServer((self.host, self.port), make_preview_handler(self))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()

    def update(self, frame, detections: Iterable[ConeDetection], payload: dict, args: argparse.Namespace) -> None:
        import cv2

        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            return
        dets = [
            {
                "xyxy": [round(float(v), 2) for v in det.xyxy],
                "confidence": round(float(det.confidence), 4),
                "class_name": det.class_name,
            }
            for det in detections
        ]
        status = {
            "timestamp": time.time(),
            "model": args.model,
            "camera": args.camera,
            "dry_run": bool(args.dry_run),
            "conf": float(args.conf),
            "detections": dets,
            "motion": payload,
        }
        with self.lock:
            self.jpg = buf.tobytes()
            self.status = status

    def get_jpg(self) -> bytes | None:
        with self.lock:
            return self.jpg

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.status)


def make_preview_handler(preview: BrowserPreview):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                html = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cone Obstacle Avoidance</title>
<style>
body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:18px}
img{max-width:100%;border:1px solid #444;background:#222}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0}
.box{background:#1f1f1f;border:1px solid #444;padding:10px}
.label{color:#aaa;font-size:13px}.value{font-size:26px;font-weight:700}
pre{background:#1b1b1b;border:1px solid #333;padding:12px;white-space:pre-wrap}
</style></head><body>
<h2>Cone Obstacle Avoidance</h2>
<img src="/stream.mjpg">
<div class="grid">
  <div class="box"><div class="label">state</div><div id="state" class="value">-</div></div>
  <div class="box"><div class="label">vx</div><div id="vx" class="value">-</div></div>
  <div class="box"><div class="label">vy</div><div id="vy" class="value">-</div></div>
  <div class="box"><div class="label">wz</div><div id="wz" class="value">-</div></div>
</div>
<pre id="status">loading...</pre>
<script>
function fmt(v){return typeof v === 'number' ? v.toFixed(3) : String(v ?? '-');}
async function tick(){
  const r = await fetch('/status.json');
  const s = await r.json();
  const m = s.motion || {};
  document.getElementById('state').textContent = m.state || '-';
  document.getElementById('vx').textContent = fmt(m.vx);
  document.getElementById('vy').textContent = fmt(m.vy);
  document.getElementById('wz').textContent = fmt(m.wz);
  document.getElementById('status').textContent = JSON.stringify(s, null, 2);
}
setInterval(tick, 500); tick();
</script></body></html>"""
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if self.path == "/status.json":
                payload = json.dumps(preview.get_status(), ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path != "/stream.mjpg":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = preview.get_jpg()
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


if __name__ == "__main__":
    raise SystemExit(main())
