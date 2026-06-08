"""Vision server for the external compute board.

This process only reads image sources and returns structured perception JSON.
It must not control the robot, arm, or speaker.

Protocol: JSON Lines over TCP.
Request examples:
  {"req": "detect_obstacles"}
  {"req": "detect_zone_letters"}
  {"req": "detect_gauges"}
  {"req": "detect_red_strips"}
  {"req": "estimate_target_pose", "target": "strip"}
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from camera_input import CameraInput, CameraInputConfig, VisionFrame, parse_roi

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from perception.detector.fixed_detector import (  # noqa: E402
    FixedDetectionConfig,
    FixedDetectionPipeline,
    empty_response,
)

@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    mode: str
    source: str
    width: int
    height: int
    fps: int
    empty_results: bool
    response_delay_sec: float
    disconnect_after: int
    flip_horizontal: bool
    roi: tuple[int, int, int, int] | None
    save_debug_frames: bool
    debug_dir: str
    save_every: int
    letter_min_confidence: float
    letter_template_dir: str
    letter_debug_save_roi: bool
    letter_debug_dir: str
    inspection_debug_save: bool
    inspection_debug_dir: str
    inspection_max_match_distance: float
    gauge_low_angle_range: tuple[float, float]
    gauge_normal_angle_range: tuple[float, float]
    gauge_high_angle_range: tuple[float, float]
    gauge_min_confidence: float
    gauge_debug_save_roi: bool
    gauge_debug_dir: str


class VisionServer:
    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        self._camera = CameraInput(CameraInputConfig(
            mode=cfg.mode,
            source=cfg.source,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            flip_horizontal=cfg.flip_horizontal,
            roi=cfg.roi,
            save_debug_frames=cfg.save_debug_frames,
            debug_dir=cfg.debug_dir,
            save_every=cfg.save_every,
        ))
        self._detector = FixedDetectionPipeline(FixedDetectionConfig(
            empty_results=cfg.empty_results,
            letter_min_confidence=cfg.letter_min_confidence,
            letter_template_dir=cfg.letter_template_dir,
            letter_debug_save_roi=cfg.letter_debug_save_roi,
            letter_debug_dir=cfg.letter_debug_dir,
            inspection_debug_save=cfg.inspection_debug_save,
            inspection_debug_dir=cfg.inspection_debug_dir,
            inspection_max_match_distance=cfg.inspection_max_match_distance,
            gauge_low_angle_range=cfg.gauge_low_angle_range,
            gauge_normal_angle_range=cfg.gauge_normal_angle_range,
            gauge_high_angle_range=cfg.gauge_high_angle_range,
            gauge_min_confidence=cfg.gauge_min_confidence,
            gauge_debug_save_roi=cfg.gauge_debug_save_roi,
            gauge_debug_dir=cfg.gauge_debug_dir,
        ))
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        self._camera.open()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self._cfg.host, self._cfg.port))
                srv.listen(8)
                srv.settimeout(0.5)
                logging.info("vision server listening on %s:%d", self._cfg.host, self._cfg.port)
                while not self._stop.is_set():
                    try:
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        finally:
            self._camera.close()

    def stop(self) -> None:
        self._stop.set()

    def _handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        logging.info("client connected: %s:%d", addr[0], addr[1])
        handled = 0
        buffer = b""
        with conn:
            conn.settimeout(10.0)
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        handled += 1
                        response = self._handle_line(line)
                        if self._cfg.response_delay_sec > 0:
                            time.sleep(self._cfg.response_delay_sec)
                        conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                        if self._cfg.disconnect_after > 0 and handled >= self._cfg.disconnect_after:
                            logging.info("debug disconnect after %d requests", handled)
                            return
                except (OSError, socket.timeout) as exc:
                    logging.info("client disconnected: %s", exc)
                    return

    def _handle_line(self, line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return {"type": "error", "message": f"invalid json: {exc}", "timestamp": time.time()}
        if not isinstance(request, dict):
            return {"type": "error", "message": "request must be a JSON object", "timestamp": time.time()}
        frame = self._camera.read()
        if frame is None:
            req = str(request.get("req", ""))
            logging.warning("empty result because no frame is available for request: %s", req)
            return empty_response(req)
        return self._detector.handle(request, frame)


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="TCP JSON vision server for the compute board")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9800)
    parser.add_argument("--mode", choices=("mock", "video", "camera"), default="mock")
    parser.add_argument("--source", default="", help="video path or camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--roi", default="", help="reserved ROI crop as x,y,w,h")
    parser.add_argument("--save-debug-frames", action="store_true")
    parser.add_argument("--debug-dir", default="output/debug_frames")
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--empty-results", action="store_true")
    parser.add_argument("--letter-min-confidence", type=float, default=0.55)
    parser.add_argument("--letter-template-dir", default="assets/templates/letters")
    parser.add_argument("--letter-debug-save-roi", action="store_true")
    parser.add_argument("--letter-debug-dir", default="output/debug_letters")
    parser.add_argument("--inspection-debug-save", action="store_true")
    parser.add_argument("--inspection-debug-dir", default="output/debug_inspection")
    parser.add_argument("--inspection-max-match-distance", type=float, default=180.0)
    parser.add_argument("--gauge-low-angle-range", default="180,250")
    parser.add_argument("--gauge-normal-angle-range", default="250,310")
    parser.add_argument("--gauge-high-angle-range", default="310,30")
    parser.add_argument("--gauge-min-confidence", type=float, default=0.55)
    parser.add_argument("--gauge-debug-save-roi", action="store_true")
    parser.add_argument("--gauge-debug-dir", default="output/debug_gauge")
    parser.add_argument("--response-delay-sec", type=float, default=0.0)
    parser.add_argument("--disconnect-after", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return ServerConfig(
        host=args.host,
        port=args.port,
        mode=args.mode,
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        empty_results=args.empty_results,
        response_delay_sec=args.response_delay_sec,
        disconnect_after=args.disconnect_after,
        flip_horizontal=args.flip_horizontal,
        roi=parse_roi(args.roi),
        save_debug_frames=args.save_debug_frames,
        debug_dir=args.debug_dir,
        save_every=args.save_every,
        letter_min_confidence=args.letter_min_confidence,
        letter_template_dir=args.letter_template_dir,
        letter_debug_save_roi=args.letter_debug_save_roi,
        letter_debug_dir=args.letter_debug_dir,
        inspection_debug_save=args.inspection_debug_save,
        inspection_debug_dir=args.inspection_debug_dir,
        inspection_max_match_distance=args.inspection_max_match_distance,
        gauge_low_angle_range=_parse_angle_range(args.gauge_low_angle_range),
        gauge_normal_angle_range=_parse_angle_range(args.gauge_normal_angle_range),
        gauge_high_angle_range=_parse_angle_range(args.gauge_high_angle_range),
        gauge_min_confidence=args.gauge_min_confidence,
        gauge_debug_save_roi=args.gauge_debug_save_roi,
        gauge_debug_dir=args.gauge_debug_dir,
    )


def _parse_angle_range(raw: str) -> tuple[float, float]:
    parts = [item.strip() for item in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("angle range must be formatted as start,end")
    return float(parts[0]), float(parts[1])

def main() -> int:
    cfg = parse_args()
    server = VisionServer(cfg)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("vision server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
