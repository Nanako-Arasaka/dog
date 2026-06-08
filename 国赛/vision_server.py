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
import threading
import time
from dataclasses import dataclass
from typing import Any


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


class FrameSource:
    """Minimal frame source wrapper.

    The current first-stage server returns mock detections regardless of frame
    contents. The source is still opened to validate mock/video/camera modes.
    """

    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        self._capture: Any = None

    def open(self) -> None:
        if self._cfg.mode == "mock":
            logging.info("frame source: mock")
            return

        try:
            import cv2  # type: ignore
        except ImportError:
            logging.warning("opencv-python not installed; %s source is not opened", self._cfg.mode)
            return

        source: int | str
        if self._cfg.mode == "camera":
            source = int(self._cfg.source) if self._cfg.source else 0
        elif self._cfg.mode == "video":
            source = self._cfg.source
            if not source:
                raise ValueError("--source is required when --mode=video")
        else:
            raise ValueError(f"unknown source mode: {self._cfg.mode}")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {self._cfg.mode} source: {source}")
        if self._cfg.mode == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        self._capture = cap
        logging.info("frame source opened: mode=%s source=%s", self._cfg.mode, source)

    def read(self) -> None:
        if self._capture is None:
            return
        ok, _frame = self._capture.read()
        if not ok and self._cfg.mode == "video":
            self._capture.set(1, 0)
            self._capture.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class MockDetector:
    """First-stage detector: protocol-complete mock results only."""

    def __init__(self, empty_results: bool = False) -> None:
        self._empty = empty_results

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        req = str(request.get("req", ""))
        ts = time.time()

        if req == "detect_obstacles":
            return {"type": "obstacles", "detections": [] if self._empty else [
                {
                    "bbox": {"x1": 260, "y1": 210, "x2": 330, "y2": 390},
                    "center_3d": [0.15, 0.0, 1.2],
                    "confidence": 0.92,
                }
            ], "timestamp": ts}

        if req == "detect_zone_letters":
            detections = [] if self._empty else [
                {"zone": "A", "confidence": 0.95, "bbox": {"x1": 80, "y1": 90, "x2": 160, "y2": 170}},
                {"zone": "B", "confidence": 0.94, "bbox": {"x1": 260, "y1": 90, "x2": 340, "y2": 170}},
                {"zone": "C", "confidence": 0.93, "bbox": {"x1": 80, "y1": 260, "x2": 160, "y2": 340}},
                {"zone": "D", "confidence": 0.96, "bbox": {"x1": 260, "y1": 260, "x2": 340, "y2": 340}},
            ]
            return {"type": "zone_letters", "detections": detections, "timestamp": ts}

        if req == "detect_gauges":
            detections = [] if self._empty else [
                {"zone": "A", "status": "low", "confidence": 0.94, "raw_value": 20.0},
                {"zone": "B", "status": "normal", "confidence": 0.95, "raw_value": 50.0},
                {"zone": "C", "status": "high", "confidence": 0.93, "raw_value": 82.0},
                {"zone": "D", "status": "normal", "confidence": 0.96, "raw_value": 52.0},
            ]
            return {"type": "gauges", "detections": detections, "timestamp": ts}

        if req == "detect_red_strips":
            detections = [] if self._empty else [
                {
                    "bbox": {"x1": 230, "y1": 200, "x2": 420, "y2": 290},
                    "center_3d": [0.05, 0.0, 0.28],
                    "confidence": 0.91,
                }
            ]
            return {"type": "red_strips", "detections": detections, "timestamp": ts}

        if req == "estimate_target_pose":
            if self._empty:
                return {"type": "target_pose", "pose": None, "confidence": 0.0, "timestamp": ts}
            return {
                "type": "target_pose",
                "pose": {"x": 0.05, "y": 0.0, "z": 0.28, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "confidence": 0.9,
                "timestamp": ts,
            }

        return {"type": "error", "message": f"unknown request: {req}", "timestamp": ts}


class VisionServer:
    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        self._source = FrameSource(cfg)
        self._detector = MockDetector(empty_results=cfg.empty_results)
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        self._source.open()
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
            self._source.close()

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
        self._source.read()
        try:
            request = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return {"type": "error", "message": f"invalid json: {exc}", "timestamp": time.time()}
        if not isinstance(request, dict):
            return {"type": "error", "message": "request must be a JSON object", "timestamp": time.time()}
        return self._detector.handle(request)


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="TCP JSON vision server for the compute board")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9800)
    parser.add_argument("--mode", choices=("mock", "video", "camera"), default="mock")
    parser.add_argument("--source", default="", help="video path or camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--empty-results", action="store_true")
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
    )


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
