"""Camera input layer for the compute-board vision server.

This module owns image acquisition only. It does not run detectors and does
not control robot motion, arm motion, or audio.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraInputConfig:
    mode: str = "mock"
    source: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_horizontal: bool = False
    roi: tuple[int, int, int, int] | None = None
    save_debug_frames: bool = False
    debug_dir: str = "output/debug_frames"
    save_every: int = 30


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int
    source_type: str


class CameraInput:
    """Unified frame reader for mock, video, and camera sources."""

    def __init__(self, cfg: CameraInputConfig) -> None:
        self._cfg = cfg
        self._capture: Any = None
        self._frame_id = 0
        self._cv2: Any = None
        self._opened = False

    def open(self) -> bool:
        if self._cfg.mode == "mock":
            self._opened = True
            logging.info("camera input opened: mock")
            return True

        try:
            import cv2  # type: ignore
        except ImportError:
            logging.error("opencv-python is required for %s input", self._cfg.mode)
            self._opened = False
            return False

        self._cv2 = cv2
        source: int | str
        if self._cfg.mode == "camera":
            source = int(self._cfg.source) if self._cfg.source else 0
        elif self._cfg.mode == "video":
            if not self._cfg.source:
                logging.error("--source is required when --mode=video")
                self._opened = False
                return False
            source = self._cfg.source
        else:
            logging.error("unknown camera input mode: %s", self._cfg.mode)
            self._opened = False
            return False

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            logging.error("failed to open %s source: %s", self._cfg.mode, source)
            self._opened = False
            return False

        if self._cfg.mode == "camera":
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
            capture.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        self._capture = capture
        self._opened = True
        logging.info("camera input opened: mode=%s source=%s", self._cfg.mode, source)
        return True

    def read(self) -> VisionFrame | None:
        if not self._opened:
            logging.warning("camera input is not open")
            return None

        if self._cfg.mode == "mock":
            image = self._mock_image()
        else:
            if self._capture is None:
                logging.warning("capture object is not available")
                return None
            ok, image = self._capture.read()
            if not ok or image is None:
                if self._cfg.mode == "video":
                    self._capture.set(1, 0)
                    ok, image = self._capture.read()
                if not ok or image is None:
                    logging.warning("failed to read frame from %s input", self._cfg.mode)
                    return None

        image = self._preprocess(image)
        self._frame_id += 1
        frame = VisionFrame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            image=image,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            source_type=self._cfg.mode,
        )
        self._save_debug_frame(frame)
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened

    def _mock_image(self) -> np.ndarray:
        image = np.zeros((self._cfg.height, self._cfg.width, 3), dtype=np.uint8)
        image[:, :, 0] = 32
        image[:, :, 1] = 48
        image[:, :, 2] = 64
        x = (self._frame_id * 7) % max(self._cfg.width, 1)
        x2 = min(x + 80, self._cfg.width)
        image[self._cfg.height // 3:self._cfg.height // 3 + 60, x:x2, :] = (0, 0, 255)
        self._draw_mock_gauge(image)
        self._draw_mock_letters(image)
        return image

    def _draw_mock_gauge(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        radius = max(28, min(w, h) // 7)
        cx = min(max(radius + 16, w // 4), w - radius - 1)
        cy = min(max(radius + 16, h // 2), h - radius - 1)
        yy, xx = np.indices((h, w))
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        image[disk] = (240, 240, 240)
        ring = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius) < 2.0
        image[ring] = (20, 20, 20)

        angle_deg = 210.0
        tip_x = int(cx + np.cos(np.deg2rad(angle_deg)) * radius * 0.75)
        tip_y = int(cy - np.sin(np.deg2rad(angle_deg)) * radius * 0.75)
        steps = max(abs(tip_x - cx), abs(tip_y - cy), 1)
        xs = np.linspace(cx, tip_x, steps).astype(np.int64)
        ys = np.linspace(cy, tip_y, steps).astype(np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                px = np.clip(xs + dx, 0, w - 1)
                py = np.clip(ys + dy, 0, h - 1)
                image[py, px] = (0, 0, 0)

    def _draw_mock_letters(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        positions = [
            ("A", int(w * 0.58), int(h * 0.14)),
            ("B", int(w * 0.74), int(h * 0.14)),
            ("C", int(w * 0.58), int(h * 0.48)),
            ("D", int(w * 0.74), int(h * 0.48)),
        ]
        for letter, x, y in positions:
            mask = _letter_mask(letter, width=48, height=72)
            pad = 12
            bg_y1 = max(0, y - pad)
            bg_x1 = max(0, x - pad)
            bg_y2 = min(h, y + mask.shape[0] + pad)
            bg_x2 = min(w, x + mask.shape[1] + pad)
            image[bg_y1:bg_y2, bg_x1:bg_x2] = (245, 245, 245)
            y2 = min(y + mask.shape[0], h)
            x2 = min(x + mask.shape[1], w)
            sub = mask[:y2 - y, :x2 - x]
            patch = image[y:y2, x:x2]
            patch[sub] = (0, 0, 0)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if self._cfg.roi is not None:
            x, y, w, h = self._cfg.roi
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            image = image[y:y + h, x:x + w]
            if image.size == 0:
                logging.warning("configured ROI produced an empty image; using black fallback")
                image = np.zeros((self._cfg.height, self._cfg.width, 3), dtype=np.uint8)

        if self._cv2 is not None:
            image = self._cv2.resize(image, (self._cfg.width, self._cfg.height))
            if self._cfg.flip_horizontal:
                image = self._cv2.flip(image, 1)
            return image

        # Mock/fallback path without OpenCV.
        if image.shape[0] != self._cfg.height or image.shape[1] != self._cfg.width:
            y_idx = np.linspace(0, image.shape[0] - 1, self._cfg.height).astype(np.int64)
            x_idx = np.linspace(0, image.shape[1] - 1, self._cfg.width).astype(np.int64)
            image = image[y_idx][:, x_idx]
        if self._cfg.flip_horizontal:
            image = image[:, ::-1]
        return np.ascontiguousarray(image)

    def _save_debug_frame(self, frame: VisionFrame) -> None:
        if not self._cfg.save_debug_frames:
            return
        every = max(int(self._cfg.save_every), 1)
        if frame.frame_id % every != 0:
            return
        out_dir = Path(self._cfg.debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        millis = int(frame.timestamp * 1000)
        if self._cv2 is not None:
            path = out_dir / f"frame_{frame.frame_id:06d}_{millis}.jpg"
            self._cv2.imwrite(str(path), frame.image)
            return
        path = out_dir / f"frame_{frame.frame_id:06d}_{millis}.ppm"
        self._write_ppm(path, frame.image)

    @staticmethod
    def _write_ppm(path: Path, image: np.ndarray) -> None:
        # PPM fallback keeps debug saving available in mock mode without cv2.
        rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
        header = f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
        path.write_bytes(header + np.ascontiguousarray(rgb).tobytes())


def _letter_mask(letter: str, width: int, height: int) -> np.ndarray:
    patterns = {
        "A": ["0011100", "0110110", "1100011", "1100011", "1111111", "1100011", "1100011", "1100011", "1100011"],
        "B": ["1111100", "1100110", "1100011", "1100110", "1111100", "1100110", "1100011", "1100110", "1111100"],
        "C": ["0011110", "0110011", "1100000", "1100000", "1100000", "1100000", "1100000", "0110011", "0011110"],
        "D": ["1111000", "1101100", "1100110", "1100011", "1100011", "1100011", "1100110", "1101100", "1111000"],
    }
    small = np.array([[ch == "1" for ch in row] for row in patterns[letter]], dtype=bool)
    y_idx = np.linspace(0, small.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, small.shape[1] - 1, width).astype(np.int64)
    return small[y_idx][:, x_idx]


def parse_roi(raw: str) -> tuple[int, int, int, int] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be formatted as x,y,w,h")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]
