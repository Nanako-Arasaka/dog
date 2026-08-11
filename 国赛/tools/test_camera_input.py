"""CameraInput smoke tests for the compute-board vision server."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera_input import CameraInput, CameraInputConfig  # noqa: E402


def _cv2_or_none():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None


def test_mock_reads_10_frames() -> None:
    camera = CameraInput(CameraInputConfig(mode="mock", width=320, height=240))
    assert camera.open() is True
    try:
        frame_ids = []
        for _ in range(10):
            frame = camera.read()
            assert frame is not None
            assert frame.width == 320
            assert frame.height == 240
            assert frame.source_type == "mock"
            assert frame.image.shape == (240, 320, 3)
            frame_ids.append(frame.frame_id)
        assert frame_ids == list(range(1, 11))
    finally:
        camera.close()


def test_video_reads_frame_when_opencv_available() -> None:
    cv2 = _cv2_or_none()
    if cv2 is None:
        print("SKIP test_video_reads_frame_when_opencv_available: cv2 not installed")
        return

    tmp_dir = ROOT / "output" / "test_camera_input_video"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        video_path = tmp_dir / "sample.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (160, 120),
        )
        assert writer.isOpened()
        for i in range(5):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :, 1] = 30 + i * 20
            writer.write(frame)
        writer.release()

        camera = CameraInput(CameraInputConfig(mode="video", source=str(video_path), width=320, height=240))
        assert camera.open() is True
        try:
            frame = camera.read()
            assert frame is not None
            assert frame.source_type == "video"
            assert frame.width == 320
            assert frame.height == 240
        finally:
            camera.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_camera_mode_fails_gracefully_without_camera() -> None:
    camera = CameraInput(CameraInputConfig(mode="camera", source="9999", width=320, height=240))
    opened = camera.open()
    try:
        if not opened:
            assert camera.read() is None
        else:
            # Extremely unlikely on normal machines, but if index 9999 exists,
            # the contract is still that read does not crash.
            _ = camera.read()
    finally:
        camera.close()


def test_debug_frame_saves() -> None:
    tmp_dir = ROOT / "output" / "test_camera_input_debug"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        camera = CameraInput(CameraInputConfig(
            mode="mock",
            width=160,
            height=120,
            save_debug_frames=True,
            debug_dir=str(tmp_dir),
            save_every=2,
        ))
        assert camera.open() is True
        try:
            for _ in range(3):
                assert camera.read() is not None
        finally:
            camera.close()
        saved = list(tmp_dir.glob("frame_*"))
        assert saved, "expected at least one debug frame"
        assert any("000002" in p.name for p in saved)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    tests = [
        test_mock_reads_10_frames,
        test_video_reads_frame_when_opencv_available,
        test_camera_mode_fails_gracefully_without_camera,
        test_debug_frame_saves,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("camera input checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
