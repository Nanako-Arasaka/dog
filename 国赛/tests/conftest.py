"""pytest 共享 fixtures —— 提供各组件的 mock 实例。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.config import (
    AppConfig,
    CameraConfig,
    PerceptionConfig,
    RemotePerceptionConfig,
    SpeakerConfig,
)
from hardware.camera.interface import MockCamera
from hardware.speaker.interface import MockSpeaker


@pytest.fixture
def mock_camera_cfg() -> CameraConfig:
    return CameraConfig(driver="mock", width=640, height=480, fps=30)


@pytest.fixture
def mock_camera(mock_camera_cfg: CameraConfig) -> MockCamera:
    cam = MockCamera(mock_camera_cfg)
    cam.start()
    yield cam
    cam.stop()


@pytest.fixture
def mock_speaker_cfg() -> SpeakerConfig:
    return SpeakerConfig(enabled=False, engine="mock", language="zh")


@pytest.fixture
def mock_speaker(mock_speaker_cfg: SpeakerConfig) -> MockSpeaker:
    return MockSpeaker(mock_speaker_cfg)


@pytest.fixture
def minimal_app_config(tmp_path) -> AppConfig:
    """最小可用的 AppConfig，用于 CI/测试。"""
    return AppConfig(
        camera=CameraConfig(driver="mock"),
        speaker=SpeakerConfig(enabled=False, engine="mock"),
        perception=PerceptionConfig(driver="remote", scenario_file=str(tmp_path / "scenario.json")),
        remote_perception=RemotePerceptionConfig(host="127.0.0.1", port=9800, timeout_sec=0.2),
        project_root=str(tmp_path),
    )
