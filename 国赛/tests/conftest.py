"""pytest 共享 fixtures —— 提供各组件的 mock 实例。"""

from __future__ import annotations

import pytest

from app.config import (
    AppConfig,
    ArmConfig,
    CameraConfig,
    MissionConfig,
    PerceptionConfig,
    RobotNetworkConfig,
    SpeakerConfig,
    TimingConfig,
)
from hardware.arm.interface import MockArm
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
def mock_arm_cfg() -> ArmConfig:
    return ArmConfig(driver="mock")


@pytest.fixture
def mock_arm(mock_arm_cfg: ArmConfig) -> MockArm:
    arm = MockArm(mock_arm_cfg)
    arm.connect()
    yield arm
    arm.disconnect()


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
        robot=RobotNetworkConfig(ip="127.0.0.1", command_port=43893, local_ip="0.0.0.0", local_telemetry_port=43897),
        timing=TimingConfig(heartbeat_hz=2.0, main_loop_hz=20.0),
        camera=CameraConfig(driver="mock"),
        arm=ArmConfig(driver="mock"),
        speaker=SpeakerConfig(enabled=False, engine="mock"),
        mission=MissionConfig(),
        perception=PerceptionConfig(scenario_file=str(tmp_path / "scenario.json")),
        project_root=str(tmp_path),
    )
