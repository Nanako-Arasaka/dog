"""hardware —— 硬件抽象层。

每类设备一个子包，包含 ABC 接口 + Mock 实现。
具体硬件驱动按需填充。
"""

from hardware.arm import ArmGateway, KinematicSolver, MockArm
from hardware.camera import CameraGateway, MockCamera
from hardware.speaker import EspeakSpeaker, MockSpeaker, SpeakerGateway

__all__ = [
    "ArmGateway",
    "CameraGateway",
    "SpeakerGateway",
    "KinematicSolver",
    "MockArm",
    "MockCamera",
    "MockSpeaker",
    "EspeakSpeaker",
]
