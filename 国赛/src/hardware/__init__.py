"""hardware —— 硬件抽象层。"""

from hardware.arm import ArmGateway, KinematicSolver, MockArm
from hardware.camera import CameraGateway, MockCamera
from hardware.speaker import AudioFileSpeaker, MockSpeaker, SpeakerGateway

__all__ = [
    "ArmGateway",
    "CameraGateway",
    "SpeakerGateway",
    "KinematicSolver",
    "MockArm",
    "MockCamera",
    "MockSpeaker",
    "AudioFileSpeaker",
]
