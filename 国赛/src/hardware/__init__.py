"""hardware —— 巡检识别闭环保留的硬件抽象层。"""

from hardware.camera import CameraGateway, MockCamera
from hardware.speaker import AudioFileSpeaker, MockSpeaker, SpeakerGateway

__all__ = [
    "CameraGateway",
    "SpeakerGateway",
    "MockCamera",
    "MockSpeaker",
    "AudioFileSpeaker",
]
