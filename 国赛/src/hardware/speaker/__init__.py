"""hardware.speaker —— 语音播报抽象。"""

from hardware.speaker.interface import EspeakSpeaker, MockSpeaker, SpeakerGateway

__all__ = ["SpeakerGateway", "EspeakSpeaker", "MockSpeaker"]
