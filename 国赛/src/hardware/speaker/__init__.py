"""hardware.speaker —— 语音播报抽象。"""

from hardware.speaker.interface import AudioFileSpeaker, MockSpeaker, SpeakerGateway

__all__ = ["SpeakerGateway", "AudioFileSpeaker", "MockSpeaker"]
