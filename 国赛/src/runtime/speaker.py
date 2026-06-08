"""向后兼容 re-export —— Speaker 已迁移至 hardware.speaker。"""

from __future__ import annotations

from hardware.speaker.interface import AudioFileSpeaker, MockSpeaker, SpeakerGateway

# 旧别名
Speaker = AudioFileSpeaker

__all__ = ["SpeakerGateway", "Speaker", "AudioFileSpeaker", "MockSpeaker"]
