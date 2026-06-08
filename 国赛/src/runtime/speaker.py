"""向后兼容 re-export —— Speaker 已迁移至 hardware.speaker。

新代码请直接:
    from hardware.speaker import SpeakerGateway, EspeakSpeaker, MockSpeaker
"""

from __future__ import annotations

from hardware.speaker.interface import EspeakSpeaker, MockSpeaker, SpeakerGateway

# 保留旧的 Speaker 别名
Speaker = EspeakSpeaker

__all__ = ["SpeakerGateway", "Speaker", "EspeakSpeaker", "MockSpeaker"]
