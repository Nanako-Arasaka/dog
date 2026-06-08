"""语音播报硬件抽象接口。"""

from __future__ import annotations

import logging
import subprocess
import threading
from abc import ABC, abstractmethod

from app.config import SpeakerConfig


class SpeakerGateway(ABC):
    """语音播报抽象。

    实现：
    - EspeakSpeaker   (本地 espeak/espeak-ng TTS)
    - MockSpeaker     (仅 log，不发声)
    - CloudTTSSpeaker (云端 TTS，后续扩展)
    """

    @abstractmethod
    def say_async(self, text: str) -> None:
        """非阻塞播报：启动后台线程/进程，立即返回。

        比赛要求语音播报，但不应阻塞主循环。
        """
        ...

    @abstractmethod
    def say_blocking(self, text: str) -> None:
        """阻塞播报：等待播报完成后再返回。

        仅用于关键节点（如任务完成宣告），大多数场景用 say_async。
        """
        ...

    @abstractmethod
    def is_speaking(self) -> bool:
        """是否正在播报。"""
        ...

    @abstractmethod
    def wait_until_done(self, timeout: float = 5.0) -> None:
        """等待当前播报完成（超时则放弃）。"""
        ...


# ── Espeak 实现 ──────────────────────────────────────────


class EspeakSpeaker(SpeakerGateway):
    """基于 espeak/espeak-ng 的本地 TTS。"""

    def __init__(self, cfg: SpeakerConfig) -> None:
        self._cfg = cfg
        self._active_thread: threading.Thread | None = None

    def say_async(self, text: str) -> None:
        logging.info("语音播报: %s", text)
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        self._active_thread = t
        t.start()

    def say_blocking(self, text: str) -> None:
        logging.info("语音播报(阻塞): %s", text)
        self._speak(text)

    def is_speaking(self) -> bool:
        return self._active_thread is not None and self._active_thread.is_alive()

    def wait_until_done(self, timeout: float = 5.0) -> None:
        if self._active_thread:
            self._active_thread.join(timeout=timeout)

    def _speak(self, text: str) -> None:
        try:
            subprocess.run(
                ["espeak", "-v", self._cfg.language, text],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            logging.warning("espeak 未安装，跳过语音播报")
        except Exception as exc:
            logging.error("语音播报异常: %s", exc)


# ── Mock 实现 ────────────────────────────────────────────


class MockSpeaker(SpeakerGateway):
    """不发声的 Speaker —— 仅 log 播报文本，用于开发/CI。"""

    def __init__(self, cfg: SpeakerConfig) -> None:
        self._cfg = cfg
        self._speaking = False

    def say_async(self, text: str) -> None:
        logging.info("语音播报(mock): %s", text)

    def say_blocking(self, text: str) -> None:
        logging.info("语音播报(mock/阻塞): %s", text)

    def is_speaking(self) -> bool:
        return False

    def wait_until_done(self, timeout: float = 5.0) -> None:
        pass
