"""语音播报硬件抽象接口 —— 基于预录音频文件播放。

预录音频文件（12 条固定播报 + 通用提示）：
  A_low.wav, A_normal.wav, A_high.wav,
  B_low.wav, B_normal.wav, B_high.wav,
  C_low.wav, C_normal.wav, C_high.wav,
  D_low.wav, D_normal.wav, D_high.wav,
  obstacle_cleared.wav, obstacle_detected.wav,
  inspection_start.wav, inspection_complete.wav,
  all_normal.wav, pickup_start.wav,
  pickup_success_A.wav, pickup_success_B.wav, pickup_success_C.wav, pickup_success_D.wav,
  drop_warning_1.wav, drop_warning_2.wav, drop_warning_3.wav,
  drop_limit.wav, task_failed.wav, task_complete.wav
"""

from __future__ import annotations

import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import SpeakerConfig


class SpeakerGateway(ABC):
    """语音播报抽象 —— 播放本地预录音频文件。

    play(key) 是关键方法，按 key 名查找音频文件播放。
    say_async(text) 作为 fallback，仅 log 不真正合成语音。
    """

    @abstractmethod
    def play(self, audio_key: str) -> None:
        """播放预录音频文件（非阻塞）。

        Args:
            audio_key: 音频标识，如 "A_low", "obstacle_cleared"。
        """
        ...

    @abstractmethod
    def say_async(self, text: str) -> None:
        """非阻塞播报 fallback —— 仅 log，不合成。
        所有关键播报应走 play() 播放预录音频。
        """
        ...

    @abstractmethod
    def is_speaking(self) -> bool: ...
    @abstractmethod
    def wait_until_done(self, timeout: float = 5.0) -> None: ...


# ── 音频文件播放实现 ────────────────────────────────────


class AudioFileSpeaker(SpeakerGateway):
    """基于本地 .wav 文件的语音播报。

    音频文件目录结构：
      audio/
        A_low.wav, A_normal.wav, A_high.wav, ...
        obstacle_cleared.wav, ...
    """

    def __init__(self, cfg: SpeakerConfig) -> None:
        self._cfg = cfg
        self._audio_dir = Path(cfg.language) if cfg.language else Path("audio")
        self._active_thread: threading.Thread | None = None
        self._player = cfg.engine  # "aplay" | "ffplay" | "powershell"

    def play(self, audio_key: str) -> None:
        logging.info("语音播报: key=%s", audio_key)
        t = threading.Thread(target=self._play_file, args=(audio_key,), daemon=True)
        self._active_thread = t
        t.start()

    def say_async(self, text: str) -> None:
        """Fallback: 仅 log，不真正合成语音。"""
        logging.info("语音播报(fallback/log): %s", text)

    def is_speaking(self) -> bool:
        return self._active_thread is not None and self._active_thread.is_alive()

    def wait_until_done(self, timeout: float = 5.0) -> None:
        if self._active_thread:
            self._active_thread.join(timeout=timeout)

    def _play_file(self, audio_key: str) -> None:
        fname = f"{audio_key}.wav"
        fpath = self._audio_dir / fname
        if not fpath.exists():
            logging.warning("音频文件不存在: %s", fpath)
            return
        try:
            self._run_player(str(fpath))
        except FileNotFoundError:
            logging.warning("音频播放器 %s 未安装", self._player)
        except Exception as exc:
            logging.error("音频播放异常: %s", exc)

    def _run_player(self, filepath: str) -> None:
        p = self._player
        if p == "aplay":
            subprocess.run(["aplay", filepath], capture_output=True, timeout=10, check=False)
        elif p == "ffplay":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", filepath],
                           capture_output=True, timeout=10, check=False)
        elif p == "powershell":
            subprocess.run(["powershell", "-c",
                f'(New-Object Media.SoundPlayer "{filepath}").PlaySync()'],
                capture_output=True, timeout=10, check=False)
        else:
            logging.warning("未知播放器: %s，跳过播放 %s", p, filepath)


# ── Mock 实现 ────────────────────────────────────────────


class MockSpeaker(SpeakerGateway):
    """不发声的 Speaker —— 仅 log，用于开发/CI。"""

    def __init__(self, cfg: SpeakerConfig) -> None:
        self._cfg = cfg

    def play(self, audio_key: str) -> None:
        logging.info("语音播报(mock): key=%s", audio_key)

    def say_async(self, text: str) -> None:
        logging.info("语音播报(mock/fallback): %s", text)

    def is_speaking(self) -> bool:
        return False

    def wait_until_done(self, timeout: float = 5.0) -> None:
        pass
