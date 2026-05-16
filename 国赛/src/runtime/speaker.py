from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerConfig:
    enabled: bool
    command_template: str


class Speaker:
    def __init__(self, cfg: SpeakerConfig) -> None:
        self._cfg = cfg

    def say(self, text: str) -> None:
        logging.info("语音播报: %s", text)
        if not self._cfg.enabled:
            return
        cmd = self._cfg.command_template.format(text=text)
        subprocess.run(cmd, shell=True, check=True)
