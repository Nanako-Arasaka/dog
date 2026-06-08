from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.config import SpeakerConfig  # noqa: E402
from hardware.speaker.interface import AudioFileSpeaker  # noqa: E402


def _write_tiny_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"RIFF\x2c\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x40\x1f\x00\x00"
        b"\x01\x00\x08\x00data\x08\x00\x00\x00\x80\x80\x80\x80\x80\x80\x80\x80"
    )


def _speaker(case_name: str) -> tuple[AudioFileSpeaker, Path]:
    base = ROOT / "output" / "test_speaker_pytest" / f"{case_name}_{uuid.uuid4().hex}"
    audio_dir = base / "audio"
    log_path = base / "playback_log.jsonl"
    _write_tiny_wav(audio_dir / "A_low.wav")
    cfg = SpeakerConfig(
        enabled=True,
        engine="mock",
        audio_dir=str(audio_dir),
        save_playback_log=True,
        playback_log_path=str(log_path),
    )
    return AudioFileSpeaker(cfg), log_path


def _read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_audio_file_speaker_play_success() -> None:
    speaker, log_path = _speaker("success")

    speaker.play("A_low")
    speaker.wait_until_done(timeout=2.0)

    rows = _read_log(log_path)
    assert rows[0]["key"] == "A_low"
    assert rows[0]["status"] == "played"
    assert "timestamp" in rows[0]


def test_audio_file_speaker_invalid_key_falls_back() -> None:
    speaker, log_path = _speaker("invalid_key")

    speaker.play("bad_key")
    speaker.wait_until_done(timeout=2.0)

    rows = _read_log(log_path)
    assert rows[0]["key"] == "bad_key"
    assert rows[0]["status"] == "missing_file"


def test_audio_file_speaker_repeated_play_stable() -> None:
    speaker, log_path = _speaker("repeated")

    for _ in range(5):
        speaker.play("A_low")
        speaker.wait_until_done(timeout=2.0)

    rows = _read_log(log_path)
    assert len(rows) == 5
    assert all(row["status"] == "played" for row in rows)
