from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
    return rows


def run(save_playback_log: bool) -> None:
    base = ROOT / "output" / "test_speaker_playback" / uuid.uuid4().hex
    audio_dir = base / "audio"
    log_path = base / "playback_log.jsonl"
    _write_tiny_wav(audio_dir / "A_low.wav")

    cfg = SpeakerConfig(
        enabled=True,
        engine="mock",
        audio_dir=str(audio_dir),
        save_playback_log=save_playback_log,
        playback_log_path=str(log_path),
    )
    speaker = AudioFileSpeaker(cfg)

    speaker.play("A_low")
    speaker.wait_until_done(timeout=2.0)
    assert not speaker.is_speaking(), "播放线程未结束"

    speaker.play("bad_key")
    speaker.wait_until_done(timeout=2.0)

    for _ in range(5):
        speaker.play("A_low")
        speaker.wait_until_done(timeout=2.0)

    if save_playback_log:
        rows = _read_log(log_path)
        assert len(rows) == 7, f"播放日志条数错误: {len(rows)}"
        assert rows[0]["key"] == "A_low"
        assert rows[0]["status"] == "played"
        assert rows[1]["key"] == "bad_key"
        assert rows[1]["status"] == "missing_file"

    print("speaker playback test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="测试预录音频播放闭环")
    parser.add_argument("--save-playback-log", action="store_true", help="保存播放 key 和时间戳日志")
    args = parser.parse_args()
    run(save_playback_log=args.save_playback_log)


if __name__ == "__main__":
    main()
