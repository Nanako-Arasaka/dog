"""配置数据类与加载逻辑。

所有配置集中在 AppConfig 下，按子系统拆分子数据类。
加载时做基本校验，避免无效配置进入运行时。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── 子配置 ──────────────────────────────────────────────


@dataclass(frozen=True)
class CameraConfig:
    """相机配置"""

    driver: str = "realsense"  # "realsense" | "mock"
    serial: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class SpeakerConfig:
    """语音播报配置"""

    enabled: bool = False
    engine: str = "mock"  # "mock" | "aplay" | "ffplay" | "powershell"
    language: str = "zh"
    audio_dir: str = "output/audio"
    save_playback_log: bool = False
    playback_log_path: str = "output/playback_log.jsonl"


@dataclass(frozen=True)
class PerceptionConfig:
    """感知层配置"""

    driver: str = "mock"  # "mock" | "local" | "remote"
    model_dir: str = "models/"
    confidence_threshold: float = 0.6
    scenario_file: str = ""  # mock 场景文件路径（driver=mock 时使用）


@dataclass(frozen=True)
class RemotePerceptionConfig:
    """远程算力板连接配置"""

    host: str = "192.168.1.200"
    port: int = 9800
    timeout_sec: float = 2.0


@dataclass(frozen=True)
class AppConfig:
    """应用总配置"""

    camera: CameraConfig
    speaker: SpeakerConfig
    perception: PerceptionConfig
    remote_perception: RemotePerceptionConfig = field(default_factory=RemotePerceptionConfig)
    project_root: str = ""  # 项目根目录，加载时自动填充


# ── 加载函数 ────────────────────────────────────────────


def _get_str(d: dict, key: str, default: str = "") -> str:
    v = d.get(key, default)
    if not isinstance(v, str):
        raise TypeError(f"config key '{key}' 应为字符串")
    return v


def _get_int(d: dict, key: str, default: int = 0) -> int:
    v = d.get(key, default)
    if not isinstance(v, int):
        raise TypeError(f"config key '{key}' 应为整数")
    return v


def _get_float(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    if not isinstance(v, (int, float)):
        raise TypeError(f"config key '{key}' 应为数字")
    return float(v)


def _get_bool(d: dict, key: str, default: bool = False) -> bool:
    v = d.get(key, default)
    if not isinstance(v, bool):
        raise TypeError(f"config key '{key}' 应为布尔值")
    return v


def load_app_config(config_path: str | Path) -> AppConfig:
    """从 JSON 文件加载并校验完整配置。"""

    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    project_root = str(path.parent.parent)

    # 相机
    camera_data = data.get("camera", {})
    camera = CameraConfig(
        driver=_get_str(camera_data, "driver", "mock"),
        serial=_get_str(camera_data, "serial", ""),
        width=_get_int(camera_data, "width", 640),
        height=_get_int(camera_data, "height", 480),
        fps=_get_int(camera_data, "fps", 30),
    )

    # 语音
    speaker_data = data.get("speaker", {})
    speaker = SpeakerConfig(
        enabled=_get_bool(speaker_data, "enabled", False),
        engine=_get_str(speaker_data, "engine", "mock"),
        language=_get_str(speaker_data, "language", "zh"),
        audio_dir=_get_str(speaker_data, "audio_dir", "output/audio"),
        save_playback_log=_get_bool(speaker_data, "save_playback_log", False),
        playback_log_path=_get_str(speaker_data, "playback_log_path", "output/playback_log.jsonl"),
    )

    # 感知
    perception_data = data.get("perception", {})
    scenario_rel = data.get("scenario_file", "config/scenario_mock.json")
    scenario_file = str((Path(project_root) / scenario_rel).resolve())
    perception = PerceptionConfig(
        driver=_get_str(perception_data, "driver", "mock"),
        model_dir=_get_str(perception_data, "model_dir", "models/"),
        confidence_threshold=_get_float(perception_data, "confidence_threshold", 0.6),
        scenario_file=scenario_file,
    )

    # 远程算力板
    remote_data = data.get("remote_perception", {})
    remote_perception = RemotePerceptionConfig(
        host=_get_str(remote_data, "host", "192.168.1.200"),
        port=_get_int(remote_data, "port", 9800),
        timeout_sec=_get_float(remote_data, "timeout_sec", 2.0),
    )

    return AppConfig(
        camera=camera,
        speaker=speaker,
        perception=perception,
        remote_perception=remote_perception,
        project_root=project_root,
    )
