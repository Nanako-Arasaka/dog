"""app 模块 —— 配置、容器、入口编排。"""

from app.config import (
    AppConfig,
    CameraConfig,
    PerceptionConfig,
    RemotePerceptionConfig,
    SpeakerConfig,
    load_app_config,
)
from app.container import AppContainer

__all__ = [
    "AppConfig",
    "AppContainer",
    "CameraConfig",
    "PerceptionConfig",
    "RemotePerceptionConfig",
    "SpeakerConfig",
    "load_app_config",
]
