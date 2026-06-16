"""巡检识别闭环依赖容器。

只组装当前保留范围内的组件：相机接口、远程感知网关、音频文件播报器。
"""

from __future__ import annotations

from app.config import AppConfig
from hardware.camera.interface import CameraGateway
from hardware.speaker.interface import SpeakerGateway
from perception.gateway import PerceptionGateway


class AppContainer:
    """DI 容器：持有巡检识别闭环组件引用。"""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

        self.camera: CameraGateway = _create_camera(cfg)
        self.speaker: SpeakerGateway = _create_speaker(cfg)
        self.perception: PerceptionGateway = _create_perception(cfg)


# ── 工厂函数 ────────────────────────────────────────────


def _create_camera(cfg: AppConfig) -> CameraGateway:
    driver = cfg.camera.driver
    if driver == "mock":
        from hardware.camera.interface import MockCamera
        return MockCamera(cfg.camera)
    if driver == "realsense":
        raise NotImplementedError("RealSense camera not yet implemented")
    raise ValueError(f"未知相机驱动: {driver}")


def _create_speaker(cfg: AppConfig) -> SpeakerGateway:
    if not cfg.speaker.enabled:
        from hardware.speaker.interface import MockSpeaker
        return MockSpeaker(cfg.speaker)
    from hardware.speaker.interface import AudioFileSpeaker
    return AudioFileSpeaker(cfg.speaker)


def _create_perception(cfg: AppConfig) -> PerceptionGateway:
    """根据 perception.driver 创建感知实现。

    - "remote" → RemotePerceptionGateway (外接 NVIDIA 算力板)
    """
    driver = cfg.perception.driver

    if driver == "remote":
        from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway
        rcfg = RemotePerceptionConfig(
            host=cfg.remote_perception.host,
            port=cfg.remote_perception.port,
            timeout_sec=cfg.remote_perception.timeout_sec,
        )
        return RemotePerceptionGateway(rcfg)

    raise ValueError(f"当前清理版本只支持 remote 感知驱动: {driver}")
