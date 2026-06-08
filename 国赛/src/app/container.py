"""依赖注入容器。

按拓扑顺序组装所有组件：
  硬件层 → 感知层 / 导航层 → 任务层

支持通过配置切换 mock / local / remote 实现。
"""

from __future__ import annotations

import logging

from app.config import AppConfig
from hardware.arm.interface import ArmGateway
from hardware.camera.interface import CameraGateway
from hardware.speaker.interface import SpeakerGateway
from mission.national_stage import NationalStageMission
from mission.perception import PerceptionGateway
from navigation.gateway import NavigationGateway
from runtime.controller import DogController, RuntimeConfig


class AppContainer:
    """DI 容器：持有所有组件引用。"""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

        # ── 1. 硬件层 ──
        self.camera: CameraGateway = _create_camera(cfg)
        self.arm: ArmGateway = _create_arm(cfg)
        self.speaker: SpeakerGateway = _create_speaker(cfg)
        self.dog: DogController = _create_dog_controller(cfg)

        # ── 2. 感知层 ──
        self.perception: PerceptionGateway = _create_perception(cfg)

        # ── 3. 导航层 ──
        self.navigation: NavigationGateway = _create_navigation(cfg, self.camera, self.dog)

        # ── 4. 任务层 ──
        self.mission: NationalStageMission = NationalStageMission(
            dog=self.dog,
            perception=self.perception,
            navigation=self.navigation,
            arm=self.arm,
            speaker=self.speaker,
            camera=self.camera,
            cfg=cfg.mission,
        )


# ── 工厂函数 ────────────────────────────────────────────


def _create_camera(cfg: AppConfig) -> CameraGateway:
    driver = cfg.camera.driver
    if driver == "mock":
        from hardware.camera.interface import MockCamera
        return MockCamera(cfg.camera)
    if driver == "realsense":
        raise NotImplementedError("RealSense camera not yet implemented")
    raise ValueError(f"未知相机驱动: {driver}")


def _create_arm(cfg: AppConfig) -> ArmGateway:
    driver = cfg.arm.driver
    if driver == "mock" or driver == "":
        from hardware.arm.interface import MockArm
        return MockArm(cfg.arm)
    raise NotImplementedError(f"未知机械臂驱动: {driver}")


def _create_speaker(cfg: AppConfig) -> SpeakerGateway:
    if not cfg.speaker.enabled:
        from hardware.speaker.interface import MockSpeaker
        return MockSpeaker(cfg.speaker)
    from hardware.speaker.interface import AudioFileSpeaker
    return AudioFileSpeaker(cfg.speaker)


def _create_dog_controller(cfg: AppConfig) -> DogController:
    runtime_cfg = RuntimeConfig(
        robot_ip=cfg.robot.ip,
        robot_command_port=cfg.robot.command_port,
        local_ip=cfg.robot.local_ip,
        local_telemetry_port=cfg.robot.local_telemetry_port,
        heartbeat_hz=cfg.timing.heartbeat_hz,
        main_loop_hz=cfg.timing.main_loop_hz,
        log_telemetry=cfg.log_telemetry,
    )
    return DogController(runtime_cfg)


def _create_perception(cfg: AppConfig) -> PerceptionGateway:
    """根据 perception.driver 创建感知实现。

    - "mock"   → JsonScenarioPerception (JSON 场景仿真)
    - "local"  → LocalPerceptionGateway (本机 CV)
    - "remote" → RemotePerceptionGateway (外接 NVIDIA 算力板)
    """
    driver = cfg.perception.driver

    if driver == "mock":
        from mission.perception import PerceptionConfig as PCfg, JsonScenarioPerception
        pcfg = PCfg(scenario_file=cfg.perception.scenario_file)
        return JsonScenarioPerception(pcfg)

    if driver == "local":
        raise NotImplementedError("LocalPerceptionGateway 尚未实现")

    if driver == "remote":
        from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway
        rcfg = RemotePerceptionConfig(
            host=cfg.remote_perception.host,
            port=cfg.remote_perception.port,
            timeout_sec=cfg.remote_perception.timeout_sec,
        )
        return RemotePerceptionGateway(rcfg)

    raise ValueError(f"未知感知驱动: {driver}")


def _create_navigation(
    cfg: AppConfig,
    camera: CameraGateway,
    dog: DogController,
) -> NavigationGateway:
    from navigation.gateway import MockNavigator
    return MockNavigator(camera, dog)
