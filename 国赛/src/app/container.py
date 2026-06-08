"""依赖注入容器。

按拓扑顺序组装所有组件：
  硬件层 → 感知层 / 导航层 → 任务层

支持通过配置切换真实硬件 / mock 实现。
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import AppConfig
from hardware.arm.interface import ArmGateway
from hardware.camera.interface import CameraGateway
from hardware.speaker.interface import SpeakerGateway
from mission.national_stage import NationalStageMission
from mission.perception import JsonScenarioPerception, PerceptionGateway
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
        # 占位：RealSense 实现
        raise NotImplementedError("RealSense camera not yet implemented")
    raise ValueError(f"未知相机驱动: {driver}")


def _create_arm(cfg: AppConfig) -> ArmGateway:
    driver = cfg.arm.driver
    if driver == "mock" or driver == "":
        from hardware.arm.interface import MockArm

        return MockArm(cfg.arm)
    # 占位：真实机械臂实现
    raise NotImplementedError(f"未知机械臂驱动: {driver}")


def _create_speaker(cfg: AppConfig) -> SpeakerGateway:
    engine = cfg.speaker.engine
    if engine == "mock":
        from hardware.speaker.interface import MockSpeaker

        return MockSpeaker(cfg.speaker)
    if engine == "espeak":
        from hardware.speaker.interface import EspeakSpeaker

        return EspeakSpeaker(cfg.speaker)
    raise ValueError(f"未知语音引擎: {engine}")


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
    """根据配置创建感知层实现。

    规则：如果 scenario_file 非空 → Mock（JSON 仿真）；
         否则根据 perception.driver 创建真实感知实现。
    """
    if cfg.perception.scenario_file:
        from mission.perception import PerceptionConfig as PCfg

        pcfg = PCfg(scenario_file=cfg.perception.scenario_file)
        return JsonScenarioPerception(pcfg)
    # 占位：真实感知实现
    raise NotImplementedError("真实感知层尚未实现，请提供 scenario_file 使用仿真模式")


def _create_navigation(
    cfg: AppConfig,
    camera: CameraGateway,
    dog: DogController,
) -> NavigationGateway:
    """创建导航层实现。

    当前仅提供 mock 实现；后续对接视觉 SLAM / AprilTag。
    """
    from navigation.gateway import MockNavigator

    return MockNavigator(camera, dog)
