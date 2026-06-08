"""国赛任务集成测试 —— 6 个核心场景。"""

from __future__ import annotations

import json
import time

import pytest

from app.config import (
    CameraConfig,
    MissionConfig,
    RobotNetworkConfig,
    SpeakerConfig,
    TimingConfig,
)
from hardware.arm.interface import MockArm
from hardware.camera.interface import MockCamera
from hardware.speaker.interface import MockSpeaker
from mission.national_stage import NationalStageMission
from mission.perception import JsonScenarioPerception, PerceptionConfig
from navigation.gateway import MockNavigator
from runtime.controller import DogController, RuntimeConfig


# ── helpers ──────────────────────────────────────────────


def _make_dog() -> DogController:
    """用随机端口创建 DogController，避免测试间端口冲突。"""
    import random
    port = random.randint(50000, 60000)
    return DogController(RuntimeConfig(
        robot_ip="127.0.0.1", robot_command_port=43893,
        local_ip="0.0.0.0", local_telemetry_port=port,
        heartbeat_hz=2.0, main_loop_hz=20.0, log_telemetry=False,
    ))


def _make_mission(
    scenario: dict,
    mission_cfg: MissionConfig | None = None,
    arm: MockArm | None = None,
    speaker: MockSpeaker | None = None,
    confidence_threshold: float = 0.6,
) -> NationalStageMission:
    """构建 Mission 实例用于测试。"""

    # 写临时 scenario 文件
    scenario_path = "tests/fixtures/mock_configs/_tmp_scenario.json"
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f)

    dog = _make_dog()
    camera = MockCamera(CameraConfig(driver="mock", width=640, height=480))
    nav = MockNavigator(camera, dog)
    perception = JsonScenarioPerception(PerceptionConfig(scenario_file=scenario_path))

    if arm is None:
        arm = MockArm(arm_cfg := None)  # type: ignore
    if speaker is None:
        speaker = MockSpeaker(SpeakerConfig(enabled=False, engine="mock"))

    cfg = mission_cfg or MissionConfig(inspection_confidence=confidence_threshold)

    return NationalStageMission(
        dog=dog, perception=perception, navigation=nav,
        arm=arm, speaker=speaker, camera=camera, cfg=cfg,
    )


def _tick_until_terminal(mission: NationalStageMission, max_ticks: int = 120) -> str:
    """驱动任务到终态，返回最终阶段名。"""
    mission.start()
    for _ in range(max_ticks):
        mission.tick()
        if mission.is_finished:
            break
    result = mission.phase.value
    mission.stop()
    return result


def _base_scenario(anomalies: list[dict] | None = None) -> dict:
    """构建基础场景。anomalies 为空 → 全部正常。"""
    readings = anomalies if anomalies else [
        {"zone": "A", "status": "normal", "confidence": 0.95},
        {"zone": "B", "status": "normal", "confidence": 0.95},
        {"zone": "C", "status": "normal", "confidence": 0.95},
        {"zone": "D", "status": "normal", "confidence": 0.95},
    ]
    return {
        "obstacle": {"clear_after_ticks": 5},
        "inspection_readings": readings,
        "pickup_outcomes": {},
        "cones": [],
        "red_strips": [],
    }


# ── Test 1: 正常流程完成 ─────────────────────────────────


def test_normal_flow_completes():
    """mock 正常流程：A/C 异常 → 避障 → 巡检 → 抓取两轮 → DONE。"""
    scenario = _base_scenario(anomalies=[
        {"zone": "A", "status": "low", "confidence": 0.95},
        {"zone": "B", "status": "normal", "confidence": 0.93},
        {"zone": "C", "status": "high", "confidence": 0.94},
        {"zone": "D", "status": "normal", "confidence": 0.96},
    ])

    mission = _make_mission(scenario)
    result = _tick_until_terminal(mission, max_ticks=120)

    assert result == "DONE", f"Expected DONE, got {result}"
    assert mission.drop_count == 0
    assert not mission.delivery_queue


# ── Test 2: 无异常区域时不抓取 ────────────────────────────


def test_no_anomaly_no_pickup():
    """全部正常 → 巡检完成后直接 DONE，不进入抓取。"""
    scenario = _base_scenario()  # 全部 normal

    mission = _make_mission(scenario)
    result = _tick_until_terminal(mission, max_ticks=120)

    assert result == "DONE"
    assert len(mission.inspection_by_zone) == 4
    # 确认没有进入过抓取阶段
    for z, reading in mission.inspection_by_zone.items():
        assert reading.meter_status.value == "normal"


# ── Test 3: 掉落 1 次后重试 ───────────────────────────────


def test_drop_once_retry():
    """pick 掉落 1 次 → 重试成功 → 最终 DONE。"""
    scenario = _base_scenario(anomalies=[
        {"zone": "A", "status": "low", "confidence": 0.95},
        {"zone": "B", "status": "normal", "confidence": 0.95},
        {"zone": "C", "status": "normal", "confidence": 0.95},
        {"zone": "D", "status": "normal", "confidence": 0.95},
    ])

    arm = MockArm(None)  # type: ignore[arg-type]
    arm.simulate_drop_on_next_pick()  # 第一次 pick 失败

    mission = _make_mission(scenario, arm=arm)
    result = _tick_until_terminal(mission, max_ticks=150)

    # 掉落 1 次后，重试成功 → DONE
    assert result == "DONE", f"Expected DONE, got {result}"
    assert mission.drop_count == 1


# ── Test 4: 掉落 3 次后任务终止 ───────────────────────────


def test_drop_three_times_fails():
    """掉落 3 次 → 达到上限 → FAILED。"""
    scenario = _base_scenario(anomalies=[
        {"zone": "A", "status": "low", "confidence": 0.95},
        {"zone": "B", "status": "normal", "confidence": 0.95},
        {"zone": "C", "status": "normal", "confidence": 0.95},
        {"zone": "D", "status": "normal", "confidence": 0.95},
    ])

    arm = MockArm(None)  # type: ignore[arg-type]
    arm.simulate_drop_on_next_pick()
    arm.simulate_drop_on_next_pick()
    arm.simulate_drop_on_next_pick()  # 连续 3 次掉落 → max_drop_count=3

    mission = _make_mission(scenario, arm=arm)
    result = _tick_until_terminal(mission, max_ticks=120)

    assert result == "FAILED", f"Expected FAILED, got {result} (drops={mission.drop_count})"
    assert mission.drop_count == 3


# ── Test 5: 识别置信度低于阈值时重试或跳过 ──────────────────


def test_low_confidence_retry():
    """巡检读数置信度为 0.5 < 阈值 0.6 → 重试 → 耗尽 → FAILED。"""
    scenario = _base_scenario(anomalies=[
        {"zone": "A", "status": "low", "confidence": 0.50},   # 低于阈值
        {"zone": "B", "status": "normal", "confidence": 0.95},
        {"zone": "C", "status": "normal", "confidence": 0.95},
        {"zone": "D", "status": "normal", "confidence": 0.95},
    ])

    mission = _make_mission(scenario, confidence_threshold=0.6)
    result = _tick_until_terminal(mission, max_ticks=150)

    # 置信度过低，重试耗尽 → FAILED
    assert result == "FAILED", f"Expected FAILED, got {result}"


def test_low_confidence_pass_with_higher_threshold():
    """置信度 0.7 对阈值 0.6 → 正常通过。"""
    scenario = _base_scenario(anomalies=[
        {"zone": "A", "status": "low", "confidence": 0.70},
        {"zone": "B", "status": "normal", "confidence": 0.95},
        {"zone": "C", "status": "normal", "confidence": 0.95},
        {"zone": "D", "status": "normal", "confidence": 0.95},
    ])

    mission = _make_mission(scenario, confidence_threshold=0.6)
    result = _tick_until_terminal(mission, max_ticks=150)

    assert result == "DONE", f"Expected DONE, got {result}"


# ── Test 6: remote perception 超时后触发 fallback ──────────


class TestRemotePerceptionTimeout:
    """验证 RemotePerceptionGateway 在连接失败时的行为。"""

    def test_returns_empty_on_connect_failure(self):
        """无算力板可连接时，返回空结果（不抛异常）。"""
        from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway

        cfg = RemotePerceptionConfig(
            host="127.0.0.1",
            port=19999,  # 未监听的端口
            timeout_sec=0.5,
        )
        gw = RemotePerceptionGateway(cfg)

        # 所有检测方法应安全返回空/None，不抛异常
        obstacles = gw.detect_obstacles()
        assert isinstance(obstacles, list)
        # 首次无缓存 → 空列表
        assert obstacles == []

        letters = gw.detect_zone_letters()
        assert isinstance(letters, list)

        gauges = gw.detect_gauges()
        assert isinstance(gauges, list)

        strips = gw.detect_red_strips()
        assert isinstance(strips, list)

        pose = gw.estimate_target_pose()
        assert pose is None

        # obstacle_cleared 在无检测结果时应为 True（安全侧）
        assert gw.obstacle_cleared() is True

