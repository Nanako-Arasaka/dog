from __future__ import annotations

import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

from mission.national_stage import MissionConfig, NationalStageMission
from mission.perception import JsonScenarioPerception, PerceptionConfig
from runtime.controller import DogController, RuntimeConfig
from runtime.speaker import Speaker, SpeakerConfig


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    mission: MissionConfig
    perception: PerceptionConfig
    speaker: SpeakerConfig


def load_config(path: Path) -> AppConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    runtime = RuntimeConfig(
        robot_ip=data["robot_ip"],
        robot_command_port=int(data["robot_command_port"]),
        local_ip=data["local_ip"],
        local_telemetry_port=int(data["local_telemetry_port"]),
        heartbeat_hz=float(data["heartbeat_hz"]),
        main_loop_hz=float(data["main_loop_hz"]),
        log_telemetry=bool(data["log_telemetry"]),
    )
    mission_data = data.get("mission", {})
    mission = MissionConfig(
        obstacle_forward_value=int(mission_data.get("obstacle_forward_value", 12000)),
        obstacle_turn_value=int(mission_data.get("obstacle_turn_value", 0)),
        obstacle_timeout_sec=float(mission_data.get("obstacle_timeout_sec", 90.0)),
        inspection_target_count=int(mission_data.get("inspection_target_count", 4)),
        max_drop_count=int(mission_data.get("max_drop_count", 3)),
    )
    root = path.parent.parent
    scenario_rel = data.get("scenario_file", "config/scenario_mock.json")
    scenario_file = str((root / scenario_rel).resolve())
    perception = PerceptionConfig(scenario_file=scenario_file)
    speaker_data = data.get("speaker", {})
    speaker = SpeakerConfig(
        enabled=bool(speaker_data.get("enabled", False)),
        command_template=str(speaker_data.get("command_template", 'echo "{text}"')),
    )
    return AppConfig(runtime=runtime, mission=mission, perception=perception, speaker=speaker)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "robot_config.json"
    cfg = load_config(cfg_path)
    controller = DogController(cfg.runtime)
    perception = JsonScenarioPerception(cfg.perception)
    speaker = Speaker(cfg.speaker)
    mission = NationalStageMission(
        controller=controller,
        perception=perception,
        speaker=speaker,
        cfg=cfg.mission,
    )

    running = True

    def _stop_handler(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    controller.start_background_loops()
    mission.start()
    try:
        while running and not mission.is_finished:
            mission.tick()
            controller.sleep_for_main_tick()
    finally:
        mission.stop()
        controller.stop_background_loops()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
