from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from dog_sdk.commands import cmd_stand_toggle
from mission.base import MissionBase
from mission.models import InspectionReading, PickupOutcome, VALID_ZONES
from mission.perception import PerceptionGateway
from runtime.controller import DogController
from runtime.speaker import Speaker


@dataclass(frozen=True)
class MissionConfig:
    obstacle_forward_value: int
    obstacle_turn_value: int
    obstacle_timeout_sec: float
    inspection_target_count: int
    max_drop_count: int


class NationalStageMission(MissionBase):
    """
    国赛流程：
    1) 避障通过
    2) 巡检 A/B/C/D 仪表状态并播报
    3) 将红色长条投放到异常区域字母箱
    """

    def __init__(
        self,
        controller: DogController,
        perception: PerceptionGateway,
        speaker: Speaker,
        cfg: MissionConfig,
    ) -> None:
        self._controller = controller
        self._perception = perception
        self._speaker = speaker
        self._cfg = cfg

        self._phase = "INIT"
        self._phase_enter_ts = 0.0

        self._inspection_by_zone: dict[str, InspectionReading] = {}
        self._delivery_queue: deque[str] = deque()
        self._drop_count = 0
        self._completed = False
        self._failed = False

    def start(self) -> None:
        self._controller.set_walk_gait()
        self._controller.enable_obstacle_avoidance()
        self._controller.send(cmd_stand_toggle())
        self._enter_phase("OBSTACLE")

    def tick(self) -> None:
        if self._completed or self._failed:
            return
        if self._phase == "OBSTACLE":
            self._tick_obstacle()
        elif self._phase == "INSPECTION":
            self._tick_inspection()
        elif self._phase == "PICKUP":
            self._tick_pickup()

    def stop(self) -> None:
        self._controller.stop_motion()
        self._phase = "STOPPED"

    @property
    def is_finished(self) -> bool:
        return self._completed or self._failed

    def _enter_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_enter_ts = time.time()
        logging.info("mission phase => %s", phase)

    def _tick_obstacle(self) -> None:
        elapsed = time.time() - self._phase_enter_ts
        if elapsed > self._cfg.obstacle_timeout_sec:
            raise RuntimeError("避障阶段超时，请检查定位/避障感知链路")
        self._controller.set_motion(
            forward=self._cfg.obstacle_forward_value,
            turn=self._cfg.obstacle_turn_value,
        )
        if self._perception.obstacle_cleared():
            self._controller.stop_motion()
            self._speaker.say("已通过障碍区域，进入巡检识别。")
            self._enter_phase("INSPECTION")

    def _tick_inspection(self) -> None:
        for reading in self._perception.poll_inspection():
            if reading.zone not in VALID_ZONES:
                continue
            if reading.zone in self._inspection_by_zone:
                continue
            self._inspection_by_zone[reading.zone] = reading
            self._speaker.say(reading.broadcast_text())

        if len(self._inspection_by_zone) >= self._cfg.inspection_target_count:
            abnormal = sorted(
                zone
                for zone, item in self._inspection_by_zone.items()
                if item.status.is_abnormal
            )
            self._delivery_queue = deque(abnormal)
            if not self._delivery_queue:
                self._speaker.say("巡检完成，全部区域正常，无需抓取红色长条。")
                self._completed = True
                self._enter_phase("DONE")
                return
            self._speaker.say("巡检完成，开始执行异常区域长条抓取投放。")
            self._enter_phase("PICKUP")

    def _tick_pickup(self) -> None:
        if not self._delivery_queue:
            self._speaker.say("长条投放任务完成。")
            self._completed = True
            self._enter_phase("DONE")
            return

        zone = self._delivery_queue[0]
        outcome = self._perception.execute_pickup_for_zone(zone)
        if outcome == PickupOutcome.SUCCESS:
            self._speaker.say(f"已将红色长条投放到{zone}区域。")
            self._delivery_queue.popleft()
            return
        if outcome == PickupOutcome.DROP:
            self._drop_count += 1
            self._speaker.say(f"搬运掉落一次，当前掉落{self._drop_count}次。")
            if self._drop_count >= self._cfg.max_drop_count:
                self._speaker.say("掉落达到上限，任务失败。")
                self._failed = True
                self._enter_phase("FAILED")
            return
        if outcome == PickupOutcome.RETRY:
            self._speaker.say(f"{zone}区域抓取失败，准备重试。")
            return
