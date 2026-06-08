"""国赛任务状态机 —— 10 阶段完整流程。

阶段流程:
  INIT → OBSTACLE_APPROACH → OBSTACLE_DETECT → OBSTACLE_CROSS
    → INSPECTION_NAV → INSPECTION_SCAN → INSPECTION_READ
    → PICKUP_PLAN → PICKUP_NAV → PICKUP_GRAB → PICKUP_TRANSPORT → PICKUP_PLACE
    → DONE / FAILED
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable

from app.config import MissionConfig
from core.exceptions import DropLimitExceededError
from core.types import (
    VALID_ZONES,
    InspectionReading,
    MeterStatus,
    MissionPhase,
    NavigationStatus,
    PickupOutcome,
    Zone,
)
from dog_sdk.commands import cmd_stand_toggle
from hardware.arm.interface import ArmGateway
from hardware.camera.interface import CameraGateway
from hardware.speaker.interface import SpeakerGateway
from mission.base import MissionBase
from mission.perception import PerceptionGateway
from navigation.gateway import NavigationGateway
from runtime.controller import DogController


class NationalStageMission(MissionBase):
    """国赛任务状态机。

    依赖（通过构造函数注入）：
    - DogController       (机器狗运动控制)
    - PerceptionGateway   (感知层)
    - NavigationGateway   (导航层)
    - ArmGateway          (机械臂)
    - SpeakerGateway      (语音播报)
    - MissionConfig       (任务参数)
    """

    def __init__(
        self,
        dog: DogController,
        perception: PerceptionGateway,
        navigation: NavigationGateway,
        arm: ArmGateway,
        speaker: SpeakerGateway,
        camera: CameraGateway,
        cfg: "MissionConfig",
    ) -> None:
        self._dog = dog
        self._perception = perception
        self._navigation = navigation
        self._arm = arm
        self._speaker = speaker
        self._camera = camera
        self._cfg = cfg

        # 阶段状态
        self._phase: MissionPhase = MissionPhase.INIT
        self._phase_enter_ts: float = 0.0

        # 巡检数据
        self._inspection_by_zone: dict[str, InspectionReading] = {}
        self._inspection_cursor: int = 0

        # 抓取队列
        self._delivery_queue: deque[str] = deque()

        # 统计数据
        self._drop_count: int = 0
        self._phase_retries: int = 0

        # 阶段调度表
        self._tick_handlers: dict[MissionPhase, Callable[[], None]] = {
            MissionPhase.OBSTACLE_APPROACH: self._tick_obstacle_approach,
            MissionPhase.OBSTACLE_DETECT: self._tick_obstacle_detect,
            MissionPhase.OBSTACLE_CROSS: self._tick_obstacle_cross,
            MissionPhase.INSPECTION_NAV: self._tick_inspection_nav,
            MissionPhase.INSPECTION_SCAN: self._tick_inspection_scan,
            MissionPhase.INSPECTION_READ: self._tick_inspection_read,
            MissionPhase.PICKUP_PLAN: self._tick_pickup_plan,
            MissionPhase.PICKUP_NAV: self._tick_pickup_nav,
            MissionPhase.PICKUP_GRAB: self._tick_pickup_grab,
            MissionPhase.PICKUP_TRANSPORT: self._tick_pickup_transport,
            MissionPhase.PICKUP_PLACE: self._tick_pickup_place,
        }

    # ── 公共 API ────────────────────────────────────────

    def start(self) -> None:
        self._speaker.say_async("国赛任务启动，进入避障阶段。")
        self._dog.set_walk_gait()
        self._dog.enable_obstacle_avoidance()
        self._dog.send(cmd_stand_toggle())
        self._enter_phase(MissionPhase.OBSTACLE_APPROACH)

    def tick(self) -> None:
        if self._phase.is_terminal:
            return
        handler = self._tick_handlers.get(self._phase)
        if handler:
            handler()

    def stop(self) -> None:
        self._dog.stop_motion()
        self._navigation.reset()
        self._enter_phase(MissionPhase.STOPPED)

    @property
    def is_finished(self) -> bool:
        return self._phase.is_terminal

    # ── 阶段管理 ────────────────────────────────────────

    def _enter_phase(self, phase: MissionPhase) -> None:
        self._phase = phase
        self._phase_enter_ts = time.time()
        self._phase_retries = 0
        logging.info("mission phase => %s", phase.value)

    def _elapsed(self) -> float:
        return time.time() - self._phase_enter_ts

    def _retry_or_fail(self, next_phase: MissionPhase, reason: str) -> bool:
        """重试管理：未超限 → 返回 True（调用者应重试）；超限 → 进入 FAILED。"""
        self._phase_retries += 1
        if self._phase_retries < self._cfg.max_retries:
            logging.warning("重试 %s/%s: %s", self._phase_retries, self._cfg.max_retries, reason)
            return True
        logging.error("重试耗尽: %s", reason)
        self._speaker.say_async(f"任务失败：{reason}")
        self._enter_phase(MissionPhase.FAILED)
        return False

    def _check_timeout(self, timeout_sec: float, error_msg: str) -> bool:
        """检查阶段超时。返回 True 表示已超时并进入 FAILED。"""
        if self._elapsed() > timeout_sec:
            self._speaker.say_async(error_msg)
            self._enter_phase(MissionPhase.FAILED)
            return True
        return False

    # ── Phase 1: OBSTACLE_APPROACH ─────────────────────

    def _tick_obstacle_approach(self) -> None:
        """朝障碍区前进，直到视野中出现锥桶或已通过障碍区。"""
        if self._check_timeout(self._cfg.obstacle_timeout_sec, "避障接近超时"):
            return

        self._dog.set_motion(forward=10000, turn=0)

        # 无相机时直接依赖 obstacle_cleared 判断
        try:
            rgb, depth = self._get_frames()
        except Exception:
            if self._perception.obstacle_cleared():
                self._skip_to_inspection()
            return

        cones = self._perception.detect_cones(rgb, depth)
        if cones:
            self._dog.stop_motion()
            self._speaker.say_async("检测到锥桶，开始规划绕行路径。")
            self._enter_phase(MissionPhase.OBSTACLE_DETECT)
            return

        # 无锥桶 → 检查是否已通过障碍区
        if self._perception.obstacle_cleared():
            self._skip_to_inspection()

    # ── Phase 2: OBSTACLE_DETECT ───────────────────────

    def _tick_obstacle_detect(self) -> None:
        """检测锥桶 3D 位置，计算避障运动指令。"""
        if self._check_timeout(30.0, "锥桶检测超时"):
            return

        try:
            rgb, depth = self._get_frames()
        except Exception:
            return

        cones = self._perception.detect_cones(rgb, depth)
        if not cones:
            # 无锥桶 → 已通过
            self._speaker.say_async("已通过障碍区域，进入巡检识别。")
            self._enter_phase(MissionPhase.INSPECTION_NAV)
            return

        fwd, turn = self._navigation.compute_avoidance(cones)
        self._dog.set_motion(forward=fwd, turn=turn)
        self._enter_phase(MissionPhase.OBSTACLE_CROSS)

    # ── Phase 3: OBSTACLE_CROSS ────────────────────────

    def _tick_obstacle_cross(self) -> None:
        """执行绕行动作并确认通过。"""
        if self._check_timeout(30.0, "通过障碍区超时"):
            return

        if self._perception.obstacle_cleared():
            self._dog.stop_motion()
            self._speaker.say_async("已通过障碍区域，进入巡检识别。")
            self._enter_phase(MissionPhase.INSPECTION_NAV)
            return

        # 仍在绕行中，维持运动指令（由 OBSTACLE_DETECT 设定）
        # 定期更新锥桶检测
        try:
            rgb, depth = self._get_frames()
            cones = self._perception.detect_cones(rgb, depth)
            if cones:
                fwd, turn = self._navigation.compute_avoidance(cones)
                self._dog.set_motion(forward=fwd, turn=turn)
        except Exception:
            pass

    # ── Phase 4: INSPECTION_NAV ────────────────────────

    def _tick_inspection_nav(self) -> None:
        """导航到检测区域。"""
        if self._check_timeout(60.0, "巡检导航超时"):
            return

        status = self._navigation.navigate_to(self._cfg.inspection_target)
        if status == NavigationStatus.ARRIVED:
            self._speaker.say_async("已到达检测区域，开始扫描设备。")
            self._enter_phase(MissionPhase.INSPECTION_SCAN)
        elif status == NavigationStatus.BLOCKED:
            self._retry_or_fail(MissionPhase.INSPECTION_NAV, "导航被阻挡")
        elif status == NavigationStatus.LOST:
            if not self._retry_or_fail(MissionPhase.INSPECTION_NAV, "导航定位丢失"):
                pass

    # ── Phase 5: INSPECTION_SCAN ───────────────────────

    def _tick_inspection_scan(self) -> None:
        """扫描寻找配电柜/变压器并识别区域字母。"""
        if self._check_timeout(60.0, "设备扫描超时"):
            return

        try:
            rgb, _depth = self._get_frames()
        except Exception:
            return

        equipment_list = self._perception.detect_equipment(rgb)

        for eq in equipment_list:
            # 优先使用 detect_equipment 预填充的 zone_letter
            if eq.zone_letter and eq.zone_confidence >= self._cfg.inspection_confidence:
                letter = eq.zone_letter
                conf = eq.zone_confidence
            else:
                letter, conf = self._perception.read_zone_letter(rgb, eq.bbox)
            if conf < self._cfg.inspection_confidence or letter not in VALID_ZONES:
                continue
            if letter in self._inspection_by_zone:
                continue
            # 暂存（不含仪表读数，等 INSPECTION_READ 阶段读取）
            self._inspection_by_zone[letter] = InspectionReading(
                zone=Zone(letter),
                meter_status=MeterStatus.NORMAL,  # 占位
                confidence=conf,
            )

        if len(self._inspection_by_zone) >= self._cfg.inspection_target_count:
            self._speaker.say_async("设备扫描完成，开始读取仪表。")
            self._inspection_cursor = 0
            self._enter_phase(MissionPhase.INSPECTION_READ)

    # ── Phase 6: INSPECTION_READ ───────────────────────

    def _tick_inspection_read(self) -> None:
        """逐区域读取仪表值并语音播报。"""
        if self._check_timeout(90.0, "仪表读取超时"):
            return

        # 按 A/B/C/D 顺序处理
        zone_order = sorted(self._inspection_by_zone.keys())
        if self._inspection_cursor >= len(zone_order):
            # 全部读完 → 判断异常并进入抓取
            self._finish_inspection()
            return

        zone = zone_order[self._inspection_cursor]

        # 读取仪表（调用感知层）
        readings = self._perception.poll_inspection()
        if readings:
            reading = readings[0]
            if reading.zone.value == zone:
                self._inspection_by_zone[zone] = reading
                self._speaker.say_async(reading.broadcast_text())
                self._inspection_cursor += 1
            else:
                # 区域不匹配 → 重试
                if not self._retry_or_fail(MissionPhase.INSPECTION_READ, f"{zone}区读数为{reading.zone.value}区，不匹配"):
                    pass

    def _finish_inspection(self) -> None:
        """巡检收尾：构建异常队列，决定是否进入抓取。"""
        abnormal = sorted(
            zone
            for zone, item in self._inspection_by_zone.items()
            if item.meter_status.is_abnormal
        )
        self._delivery_queue = deque(abnormal)

        if not self._delivery_queue:
            self._speaker.say_async("巡检完成，全部区域正常，无需抓取红色长条。")
            self._enter_phase(MissionPhase.DONE)
            return

        zones_str = "、".join(self._delivery_queue)
        self._speaker.say_async(f"巡检完成，{zones_str}区域异常，开始执行长条抓取投放。")
        self._enter_phase(MissionPhase.PICKUP_PLAN)

    # ── Phase 7: PICKUP_PLAN ───────────────────────────

    def _tick_pickup_plan(self) -> None:
        """规划抓取顺序和路径。"""
        if not self._delivery_queue:
            self._speaker.say_async("长条投放任务完成。")
            self._enter_phase(MissionPhase.DONE)
            return

        # 当前实现：直接按队列顺序逐个处理
        self._enter_phase(MissionPhase.PICKUP_NAV)

    # ── Phase 8: PICKUP_NAV ────────────────────────────

    def _tick_pickup_nav(self) -> None:
        """导航到红色长条位置。"""
        if self._check_timeout(60.0, "抓取导航超时"):
            return

        target_zone = self._delivery_queue[0]
        target = self._cfg.pickup_position_for_zone.get(target_zone, (1.0, 0.0))

        status = self._navigation.navigate_to(target)
        if status == NavigationStatus.ARRIVED:
            self._speaker.say_async(f"已到达{target_zone}区长条位置，准备抓取。")
            self._enter_phase(MissionPhase.PICKUP_GRAB)
        elif status == NavigationStatus.BLOCKED or status == NavigationStatus.LOST:
            self._retry_or_fail(MissionPhase.PICKUP_NAV, f"导航到{target_zone}区失败")

    # ── Phase 9: PICKUP_GRAB ───────────────────────────

    def _tick_pickup_grab(self) -> None:
        """视觉定位红色长条 → 机械臂抓取。"""
        if self._check_timeout(30.0, "抓取超时"):
            return

        target_zone = self._delivery_queue[0]
        outcome_str = self._perception.execute_pickup_for_zone(target_zone)
        outcome = PickupOutcome(outcome_str)

        if outcome == PickupOutcome.SUCCESS:
            self._speaker.say_async(f"成功抓取{target_zone}区红色长条，开始搬运。")
            self._enter_phase(MissionPhase.PICKUP_TRANSPORT)
        elif outcome == PickupOutcome.DROP:
            self._drop_count += 1
            self._speaker.say_async(f"搬运掉落，当前掉落{self._drop_count}次。")
            if self._drop_count >= self._cfg.max_drop_count:
                self._speaker.say_async("掉落达到上限，任务失败。")
                self._enter_phase(MissionPhase.FAILED)
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)  # 重新抓取
        elif outcome == PickupOutcome.RETRY:
            if self._retry_or_fail(MissionPhase.PICKUP_GRAB, f"{target_zone}区抓取失败，重试"):
                self._arm.move_home()  # 复位后重试
        elif outcome == PickupOutcome.ARM_ERROR:
            self._retry_or_fail(MissionPhase.PICKUP_GRAB, f"{target_zone}区机械臂异常")

    # ── Phase 10: PICKUP_TRANSPORT ─────────────────────

    def _tick_pickup_transport(self) -> None:
        """搬运长条到目标区域，持续监控掉落。"""
        if self._check_timeout(60.0, "搬运超时"):
            return

        target_zone = self._delivery_queue[0]
        target = self._cfg.placement_position_for_zone.get(target_zone, (2.0, 0.0))

        # 掉落检测
        try:
            rgb, _depth = self._get_frames()
            if self._perception.check_drop(rgb):
                self._drop_count += 1
                self._speaker.say_async(f"搬运掉落，当前掉落{self._drop_count}次。")
                if self._drop_count >= self._cfg.max_drop_count:
                    self._speaker.say_async("掉落达到上限，任务失败。")
                    self._enter_phase(MissionPhase.FAILED)
                    return
                # 回到抓取阶段
                self._enter_phase(MissionPhase.PICKUP_GRAB)
                return
        except Exception:
            pass  # 无视觉时跳过掉落检测

        status = self._navigation.navigate_to(target)
        if status == NavigationStatus.ARRIVED:
            self._speaker.say_async(f"已到达{target_zone}区域，准备放置。")
            self._enter_phase(MissionPhase.PICKUP_PLACE)
        elif status == NavigationStatus.BLOCKED or status == NavigationStatus.LOST:
            self._retry_or_fail(MissionPhase.PICKUP_TRANSPORT, f"搬运到{target_zone}区失败")

    # ── Phase 11: PICKUP_PLACE ─────────────────────────

    def _tick_pickup_place(self) -> None:
        """机械臂放置长条到对应字母箱。"""
        if self._check_timeout(20.0, "放置超时"):
            return

        target_zone = self._delivery_queue[0]

        # 执行放置（与抓取共用接口，zone 决定目标箱位置）
        outcome_str = self._perception.execute_pickup_for_zone(target_zone)
        outcome = PickupOutcome(outcome_str)

        if outcome == PickupOutcome.SUCCESS:
            self._speaker.say_async(f"已将红色长条投放到{target_zone}区域。")
            self._delivery_queue.popleft()
            self._arm.move_home()
            # 处理下一个（如有）
            self._enter_phase(MissionPhase.PICKUP_PLAN)
        elif outcome == PickupOutcome.DROP:
            self._drop_count += 1
            if self._drop_count >= self._cfg.max_drop_count:
                self._speaker.say_async("掉落达到上限，任务失败。")
                self._enter_phase(MissionPhase.FAILED)
                return
            self._speaker.say_async(f"放置掉落，当前掉落{self._drop_count}次。")
            self._enter_phase(MissionPhase.PICKUP_GRAB)
        elif outcome == PickupOutcome.RETRY:
            self._retry_or_fail(MissionPhase.PICKUP_PLACE, f"{target_zone}区放置失败，重试")
        elif outcome == PickupOutcome.ARM_ERROR:
            self._retry_or_fail(MissionPhase.PICKUP_PLACE, f"{target_zone}区放置时机械臂异常")

    # ── 辅助方法 ────────────────────────────────────────

    def _skip_to_inspection(self) -> None:
        """跳过避障阶段，直接进入巡检。"""
        self._dog.stop_motion()
        self._speaker.say_async("已通过障碍区域，进入巡检识别。")
        self._enter_phase(MissionPhase.INSPECTION_NAV)

    def _get_frames(self) -> tuple:
        """获取对齐后的 RGB + Depth 帧。"""
        if not self._camera.is_running:
            self._camera.start()
        return self._camera.get_aligned_frames()


