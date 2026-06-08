"""国赛任务状态机 —— 10 阶段完整流程。

职责边界：
  - PerceptionGateway  → 纯检测（不控制任何硬件）
  - ArmGateway         → 机械臂原子动作（pick/lift/place）
  - NavigationGateway  → 导航到目标点
  - DogController      → 四足运动控制
  - SpeakerGateway     → 播放预录音频
  - Mission            → 状态机调度（唯一决策者）
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable

from app.config import MissionConfig
from core.types import (
    VALID_ZONES,
    InspectionReading,
    MeterStatus,
    MissionPhase,
    NavigationStatus,
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
    """国赛任务状态机。"""

    def __init__(
        self,
        dog: DogController,
        perception: PerceptionGateway,
        navigation: NavigationGateway,
        arm: ArmGateway,
        speaker: SpeakerGateway,
        camera: CameraGateway,
        cfg: MissionConfig,
    ) -> None:
        self._dog = dog
        self._perception = perception
        self._navigation = navigation
        self._arm = arm
        self._speaker = speaker
        self._camera = camera
        self._cfg = cfg

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

        # 阶段调度
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
        self._speaker.play("task_start")
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
        self._phase_retries += 1
        if self._phase_retries < self._cfg.max_retries:
            logging.warning("重试 %s/%s: %s", self._phase_retries, self._cfg.max_retries, reason)
            return True
        logging.error("重试耗尽: %s", reason)
        self._speaker.play("task_failed")
        self._enter_phase(MissionPhase.FAILED)
        return False

    def _check_timeout(self, timeout_sec: float, error_msg: str) -> bool:
        if self._elapsed() > timeout_sec:
            logging.error(error_msg)
            self._speaker.play("task_failed")
            self._enter_phase(MissionPhase.FAILED)
            return True
        return False

    def _record_drop(self) -> bool:
        """记录一次掉落。返回 True 表示已达上限 → 任务失败。"""
        self._drop_count += 1
        key = f"drop_warning_{self._drop_count}" if self._drop_count <= 3 else "drop_warning_3"
        self._speaker.play(key)
        if self._drop_count >= self._cfg.max_drop_count:
            self._speaker.play("drop_limit")
            self._enter_phase(MissionPhase.FAILED)
            return True
        return False

    # ── Phase 1-3: 避障（保持原逻辑）────────────────────

    def _tick_obstacle_approach(self) -> None:
        if self._check_timeout(self._cfg.obstacle_timeout_sec, "避障接近超时"):
            return
        self._dog.set_motion(forward=10000, turn=0)
        try:
            rgb, _depth = self._get_frames()
        except Exception:
            if self._perception.obstacle_cleared():
                self._skip_to_inspection()
            return

        cones = self._perception.detect_obstacles(rgb)
        if cones:
            self._dog.stop_motion()
            self._speaker.play("obstacle_detected")
            self._enter_phase(MissionPhase.OBSTACLE_DETECT)
            return
        if self._perception.obstacle_cleared():
            self._skip_to_inspection()

    def _tick_obstacle_detect(self) -> None:
        if self._check_timeout(30.0, "锥桶检测超时"):
            return
        try:
            rgb, _depth = self._get_frames()
        except Exception:
            return
        cones = self._perception.detect_obstacles(rgb)
        if not cones:
            self._skip_to_inspection()
            return
        fwd, turn = self._navigation.compute_avoidance(cones)
        self._dog.set_motion(forward=fwd, turn=turn)
        self._enter_phase(MissionPhase.OBSTACLE_CROSS)

    def _tick_obstacle_cross(self) -> None:
        if self._check_timeout(30.0, "通过障碍区超时"):
            return
        if self._perception.obstacle_cleared():
            self._dog.stop_motion()
            self._skip_to_inspection()
            return
        try:
            rgb, _depth = self._get_frames()
            cones = self._perception.detect_obstacles(rgb)
            if cones:
                fwd, turn = self._navigation.compute_avoidance(cones)
                self._dog.set_motion(forward=fwd, turn=turn)
        except Exception:
            pass

    # ── Phase 4-6: 巡检（使用新接口）────────────────────

    def _tick_inspection_nav(self) -> None:
        if self._check_timeout(60.0, "巡检导航超时"):
            return
        status = self._navigation.navigate_to(self._cfg.inspection_target)
        if status == NavigationStatus.ARRIVED:
            self._speaker.play("inspection_start")
            self._enter_phase(MissionPhase.INSPECTION_SCAN)
        elif status in (NavigationStatus.BLOCKED, NavigationStatus.LOST):
            self._retry_or_fail(MissionPhase.INSPECTION_NAV, "巡检导航失败")

    def _tick_inspection_scan(self) -> None:
        if self._check_timeout(60.0, "设备扫描超时"):
            return
        try:
            rgb, _depth = self._get_frames()
        except Exception:
            return

        # 使用新接口：detect_zone_letters
        zone_results = self._perception.detect_zone_letters(rgb)
        for zr in zone_results:
            if zr.confidence < self._cfg.inspection_confidence:
                continue
            letter = zr.zone.value
            if letter in self._inspection_by_zone:
                continue
            self._inspection_by_zone[letter] = InspectionReading(
                zone=zr.zone,
                meter_status=MeterStatus.NORMAL,  # 占位，INSPECTION_READ 阶段填充
                confidence=zr.confidence,
            )

        if len(self._inspection_by_zone) >= self._cfg.inspection_target_count:
            self._inspection_cursor = 0
            self._enter_phase(MissionPhase.INSPECTION_READ)

    def _tick_inspection_read(self) -> None:
        if self._check_timeout(90.0, "仪表读取超时"):
            return
        zone_order = sorted(self._inspection_by_zone.keys())
        if self._inspection_cursor >= len(zone_order):
            self._finish_inspection()
            return

        zone = zone_order[self._inspection_cursor]
        readings = self._perception.poll_inspection()
        if readings:
            reading = readings[0]
            if reading.zone.value == zone:
                if reading.confidence < self._cfg.inspection_confidence:
                    if not self._retry_or_fail(MissionPhase.INSPECTION_READ,
                                               f"{zone}区置信度{reading.confidence:.2f}低于阈值"):
                        pass
                    return
                self._inspection_by_zone[zone] = reading
                # 播报预录音频: "A_low", "B_normal" 等
                audio_key = f"{zone}_{reading.meter_status.value}"
                self._speaker.play(audio_key)
                self._inspection_cursor += 1
            else:
                self._retry_or_fail(MissionPhase.INSPECTION_READ,
                                    f"{zone}区读数为{reading.zone.value}区，不匹配")

    def _finish_inspection(self) -> None:
        abnormal = sorted(
            z for z, item in self._inspection_by_zone.items()
            if item.meter_status.is_abnormal
        )
        self._delivery_queue = deque(abnormal)

        if not self._delivery_queue:
            self._speaker.play("all_normal")
            self._enter_phase(MissionPhase.DONE)
            return

        self._speaker.play("inspection_complete")
        self._speaker.play("pickup_start")
        self._enter_phase(MissionPhase.PICKUP_PLAN)

    # ── Phase 7: PICKUP_PLAN ───────────────────────────

    def _tick_pickup_plan(self) -> None:
        if not self._delivery_queue:
            self._speaker.play("task_complete")
            self._enter_phase(MissionPhase.DONE)
            return
        self._enter_phase(MissionPhase.PICKUP_NAV)

    # ── Phase 8: PICKUP_NAV（导航到长条抓取位置）───────

    def _tick_pickup_nav(self) -> None:
        if self._check_timeout(60.0, "抓取导航超时"):
            return
        target_zone = self._delivery_queue[0]
        target = self._cfg.pickup_position_for_zone.get(target_zone, (1.0, 0.0))
        status = self._navigation.navigate_to(target)
        if status == NavigationStatus.ARRIVED:
            self._enter_phase(MissionPhase.PICKUP_GRAB)
        elif status in (NavigationStatus.BLOCKED, NavigationStatus.LOST):
            self._retry_or_fail(MissionPhase.PICKUP_NAV, f"导航到{target_zone}区失败")

    # ── Phase 9: PICKUP_GRAB（视觉定位 + 机械臂抓取）────

    def _tick_pickup_grab(self) -> None:
        """Perception 定位 → Arm.pick() → Arm.lift() → 确认抓取。"""
        if self._check_timeout(30.0, "抓取超时"):
            return

        target_zone = self._delivery_queue[0]

        # Step 1: 视觉定位目标
        strips = self._perception.detect_red_strips()
        if not strips:
            if self._retry_or_fail(MissionPhase.PICKUP_GRAB, "未检测到红色长条"):
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)
            return

        # Step 2: 估计 3D 位姿
        target_pose = self._perception.estimate_target_pose()
        if target_pose is None or target_pose.confidence < self._cfg.inspection_confidence:
            if self._retry_or_fail(MissionPhase.PICKUP_GRAB, "目标位姿置信度过低"):
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)
            return

        # Step 3: 机械臂抓取
        success = self._arm.pick(
            ArmPose(x=target_pose.x, y=target_pose.y, z=target_pose.z,
                    roll=target_pose.roll, pitch=target_pose.pitch, yaw=target_pose.yaw)
        )
        if not success:
            if self._record_drop():
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)
            return

        # Step 4: 抬起
        self._arm.lift()

        # Step 5: 确认
        if not self._arm.has_object:
            if self._record_drop():
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)
            return

        self._speaker.play(f"pickup_success_{target_zone}")
        self._enter_phase(MissionPhase.PICKUP_TRANSPORT)

    # ── Phase 10: PICKUP_TRANSPORT（导航搬运 + 掉落监控）─

    def _tick_pickup_transport(self) -> None:
        if self._check_timeout(60.0, "搬运超时"):
            return

        target_zone = self._delivery_queue[0]
        target = self._cfg.placement_position_for_zone.get(target_zone, (2.0, 0.0))

        # 掉落监控：每次 tick 检查夹爪状态
        if self._arm.is_connected and not self._arm.has_object:
            if self._record_drop():
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)
            return

        status = self._navigation.navigate_to(target)
        if status == NavigationStatus.ARRIVED:
            self._enter_phase(MissionPhase.PICKUP_PLACE)
        elif status in (NavigationStatus.BLOCKED, NavigationStatus.LOST):
            self._retry_or_fail(MissionPhase.PICKUP_TRANSPORT, f"搬运到{target_zone}区失败")

    # ── Phase 11: PICKUP_PLACE（机械臂放置）─────────────

    def _tick_pickup_place(self) -> None:
        """Arm.place() → 确认 → 下一个/完成。"""
        if self._check_timeout(20.0, "放置超时"):
            return

        target_zone = self._delivery_queue[0]
        target_coord = self._cfg.placement_position_for_zone.get(target_zone, (2.0, 0.0))

        # 机械臂放置
        place_pose = ArmPose(x=target_coord[0], y=target_coord[1], z=0.05)
        placed = self._arm.place(place_pose)

        if not placed:
            if self._record_drop():
                return
            self._enter_phase(MissionPhase.PICKUP_GRAB)
            return

        self._arm.move_home()
        self._speaker.play(f"pickup_success_{target_zone}")
        self._delivery_queue.popleft()

        # 处理下一个
        self._enter_phase(MissionPhase.PICKUP_PLAN)

    # ── 辅助 ────────────────────────────────────────────

    def _skip_to_inspection(self) -> None:
        self._dog.stop_motion()
        self._speaker.play("obstacle_cleared")
        self._enter_phase(MissionPhase.INSPECTION_NAV)

    def _get_frames(self) -> tuple:
        if not self._camera.is_running:
            self._camera.start()
        return self._camera.get_aligned_frames()

    # Public test accessors
    @property
    def phase(self) -> MissionPhase:
        return self._phase

    @property
    def drop_count(self) -> int:
        return self._drop_count

    @property
    def delivery_queue(self) -> list[str]:
        return list(self._delivery_queue)

    @property
    def inspection_by_zone(self) -> dict:
        return dict(self._inspection_by_zone)


# 类型别名，避免循环导入
from core.types import ArmPose
