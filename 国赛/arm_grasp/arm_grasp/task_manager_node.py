#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final national-competition FSM.

This node is intentionally a coordinator. It does not implement SLAM, cone
avoidance, inspection, or arm IK; it only switches phases and calls existing
modules through topics/services.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Bool, String


class State(str, Enum):
    WAIT_LOCALIZATION = "WAIT_LOCALIZATION"
    GO_START = "GO_START"
    GO_OBSTACLE_ENTRY = "GO_OBSTACLE_ENTRY"
    OBSTACLE_ZONE = "OBSTACLE_ZONE"
    GO_INSPECTION = "GO_INSPECTION"
    WAIT_INSPECTION = "WAIT_INSPECTION"
    GO_PICK = "GO_PICK"
    GRASP = "GRASP"
    CENTER = "CENTERING"          # 底座微调中 (边缘修正 / 空抓重试旋转)
    VERIFY = "VERIFYING"          # 抓取后二次视觉确认
    GO_PLACE = "GO_PLACE"
    PLACE = "PLACE"
    GO_FINISH = "GO_FINISH"
    DONE = "DONE"
    ERROR = "ERROR"


# 抓取闭环策略常量 (对齐 arm_grasp 源码版 task_manager_node)
GRASP_EDGE_MARGIN = 80        # 像素, cx 距左右边小于此值触发底座微调
GRASP_EDGE_ADJUST = 60        # 像素, 首次边缘微调时的 cx 修正量
GRASP_RETRY_DEG_MIN = 40      # 像素, 空抓重试最小底座旋转量 (~4°)
GRASP_RETRY_DEG_MAX = 150     # 像素, 空抓重试最大底座旋转量 (~15°)
GRASP_Z_TOL = 0.03            # 米, 抓取后 z 变化超过此值视为物体被拎起
MAX_GRASP_RETRIES = 10        # 最多空抓重试次数 (10×~5°≈50°后回 home 放弃)
VISION_ROLLBACK_MAX = 3       # 视觉丢失回退底座最多次数

# 放置区字母确认常量
# 字母识别失败/不匹配的「额外」重试次数（不含初次尝试）。
# 总尝试 = 1 (初次) + PLACEMENT_ZONE_RETRIES (重试) = 4 次。
PLACEMENT_ZONE_RETRIES = 3
PLACEMENT_ZONE_TIMEOUT_SEC = 8.0 # 单次识别等待超时(秒)，超时后按航点兜底放置


@dataclass
class ArmRequest:
    kind: str
    target_zone: str = ""
    attempt: int = 1
    started_at: float = 0.0
    command: str = ""
    waiting_for_vision: bool = False
    phase: str = "detect"   # grasp 子阶段: detect(等首检视觉) | center(等底座反馈) | grasp(等抓取反馈) | verify(等二次视觉)


def load_yaml(path: str) -> dict[str, Any]:
    expanded = os.path.expandvars(path or "")
    if expanded and os.path.exists(expanded):
        with open(expanded, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def normalize_zone(text: str) -> str:
    value = (text or "").strip().upper()
    if ":" in value:
        value = value.split(":", 1)[0].strip()
    if value.startswith("ZONE_"):
        value = value[-1]
    return value if value in ("A", "B", "C", "D") else ""


class TaskManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("task_manager_node")
        self.declare_parameter("config_path", "")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("auto_exit_on_done", False)
        self.declare_parameter("auto_exit_delay_sec", 2.0)

        self.config_path = str(self.get_parameter("config_path").value)
        self.config = load_yaml(self.config_path)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.auto_exit_on_done = bool(self.get_parameter("auto_exit_on_done").value)
        self.auto_exit_delay_sec = float(self.get_parameter("auto_exit_delay_sec").value)

        fsm_cfg = self.config.get("fsm", {})
        arm_cfg = self.config.get("arm", {})
        inspection_cfg = self.config.get("inspection", {})

        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.start_waypoint = fsm_cfg.get("start_waypoint", "start_exit")
        self.obstacle_entry_waypoint = fsm_cfg.get("obstacle_entry_waypoint", "obstacle_entry")
        self.obstacle_exit_waypoint = fsm_cfg.get("obstacle_exit_waypoint", "obstacle_exit")
        self.inspection_waypoints = list(
            fsm_cfg.get(
                "inspection_waypoints",
                [
                    "inspection_box_1_side_1",
                    "inspection_box_1_side_2",
                    "inspection_box_2_side_1",
                    "inspection_box_2_side_2",
                ],
            )
        )
        self.pick_waypoint = fsm_cfg.get("pick_waypoint", "pick_area")
        self.place_waypoints = dict(fsm_cfg.get("place_waypoints", {}))
        self.final_waypoint = fsm_cfg.get("final_waypoint", "finish")
        self.max_abnormal_zones = int(fsm_cfg.get("max_abnormal_zones", 2))
        # 放置区字母确认：到达放置区后先识别字母，与记忆中的异常目标一致才松爪。
        # false 时退化为纯航点对齐（原行为）。
        self.place_visual_confirm = bool(fsm_cfg.get("place_visual_confirm", True))
        self.inspection_timeout_sec = float(inspection_cfg.get("wait_timeout_sec", 45.0))
        self.inspection_per_waypoint_sec = float(inspection_cfg.get("per_waypoint_wait_sec", 5.0))
        self.arm_max_retries = int(arm_cfg.get("max_retries", 3))
        self.arm_feedback_timeout_sec = float(arm_cfg.get("feedback_timeout_sec", 20.0))
        self.use_services = bool(arm_cfg.get("use_services", True))
        self.grasp_service_name = arm_cfg.get("grasp_service", "/arm/grasp_red_bar")
        self.place_service_prefix = arm_cfg.get("place_service_prefix", "/arm/place_")
        self.arm_command_topic = arm_cfg.get("arm_command_topic", "/arm/command")
        self.feedback_topic = arm_cfg.get("feedback_topic", "/arm/feedback")
        self.direct_grasp_topic = arm_cfg.get("direct_grasp_topic", "/task/direct_grasp")
        self.placement_zones = load_yaml(arm_cfg.get("grasp_config", "")).get("placement_zones", {})
        # 放置区字母识别结果 topic（必须与 vision_node 的 zone_topic 一致，默认 /placement/recognized_zone）
        self.zone_topic = str(arm_cfg.get("zone_topic", "/placement/recognized_zone"))

        self.state = State.WAIT_LOCALIZATION
        self.localization_ok = self.dry_run
        self.current_goal = ""
        self.inspection_index = 0
        self.inspection_started_at = 0.0
        self.inspection_all = ""
        self.abnormal_zones: list[str] = []
        self.target_index = 0
        self.arm_request: ArmRequest | None = None
        self.last_status = ""
        self.started = False
        self._exit_timer = None
        # 导航航点超时(秒): 某航点迟迟未到达时跳过继续, 不卡死主流程
        self.nav_timeout_sec = float(fsm_cfg.get("nav_timeout_sec", 60.0))
        self._nav_goal_started_at = 0.0

        # ── 抓取闭环策略状态 (对齐源码版) ──────
        self._grasp_retries = 0        # 当前物体空抓重试次数
        self._pre_x = 0.0              # 抓取前物体坐标 (验证用)
        self._pre_y = 0.0
        self._pre_z = 0.0
        self._pre_cx = 320
        self._verify_cx = 320          # VERIFY 时检测到的 cx, 判断旋转方向
        self._desired_base = 512       # 期望底座位置, 跨方法共享传给 direct_grasp
        self._last_seen_cx: int | None = None   # 每次视觉成功更新, 供丢失回退
        self._rollback_count = 0       # 视觉丢失回退次数
        self._grasp_pose: dict | None = None    # 最近一次有效位姿 (微调失败兜底直抓)

        # ── 放置区字母确认状态 ──────────────────────
        self._place_zone_pending = False        # 是否在等待放置区字母识别
        self._place_zone_retries = 0            # 当前放置区确认重试次数
        self._place_zone_started_at = 0.0       # 本次确认开始时间
        # 到达放置区时快照的目标字母（防止 abnormal_zones 在 pending 期间被新记忆覆盖后比对错位）
        self._place_zone_target = ""

        self.pub_goal = self.create_publisher(String, self.config.get("navigation", {}).get("goal_topic", "/waypoint/goal"), 10)
        self.pub_cone = self.create_publisher(Bool, self.config.get("cone_avoidance", {}).get("enabled_topic", "/motion/enable_cone_avoidance"), 10)
        self.pub_stop = self.create_publisher(Bool, self.config.get("motion", {}).get("stop_topic", "/motion/stop"), 10)
        self.pub_status = self.create_publisher(String, "/task/status", 10)
        self.pub_targets = self.create_publisher(String, inspection_cfg.get("target_topic", "/inspection/target_zones"), 10)
        self.pub_vision_request = self.create_publisher(String, "/vision/detect_request", 10)
        self.pub_arm_command = self.create_publisher(String, self.arm_command_topic, 10)

        self.create_subscription(Bool, "/localization/ok", self._on_localization, 10)
        self.create_subscription(String, self.config.get("navigation", {}).get("status_topic", "/waypoint/status"), self._on_waypoint_status, 10)
        self.create_subscription(String, inspection_cfg.get("result_topic", "/inspection/all"), self._on_inspection_all, 10)
        self.create_subscription(String, inspection_cfg.get("target_topic", "/inspection/target_zones"), self._on_target_zones, 10)
        self.create_subscription(String, self.zone_topic, self._on_placement_zone, 10)
        self.create_subscription(String, "/vision/grasp_pose", self._on_vision_pose, 10)
        self.create_subscription(String, self.feedback_topic, self._on_arm_feedback, 10)
        self.create_subscription(Bool, "/task/reset", self._on_reset, 10)
        self.create_subscription(String, "/task/start", self._on_start, 10)
        # 仪表盘结果记忆（语音播报时存储）→ 放置目标直接来自存储的记忆
        self.create_subscription(String, "/inspection/gauge_memory", self._on_gauge_memory, 10)

        self.trigger_type = None
        self.grasp_client = None
        self.place_clients = {}
        if self.use_services:
            try:
                from std_srvs.srv import Trigger

                self.trigger_type = Trigger
                self.grasp_client = self.create_client(Trigger, self.grasp_service_name)
                for zone in ("A", "B", "C", "D"):
                    self.place_clients[zone] = self.create_client(Trigger, f"{self.place_service_prefix}{zone}")
            except Exception as exc:
                self.get_logger().warn(f"std_srvs unavailable, using topic fallback: {exc}")
                self.use_services = False

        self.create_timer(0.2, self._tick)
        self.get_logger().info(f"guosai final FSM ready dry_run={self.dry_run} config={self.config_path}")
        self._publish_stop(True)
        self._publish_cone(False)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_localization(self, msg: Bool) -> None:
        if self.dry_run:
            self.localization_ok = True
            return
        was_ok = self.localization_ok
        self.localization_ok = bool(msg.data)
        if was_ok and not self.localization_ok and self.state not in (State.DONE, State.ERROR):
            self.get_logger().error("localization lost, stopping final FSM")
            self._fail("localization_lost")

    def _on_start(self, msg: String) -> None:
        if not self.started:
            self.get_logger().info("received /task/start")
            self.started = True

    def _on_reset(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info("reset requested")
            self._reset()

    def _on_waypoint_status(self, msg: String) -> None:
        text = msg.data.strip()
        if not text.startswith("arrived:"):
            return
        name = text.split(":", 1)[1].strip()
        if name == self.current_goal:
            self.get_logger().info(f"arrived at {name}")
            self._handle_arrival(name)

    def _on_inspection_all(self, msg: String) -> None:
        self.inspection_all = msg.data.strip()
        zones = self._parse_abnormal_zones(self.inspection_all)
        if zones is not None:
            # ★ pending 期间忽略：目标字母已在 _on_place_arrival 快照到 _place_zone_target
            if self._place_zone_pending:
                self.get_logger().info(
                    f"inspection_all 到达但正在确认放置区，忽略 (current target={self._place_zone_target})")
                return
            self.abnormal_zones = zones[: self.max_abnormal_zones]
            self.pub_targets.publish(String(data=",".join(self.abnormal_zones)))
            self.get_logger().info(f"abnormal_zones={self.abnormal_zones}")

    def _on_target_zones(self, msg: String) -> None:
        zones = [normalize_zone(item) for item in msg.data.split(",")]
        zones = [zone for zone in zones if zone]
        if zones and not self._place_zone_pending:
            self.abnormal_zones = zones[: self.max_abnormal_zones]

    def _on_gauge_memory(self, msg: String) -> None:
        """仪表盘结果记忆（语音播报时存储）→ 异常区域即放置目标。

        防御：
          - JSON 解析失败 / 顶层非 dict → 忽略
          - abnormal_zones 不是 list（兼容老版本发布字符串如 "ABCD"）→ 忽略
          - pending 期间不会改 abnormal_zones（目标字母已在 _on_place_arrival 快照到 _place_zone_target）
        """
        try:
            data = json.loads(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"gauge_memory parse failed: {exc}")
            return
        if not isinstance(data, dict):
            return
        zones_raw = data.get("abnormal_zones", [])
        if not isinstance(zones_raw, list):
            self.get_logger().warn(f"gauge_memory.abnormal_zones not a list: {type(zones_raw).__name__}")
            return
        zones = [normalize_zone(z) for z in zones_raw]
        zones = [z for z in zones if z]
        if not zones:
            return
        # ★ pending 期间禁止改 abnormal_zones，否则字母比对的目标会偏移
        if self._place_zone_pending:
            self.get_logger().info(
                f"gauge_memory 到达但正在确认放置区，忽略 (current target={self._place_zone_target})")
            return
        self.abnormal_zones = zones[: self.max_abnormal_zones]
        self.pub_targets.publish(String(data=",".join(self.abnormal_zones)))
        self.get_logger().info(f"abnormal_zones (from gauge_memory)={self.abnormal_zones}")

    def _on_placement_zone(self, msg: String) -> None:
        """放置区字母识别结果 → 与到达时刻快照的目标比对，匹配才松爪放置。

        放置流程: GO_PLACE 到达 → 快照目标到 _place_zone_target → 请求视觉识别字母 →
        本回调比对:
          - 字母 == _place_zone_target（到达时刻的目标） → 确认，执行放置
          - 字母 != 目标 / none → 重试（PLACEMENT_ZONE_RETRIES 次）后按航点兜底放置
        非确认阶段收到字母仅打日志（观察用）。

        注意：目标字母在 _on_place_arrival 时已快照，后续 abnormal_zones 变化不影响本次比对。
        """
        zone = normalize_zone(msg.data)
        if not self._place_zone_pending:
            if zone:
                self.get_logger().info(f"placement camera sees zone {zone}")
            return
        target = self._place_zone_target
        if not zone:
            self.get_logger().warn("place zone confirm: 识别到 none，重试")
            self._place_zone_retry("none")
            return
        if zone == target:
            self.get_logger().info(
                f"═══════════════════════════════════════\n"
                f"  ★ 放置区字母 {zone} == 目标 {target}（记忆异常）→ 确认放置 ★\n"
                "═══════════════════════════════════════")
            self._place_zone_pending = False
            self._start_place(target)
        else:
            self.get_logger().warn(
                f"place zone confirm: 识别字母 {zone} != 目标 {target}（异常区），重试")
            self._place_zone_retry(f"mismatch {zone}!={target}")

    def _place_zone_retry(self, reason: str) -> None:
        self._place_zone_retries += 1
        if self._place_zone_retries > PLACEMENT_ZONE_RETRIES:
            self.get_logger().error(
                f"place zone confirm 失败({reason}) {PLACEMENT_ZONE_RETRIES} 次，按航点兜底放置")
            self._place_zone_pending = False
            self._start_place(self._place_zone_target)
            return
        self.get_logger().info(f"place zone confirm retry {self._place_zone_retries}/{PLACEMENT_ZONE_RETRIES}")
        self._request_place_zone_vision()

    def _request_place_zone_vision(self) -> None:
        self._place_zone_started_at = self._now()
        self.get_logger().info(
            f"请求放置区字母识别 (attempt {self._place_zone_retries + 1})")
        if self.dry_run:
            # dry-run 无法真识别，直接按航点放置
            self._place_zone_pending = False
            self._start_place(self._place_zone_target)
        else:
            self.pub_vision_request.publish(String(data="zone"))

    def _check_place_zone_timeout(self) -> None:
        """放置区字母确认超时兜底：超过 PLACEMENT_ZONE_TIMEOUT_SEC 直接按航点放置。"""
        if self._now() - self._place_zone_started_at > PLACEMENT_ZONE_TIMEOUT_SEC:
            self.get_logger().warn(
                f"place zone confirm timeout ({PLACEMENT_ZONE_TIMEOUT_SEC}s)，按航点兜底放置")
            self._place_zone_pending = False
            self._start_place(self._place_zone_target)

    def _on_vision_pose(self, msg: String) -> None:
        if self.arm_request is None or not self.arm_request.waiting_for_vision:
            return
        data = msg.data.strip()
        if data in ("", "none", "invalid_depth", "low_conf"):
            self.get_logger().warn(f"grasp vision failed: {data}")
            self._handle_grasp_vision_fail(data)
            return
        parts = data.split("|")
        if parts[0] != "grasp" or len(parts) < 5:
            self.get_logger().warn(f"bad grasp pose: {data}")
            self._handle_grasp_vision_fail(data)
            return

        cx = int(parts[6]) if len(parts) > 6 else 320
        cy = int(parts[7]) if len(parts) > 7 else 240
        pose = {
            "x": float(parts[1]),
            "y": float(parts[2]),
            "z": float(parts[3]),
            "angle": float(parts[4]),
            "conf": float(parts[5]) if len(parts) > 5 else 0.5,
            "cx": cx,
            "cy": cy,
        }
        # ★ 每次成功检测都记录, 供视觉丢失回退用
        self._last_seen_cx = cx
        self._rollback_count = 0
        self.get_logger().info(
            f"grasp pose x={pose['x']:.3f} y={pose['y']:.3f} z={pose['z']:.3f} "
            f"angle={pose['angle']:.1f} conf={pose['conf']:.2f} px=({cx},{cy})"
        )

        # ── VERIFY: 抓取后二次视觉确认 ────────
        if self.arm_request.phase == "verify":
            self._verify_grasp(pose)
            return

        # ── DETECT: 边缘微调(仅首次) / 直抓 ────
        self._grasp_pose = pose
        if self._grasp_retries > 0:
            # 重试中跳过边缘微调, 避免与定向旋转冲突
            self.get_logger().info(f"retrying grasp, skip edge align (retries={self._grasp_retries})")
            self._do_grasp(pose)
            return

        if cx < GRASP_EDGE_MARGIN:
            adj_cx = cx - GRASP_EDGE_ADJUST
            self.get_logger().info(
                f"object too left (cx={cx}<{GRASP_EDGE_MARGIN}), rotate base →cx={adj_cx}")
            self._send_center_base(adj_cx)
            return
        if cx > 640 - GRASP_EDGE_MARGIN:
            adj_cx = cx + GRASP_EDGE_ADJUST
            self.get_logger().info(
                f"object too right (cx={cx}>{640 - GRASP_EDGE_MARGIN}), rotate base →cx={adj_cx}")
            self._send_center_base(adj_cx)
            return

        self.get_logger().info(f"object in safe zone (cx={cx},cy={cy}), direct grasp")
        self._do_grasp(pose)

    def _on_arm_feedback(self, msg: String) -> None:
        if self.arm_request is None:
            return
        parts = msg.data.split("|")
        cmd = parts[0].strip() if parts else ""
        ok = len(parts) > 1 and parts[1].strip() == "success"
        if cmd != self.arm_request.command:
            return
        self.get_logger().info(f"arm feedback {cmd} ok={ok}")

        # ── 底座微调完成 → 重新检测 ────────────
        if cmd == "center_base":
            self._on_center_base_done(ok)
            return

        # ── 直抓完成 → 二次视觉确认 ────────────
        if cmd == "direct_grasp":
            if ok:
                self.arm_request.phase = "verify"
                self.arm_request.waiting_for_vision = True
                self.arm_request.started_at = self._now()
                self.state = State.VERIFY
                self.get_logger().info("arm lifted, second vision confirm...")
                if self.dry_run:
                    self._complete_arm_request(True)
                else:
                    self.pub_vision_request.publish(String(data="red"))
            else:
                self._retry_or_fail_arm()
            return

        if ok:
            self._complete_arm_request(True)
        else:
            self._retry_or_fail_arm()

    def _tick(self) -> None:
        self._publish_status()
        if self.state == State.WAIT_LOCALIZATION:
            if self.localization_ok and (self.auto_start or self.started):
                self.started = True
                self._publish_stop(False)
                self._go(self.start_waypoint, State.GO_START)
            return

        if self.arm_request is not None:
            elapsed = self._now() - self.arm_request.started_at
            if elapsed > self.arm_feedback_timeout_sec:
                self.get_logger().warn(f"arm request timeout: {self.arm_request}")
                if self.arm_request.phase == "verify":
                    # 二次确认超时 → 物体大概率被夹爪完全遮挡 → 判定成功
                    self.get_logger().info("verify timeout, treat as success (object occluded)")
                    self._complete_arm_request(True)
                else:
                    self._retry_or_fail_arm()
            return

        # 放置区字母确认超时 → 按航点兜底放置（不卡死）
        if self._place_zone_pending:
            self._check_place_zone_timeout()
            return

        # ── 导航航点超时: 某个 GO_* 航点迟迟未到达 → 跳过继续 (不卡死主流程) ──
        if self.state in (
            State.GO_START, State.GO_OBSTACLE_ENTRY, State.OBSTACLE_ZONE,
            State.GO_INSPECTION, State.GO_PICK, State.GO_PLACE, State.GO_FINISH,
        ):
            elapsed = self._now() - self._nav_goal_started_at
            if elapsed > self.nav_timeout_sec:
                self.get_logger().warn(
                    f"nav timeout: {self.state.value} goal={self.current_goal} "
                    f"({elapsed:.0f}s > {self.nav_timeout_sec:.0f}s), skip and continue"
                )
                self._handle_arrival("__nav_timeout__")
            return

        if self.state == State.WAIT_INSPECTION:
            if self.dry_run and not self.abnormal_zones:
                self.inspection_all = "A:abnormal,B:normal,C:abnormal,D:normal"
                self.abnormal_zones = ["A", "C"][: self.max_abnormal_zones]
                self.pub_targets.publish(String(data=",".join(self.abnormal_zones)))
            if self._inspection_complete():
                self.get_logger().info(f"inspection complete: {self.inspection_all}")
                self._go(self.pick_waypoint, State.GO_PICK)
            elif (
                self.inspection_index < len(self.inspection_waypoints)
                and self._now() - self.inspection_started_at > self.inspection_per_waypoint_sec
            ):
                self.get_logger().info("inspection not complete, moving to next inspection side")
                self._go_next_inspection()
            elif self._now() - self.inspection_started_at > self.inspection_timeout_sec:
                self.get_logger().warn("inspection timeout, continuing with current abnormal_zones")
                self._go(self.pick_waypoint, State.GO_PICK)

    def _handle_arrival(self, name: str) -> None:
        if self.state == State.GO_START:
            self._go(self.obstacle_entry_waypoint, State.GO_OBSTACLE_ENTRY)
        elif self.state == State.GO_OBSTACLE_ENTRY:
            self._publish_cone(True)
            self._go(self.obstacle_exit_waypoint, State.OBSTACLE_ZONE)
        elif self.state == State.OBSTACLE_ZONE:
            self._publish_cone(False)
            self.inspection_index = 0
            self._go_next_inspection()
        elif self.state == State.GO_INSPECTION:
            self._start_inspection_wait()
        elif self.state == State.GO_PICK:
            if not self.abnormal_zones:
                self.get_logger().info("no abnormal zones, skipping pick/place")
                self._go_finish_or_done()
            else:
                self._start_grasp()
        elif self.state == State.GO_PLACE:
            self._on_place_arrival()
        elif self.state == State.GO_FINISH:
            self._done()

    def _go_next_inspection(self) -> None:
        if self.inspection_index >= len(self.inspection_waypoints):
            self._start_inspection_wait()
            return
        waypoint = self.inspection_waypoints[self.inspection_index]
        self.inspection_index += 1
        self._go(waypoint, State.GO_INSPECTION)

    def _start_inspection_wait(self) -> None:
        self.state = State.WAIT_INSPECTION
        self.inspection_started_at = self._now()
        self.get_logger().info("waiting final /inspection/all")

    def _start_grasp(self) -> None:
        self.state = State.GRASP
        # 重置本次抓取的闭环策略状态
        self._grasp_retries = 0
        self._desired_base = 512
        self._last_seen_cx = None
        self._rollback_count = 0
        self._grasp_pose = None
        self._pre_x = 0.0
        self._pre_y = 0.0
        self._pre_z = 0.0
        self._pre_cx = 320
        self.arm_request = ArmRequest(kind="grasp", attempt=1, started_at=self._now())
        if self._call_grasp_service_if_ready():
            return
        self._request_grasp_vision()

    def _start_place(self, zone: str | None = None) -> None:
        # 优先用显式传入的 zone（来自 _place_zone_target 快照），
        # 兜底才读 abnormal_zones[target_index]（向后兼容直接调用的旧路径）。
        if not zone:
            zone = self.abnormal_zones[self.target_index] if self.target_index < len(self.abnormal_zones) else ""
        if not zone:
            self.get_logger().error(f"place: no zone to place (target_index={self.target_index}, "
                                    f"abnormal_zones={self.abnormal_zones})")
            self._fail("place_no_zone")
            return
        self.state = State.PLACE
        self.arm_request = ArmRequest(kind="place", target_zone=zone, attempt=1, started_at=self._now())
        if self._call_place_service_if_ready(zone):
            return
        self._publish_place_command(zone)

    def _on_place_arrival(self) -> None:
        """到达放置区：先做相机字母确认（与记忆中的异常目标一致才放），失败兜底。

        ★ 关键：到达瞬间把当前目标字母快照到 _place_zone_target，后续视觉比对都用这个快照；
        即使 abnormal_zones 在 pending 期间被新记忆覆盖也不会影响本次确认。
        """
        self.state = State.PLACE
        if not self.abnormal_zones:
            self.get_logger().warn("place arrival but no abnormal_zones")
            self._go_finish_or_done()
            return
        target = self.abnormal_zones[self.target_index] if self.target_index < len(self.abnormal_zones) else ""
        if not target:
            self.get_logger().error(f"place arrival but target_index invalid: {self.target_index}")
            self._go_finish_or_done()
            return
        self._place_zone_target = target   # 快照目标字母
        if not self.place_visual_confirm:
            self.get_logger().info("place_visual_confirm=false，按航点直接放置")
            self._start_place(self._place_zone_target)
            return
        self._place_zone_pending = True
        self._place_zone_retries = 0
        self._place_zone_started_at = self._now()
        self.get_logger().info(
            f"到达放置区，等待字母确认 (目标={self._place_zone_target})")
        self._request_place_zone_vision()

    def _call_grasp_service_if_ready(self) -> bool:
        if not self.use_services or self.grasp_client is None or not self.grasp_client.service_is_ready():
            return False
        self.get_logger().info(f"calling {self.grasp_service_name}")
        future = self.grasp_client.call_async(self.trigger_type.Request())
        future.add_done_callback(lambda fut: self._on_service_done(fut, "grasp"))
        return True

    def _call_place_service_if_ready(self, zone: str) -> bool:
        client = self.place_clients.get(zone)
        if not self.use_services or client is None or not client.service_is_ready():
            return False
        self.get_logger().info(f"calling {self.place_service_prefix}{zone}")
        future = client.call_async(self.trigger_type.Request())
        future.add_done_callback(lambda fut: self._on_service_done(fut, "place"))
        return True

    def _on_service_done(self, future, kind: str) -> None:
        try:
            response = future.result()
            ok = bool(response.success)
            self.get_logger().info(f"{kind} service ok={ok} message={response.message}")
        except Exception as exc:
            self.get_logger().warn(f"{kind} service failed: {exc}")
            ok = False
        if ok:
            self._complete_arm_request(True)
        else:
            self._retry_or_fail_arm()

    def _request_grasp_vision(self) -> None:
        if self.arm_request is None:
            return
        self.arm_request.phase = "detect"
        self.arm_request.waiting_for_vision = True
        self.arm_request.started_at = self._now()
        self.state = State.GRASP
        self.get_logger().info(f"requesting red-bar vision attempt {self.arm_request.attempt}")
        if self.dry_run:
            self._complete_arm_request(True)
        else:
            self.pub_vision_request.publish(String(data="red"))

    def _do_grasp(self, pose: dict) -> None:
        """直抓: 使用 _desired_base (跨方法共享), 记录抓取前坐标供验证"""
        if self.arm_request is None:
            return
        self._pre_x = pose["x"]
        self._pre_y = pose["y"]
        self._pre_z = pose["z"]
        self._pre_cx = pose["cx"]
        self.get_logger().info(
            f"█████ direct grasp: pre_pos=({self._pre_x:.3f},{self._pre_y:.3f},"
            f"{self._pre_z:.3f}) base={self._desired_base} "
            f"attempt={self.arm_request.attempt} █████"
        )
        cmd = (
            f"direct_grasp|{pose['x']:.4f}|{pose['y']:.4f}|{pose['z']:.4f}|"
            f"{pose['angle']:.1f}|3.0|{pose['cx']}|{pose['cy']}|{self._desired_base}"
        )
        self.arm_request.command = "direct_grasp"
        self.arm_request.phase = "grasp"
        self.arm_request.waiting_for_vision = False
        self.arm_request.started_at = self._now()
        self.state = State.GRASP
        if self.dry_run:
            self._complete_arm_request(True)
        else:
            self.pub_arm_command.publish(String(data=cmd))

    def _send_center_base(self, cx_target: int) -> None:
        """发送底座微调命令, 同时记录期望底座位置 (跨方法共享)"""
        if self.arm_request is None:
            return
        self._desired_base = max(200, min(800, int(512 + (cx_target - 320) * 0.5)))
        cmd = f"center_base|||0|0|1.0|{cx_target}|0"
        self.arm_request.command = "center_base"
        self.arm_request.phase = "center"
        self.arm_request.waiting_for_vision = False
        self.arm_request.started_at = self._now()
        self.state = State.CENTER
        self.get_logger().info(
            f"send center_base →cx={cx_target} desired_base={self._desired_base}")
        if self.dry_run:
            self._on_center_base_done(True)
        else:
            self.pub_arm_command.publish(String(data=cmd))

    def _on_center_base_done(self, ok: bool) -> None:
        """底座微调反馈: 成功→重新检测; 失败→用旧位姿兜底直抓"""
        if self.arm_request is None:
            return
        if not ok:
            self.get_logger().error("base align failed, fallback direct grasp with last pose")
            if self._grasp_pose is not None:
                self._do_grasp(self._grasp_pose)
            else:
                self._retry_or_fail_arm()
            return
        self.get_logger().info("base aligned, re-detecting...")
        self.arm_request.phase = "detect"
        self.arm_request.waiting_for_vision = True
        self.arm_request.started_at = self._now()
        self.state = State.GRASP
        if self.dry_run:
            self._complete_arm_request(True)
        else:
            self.pub_vision_request.publish(String(data="red"))

    def _verify_grasp(self, pose: dict) -> None:
        """
        z 轴判断法: 对比抓取前后物体 z 坐标变化
        - Δz 明显 → 物体被拎起 → 成功
        - Δz 几乎不变 → 空抓 → 按 |cx-320| 动态定向旋转底座重试
        """
        dz = abs(pose["z"] - self._pre_z)
        self.get_logger().info(
            f"verify: pre_z={self._pre_z:.3f} post_z={pose['z']:.3f} "
            f"Δz={dz:.3f}m (tol={GRASP_Z_TOL:.3f}m)")

        if dz > GRASP_Z_TOL:
            self.get_logger().info(
                "═══════════════════════════════════════\n"
                "  ★★★★★ grasp verified: object lifted ★★★★★\n"
                "═══════════════════════════════════════")
            self._complete_arm_request(True)
            return

        self._grasp_retries += 1
        self.get_logger().warn(
            f"✗ empty grasp! object still on table "
            f"({self._grasp_retries}/{MAX_GRASP_RETRIES})")

        if self._grasp_retries > MAX_GRASP_RETRIES:
            self.get_logger().error("grasp retries exhausted, return base home and give up")
            self.pub_arm_command.publish(String(data="center_base|||0|0|1.0|320|0"))
            self.arm_request = None
            self._fail("grasp_empty_after_retries")
            return

        # ★ 定向旋转: 按物体离中心距离动态计算旋转量 (40~150px)
        cx_now = pose["cx"]
        self._verify_cx = cx_now
        pixel_off = abs(cx_now - 320)
        degree_offset = max(
            GRASP_RETRY_DEG_MIN,
            min(GRASP_RETRY_DEG_MAX, int(40 + pixel_off * 0.35)),
        )
        approx_deg = degree_offset * 0.1
        if cx_now > 320:
            adj_cx = 320 - degree_offset
            direction = "right"
        else:
            adj_cx = 320 + degree_offset
            direction = "left"
        self.get_logger().info(
            f"→ retry: rotate base {direction} ~{approx_deg:.1f}° "
            f"(|cx-320|={pixel_off}px, offset={degree_offset}px, adj_cx={adj_cx})")
        self._send_center_base(adj_cx)

    def _handle_grasp_vision_fail(self, reason: str) -> None:
        """视觉检测失败处理:
        - VERIFY 时失败 → 物体被夹爪遮挡/消失 → 判定成功
        - 检测时失败 → 回退到底座记忆位置 (最多 VISION_ROLLBACK_MAX 次)"""
        if self.arm_request is None:
            return
        if self.arm_request.phase == "verify":
            self.get_logger().info(
                "═══════════════════════════════════════\n"
                "  ★★★★★ grasp success (object occluded/disappeared) ★★★★★\n"
                "═══════════════════════════════════════")
            self._complete_arm_request(True)
            return

        rollback_cx = self._last_seen_cx if self._last_seen_cx is not None else self._pre_cx
        if self._rollback_count < VISION_ROLLBACK_MAX:
            self._rollback_count += 1
            self.get_logger().warn(
                f"vision lost ({reason})! rollback base →cx={rollback_cx} "
                f"(src={'memory' if self._last_seen_cx is not None else 'pre_grasp'}, "
                f"{self._rollback_count}/{VISION_ROLLBACK_MAX})")
            self._send_center_base(rollback_cx)
            return
        self.get_logger().error(f"vision rollback {VISION_ROLLBACK_MAX} times, retry vision")
        self._retry_or_fail_arm()

    def _publish_place_command(self, zone: str) -> None:
        if self.arm_request is None:
            return
        pos = self.placement_zones.get(zone, {}).get("position", [0.2, 0.0, 0.0])
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        cmd = f"place|{x:.4f}|{y:.4f}|{z:.4f}|0|3.0"
        self.arm_request.command = "place"
        self.arm_request.started_at = self._now()
        self.get_logger().info(f"fallback place {zone} attempt {self.arm_request.attempt}")
        if self.dry_run:
            self._complete_arm_request(True)
        else:
            self.pub_arm_command.publish(String(data=cmd))

    def _complete_arm_request(self, ok: bool) -> None:
        request = self.arm_request
        self.arm_request = None
        if not ok or request is None:
            self._fail("arm_action_failed")
            return
        if request.kind == "grasp":
            # 抓取成功 → 清理闭环策略状态, 进入搬运
            self._grasp_retries = 0
            self._last_seen_cx = None
            zone = self.abnormal_zones[self.target_index]
            waypoint = self.place_waypoints.get(zone)
            if not waypoint:
                self._fail(f"missing_place_waypoint_{zone}")
                return
            self._go(waypoint, State.GO_PLACE)
        elif request.kind == "place":
            self.target_index += 1
            if self.target_index >= len(self.abnormal_zones):
                self._go_finish_or_done()
            else:
                self._go(self.pick_waypoint, State.GO_PICK)

    def _retry_or_fail_arm(self) -> None:
        if self.arm_request is None:
            self._fail("arm_action_failed")
            return
        self.arm_request.attempt += 1
        if self.arm_request.attempt > self.arm_max_retries:
            self._fail(f"{self.arm_request.kind}_failed_after_retries")
            return
        self.get_logger().warn(
            f"retry {self.arm_request.kind} {self.arm_request.attempt}/{self.arm_max_retries}"
        )
        if self.arm_request.kind == "grasp":
            if self._call_grasp_service_if_ready():
                return
            self._request_grasp_vision()
        else:
            zone = self.arm_request.target_zone
            if self._call_place_service_if_ready(zone):
                return
            self._publish_place_command(zone)

    def _go(self, waypoint: str, next_state: State) -> None:
        if not waypoint:
            self._fail(f"missing_waypoint_for_{next_state.value}")
            return
        self.current_goal = waypoint
        self.state = next_state
        self._nav_goal_started_at = self._now()
        self.get_logger().info(f"go {waypoint} state={next_state.value}")
        if self.dry_run:
            self._handle_arrival(waypoint)
        else:
            self.pub_goal.publish(String(data=waypoint))

    def _go_finish_or_done(self) -> None:
        if self.final_waypoint:
            self._go(self.final_waypoint, State.GO_FINISH)
        else:
            self._done()

    def _done(self) -> None:
        self.state = State.DONE
        self._publish_cone(False)
        self._publish_stop(True)
        self.pub_goal.publish(String(data=""))
        self.get_logger().info("FINAL TASK DONE")
        if self.auto_exit_on_done and self._exit_timer is None:
            self.get_logger().info(
                f"auto exit scheduled in {self.auto_exit_delay_sec:.1f}s"
            )
            self._exit_timer = self.create_timer(
                max(0.2, self.auto_exit_delay_sec), self._auto_exit
            )

    def _fail(self, reason: str) -> None:
        # ── 容错: 非致命失败(抓取/放置/航点缺失) → 跳过当前目标继续, 不卡死 ERROR
        # 致命失败(定位丢失) → ERROR 安全停(狗不能乱走)
        skip = self._is_skippable_failure(reason)
        if skip:
            self.get_logger().warn(f"skip non-fatal failure: {reason} — 跳过当前目标继续")
            self.arm_request = None
            self._place_zone_pending = False
            self.target_index += 1
            if self.target_index >= len(self.abnormal_zones):
                self._go_finish_or_done()
            else:
                self._go(self.pick_waypoint, State.GO_PICK)
            return
        self.state = State.ERROR
        self._publish_cone(False)
        self._publish_stop(True)
        self.pub_goal.publish(String(data=""))
        self.get_logger().error(f"FINAL TASK ERROR: {reason}")

    @staticmethod
    def _is_skippable_failure(reason: str) -> bool:
        """哪些失败可以跳过继续(不影响主流程安全)。"""
        skippable = (
            "grasp_empty_after_retries",
            "grasp_failed_after_retries",
            "arm_action_failed",
            "place_failed_after_retries",
            "place_no_zone",
        )
        if reason in skippable:
            return True
        # 缺失放置航点/缺失目标航点 → 跳过该目标 (不跳过 start/obstacle/finish 关键路径)
        if reason.startswith("missing_place_waypoint_"):
            return True
        if reason.startswith("missing_waypoint_for_") and reason not in (
            "missing_waypoint_for_GO_START",
            "missing_waypoint_for_GO_FINISH",
        ):
            return True
        return False

    def _auto_exit(self) -> None:
        self._publish_cone(False)
        self._publish_stop(True)
        self.get_logger().info("dry-run complete, shutting down task_manager_node")
        if rclpy.ok():
            rclpy.shutdown()

    def _reset(self) -> None:
        self.state = State.WAIT_LOCALIZATION
        self.current_goal = ""
        self.inspection_index = 0
        self.inspection_all = ""
        self.abnormal_zones = []
        self.target_index = 0
        self.arm_request = None
        self.started = False
        # 清零抓取闭环策略状态
        self._grasp_retries = 0
        self._pre_x = 0.0
        self._pre_y = 0.0
        self._pre_z = 0.0
        self._pre_cx = 320
        self._verify_cx = 320
        self._desired_base = 512
        self._last_seen_cx = None
        self._rollback_count = 0
        self._grasp_pose = None
        # 放置区确认状态清零
        self._place_zone_pending = False
        self._place_zone_retries = 0
        self._place_zone_started_at = 0.0
        self._place_zone_target = ""
        self._publish_cone(False)
        self._publish_stop(True)
        self.pub_goal.publish(String(data=""))

    def _publish_cone(self, enabled: bool) -> None:
        self.pub_cone.publish(Bool(data=enabled))

    def _publish_stop(self, stop: bool) -> None:
        self.pub_stop.publish(Bool(data=stop))

    def _inspection_complete(self) -> bool:
        if not self.inspection_all:
            return False
        states = {}
        for part in self.inspection_all.split(","):
            if ":" not in part:
                continue
            zone, state = part.split(":", 1)
            zone = normalize_zone(zone)
            if zone:
                states[zone] = state.strip().lower()
        return all(states.get(zone) in ("normal", "abnormal") for zone in ("A", "B", "C", "D"))

    def _parse_abnormal_zones(self, text: str) -> list[str] | None:
        states = {}
        for part in text.split(","):
            if ":" not in part:
                continue
            zone, state = part.split(":", 1)
            zone = normalize_zone(zone)
            if zone:
                states[zone] = state.strip().lower()
        if not states:
            return None
        return [zone for zone in ("A", "B", "C", "D") if states.get(zone) == "abnormal"]

    def _publish_status(self) -> None:
        status = (
            f"state:{self.state.value} | goal:{self.current_goal or '-'} | "
            f"inspection:{self.inspection_all or '-'} | abnormal:{','.join(self.abnormal_zones) or '-'} | "
            f"target_index:{self.target_index}/{len(self.abnormal_zones)} | dry_run:{self.dry_run}"
        )
        if status != self.last_status:
            self.get_logger().info(status)
            self.last_status = status
        self.pub_status.publish(String(data=status))


def main() -> None:
    rclpy.init()
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node._publish_cone(False)
        node._publish_stop(True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
