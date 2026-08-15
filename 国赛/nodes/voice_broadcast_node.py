#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语音播报节点 —— 已接入 detailed topic 与真实播放。

职责
----
订阅巡检冻结结果，按 A→B→C→D 的固定顺序，依次播放预录音频文件 {zone}_{state}.wav。

比赛规则（40 分 / 4 次播报，每次 10 分：字母 5 + 状态 5）：
  - 黄针 = 偏低(low)      → 异常
  - 绿针 = 正常(normal)
  - 红针 = 偏高(high)     → 异常
无声播报但终端有输出，只给 2.5 分/次，因此必须真正出声（engine 改 aplay/ffplay）。

触发与数据来源
--------------
主用 /inspection/all_detailed（格式 "A:low,B:normal,C:high,D:normal"，保留偏低/偏高）。
  - integration_bridge 现在会同时发布 /inspection/all（abnormal/normal，FSM 用）
    和 /inspection/all_detailed（low/normal/high，本节点用）。
  - 若 detailed topic 一直没数据（旧链路或退化），自动降级用 /inspection/all，
    但此时 abnormal 无法区分偏低/偏高，该区只能跳过（会丢状态分，仅作兜底）。
触发时机：收到冻结结果即播报一次；同一文本重复到达不重播；
  收到 /competition/state 回到 WAITING_INSPECTION 时重新武装。

播放器
------
内联 MiniSpeaker（mock / aplay / ffplay），与 src/hardware/speaker/interface.py
的 AudioFileSpeaker 解耦（后者依赖 app.config 注入架构）。Jetson 现场把 engine 改 aplay。
12 个预录 wav：A/B/C/D × low/normal/high，放 output/audio/，命名 A_low.wav 等。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

# 仪表盘结果记忆存储（兼容 python3 nodes/voice_broadcast_node.py 与 -m 两种运行方式）
try:
    from nodes.gauge_memory import GaugeMemory
except ImportError:  # pragma: no cover
    from gauge_memory import GaugeMemory

ZONES = ["A", "B", "C", "D"]

# zone 状态词 → 音频 key 后缀。
# low/normal/high 三态直接对应预录 wav；abnormal 是退化态（detailed 不可用时），
# 无法区分偏低/偏高，按规则会丢状态分，仅作兜底跳过。
STATE_TO_KEY = {
    "low": "low",
    "abnormal_low": "low",
    "high": "high",
    "abnormal_high": "high",
    "normal": "normal",
    "abnormal": "abnormal",  # 退化态：无 low/high 区分，无对应 wav，播放时跳过
    "unknown": "unknown",
}


def normalize_zone(text: str) -> str:
    value = (text or "").strip().upper()
    if value.startswith("ZONE_"):
        value = value[-1]
    return value if value in ZONES else ""


# ── 最小播放器（与 src/hardware/speaker 解耦，赛前可换成 AudioFileSpeaker）────


class MiniSpeaker:
    """仅支持本地 .wav 播放的最小实现，mock/aplay/ffplay 三选一。"""

    def __init__(self, audio_dir: str, engine: str, device: str = "", log_path: Optional[str] = None) -> None:
        self.audio_dir = Path(audio_dir)
        self.engine = engine
        # aplay -D 指定声卡（指向机器狗自带扬声器或外置 USB 卡）；空=系统默认卡
        # 仅允许安全字符，避免命令注入
        if device and all(c.isalnum() or c in ":.,-_" for c in device):
            self.device = device
        else:
            self.device = ""
        self.log_path = Path(log_path) if log_path else None
        self._lock = threading.Lock()

    def play(self, key: str) -> threading.Thread:
        t = threading.Thread(target=self._play, args=(key,), daemon=True)
        t.start()
        return t

    def play_blocking(self, key: str) -> None:
        """同步播放（当前线程等到播放结束），用于逐条播报时避免打断/抢声卡。"""
        self._play(key)

    def _play(self, key: str) -> None:
        fpath = self.audio_dir / f"{key}.wav"
        if not fpath.exists():
            # TODO: 退化处理——可改为播放通用 "abnormal" 或语音合成兜底
            print(f"[voice] audio missing: {fpath}")
            self._log(key, "missing")
            return
        try:
            if self.engine == "mock":
                print(f"[voice][mock] play {key}.wav")
            elif self.engine == "aplay":
                # 用 paplay 走 PulseAudio（Jetson 上 PA 占着 /dev/snd/controlC2，
                # 直 aplay -D plughw:2,0 拿不到硬件，PCM 状态始终 closed 无声）
                os.system(f'paplay --volume=65536 "{fpath}"')
            elif self.engine == "ffplay":
                os.system(f"ffplay -nodisp -autoexit -loglevel quiet {fpath}")
            self._log(key, "played")
        except Exception as exc:  # noqa: BLE001
            print(f"[voice] play error {key}: {exc}")
            self._log(key, "error")

    def _log(self, key: str, status: str) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f}\t{key}\t{status}\n")


# ── ROS2 节点 ────────────────────────────────────────────────


class VoiceBroadcastNode(Node):
    def __init__(self) -> None:
        super().__init__("voice_broadcast_node")

        self.declare_parameter("enabled", True)
        self.declare_parameter("audio_dir", "output/audio")
        self.declare_parameter("engine", "mock")  # mock | aplay | ffplay
        self.declare_parameter("device", "")  # aplay 声卡, 如 plughw:1,0; 空=默认卡(优先用狗自带扬声器)
        self.declare_parameter("gap_sec", 0.4)
        self.declare_parameter("result_topic", "/inspection/all")
        self.declare_parameter("detailed_topic", "/inspection/all_detailed")
        self.declare_parameter("state_topic", "/competition/state")
        self.declare_parameter("playback_log_path", "output/voice_broadcast/playback.tsv")
        self.declare_parameter("memory_path", "output/gauge_memory.json")

        self.enabled = bool(self.get_parameter("enabled").value)
        audio_dir = str(self.get_parameter("audio_dir").value)
        audio_dir = os.path.expandvars(audio_dir)
        self.engine = str(self.get_parameter("engine").value)
        self.device = str(self.get_parameter("device").value)
        self.gap_sec = float(self.get_parameter("gap_sec").value)
        result_topic = str(self.get_parameter("result_topic").value)
        detailed_topic = str(self.get_parameter("detailed_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)
        log_path = os.path.expandvars(str(self.get_parameter("playback_log_path").value))
        memory_path = os.path.expandvars(str(self.get_parameter("memory_path").value))

        self.speaker = MiniSpeaker(audio_dir, self.engine, self.device, log_path)
        self._last_text: Optional[str] = None
        self._armed = True
        # 本轮是否已用过 detailed；用于 /inspection/all 兜底去重
        self._got_detailed_this_round = False

        # ── 仪表盘结果记忆存储（播报确定时写入，供抓取/放置阶段查询）──
        self.memory = GaugeMemory(memory_path)
        # 记忆全量（JSON），供现场 echo / 下游消费
        self.pub_gauge_memory = self.create_publisher(String, "/inspection/gauge_memory", 10)
        # 新一轮比赛前重置记忆
        self.create_subscription(Bool, "/inspection/gauge_memory_reset", self._on_memory_reset, 10)

        # detailed 为主触发源；result_topic(/inspection/all) 仅在 detailed 缺失时兜底
        if detailed_topic:
            self.create_subscription(String, detailed_topic, self._on_inspection_detailed, 10)
        if result_topic and result_topic != detailed_topic:
            self.create_subscription(String, result_topic, self._on_inspection_fallback, 10)
        self.create_subscription(String, state_topic, self._on_state, 10)

        self.get_logger().info(
            f"voice_broadcast_node ready enabled={self.enabled} engine={self.engine} "
            f"detailed={detailed_topic or '-'} fallback={result_topic or '-'}"
        )

    # ---- 订阅回调 ----

    def _on_inspection_detailed(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text or not self._armed:
            return
        self._got_detailed_this_round = True
        self._trigger(text, source="detailed")

    def _on_inspection_fallback(self, msg: String) -> None:
        # 仅当 detailed 本轮没来过时，才用 /inspection/all 兜底（abnormal 无法区分偏低/偏高）
        if self._got_detailed_this_round:
            return
        text = (msg.data or "").strip()
        if not text or not self._armed:
            return
        self._trigger(text, source="fallback")

    def _trigger(self, text: str, source: str) -> None:
        if text == self._last_text:
            return  # 同一冻结结果，避免定时器重复触发导致的多播
        self._last_text = text
        self._armed = False
        self._schedule(text, source)

    def _on_state(self, msg: String) -> None:
        # 复位：巡检重新开始时重新武装，允许下一轮播报
        if "WAITING_INSPECTION" in (msg.data or ""):
            self._armed = True
            self._last_text = None
            self._got_detailed_this_round = False

    # ---- 播报编排 ----

    def _schedule(self, text: str, source: str) -> None:
        states = self._parse(text)
        # ★ 存储：扬声器播出 A/B/C/D 区域正常/异常的同一时刻，写入仪表盘结果记忆
        #   （黄/红=异常，绿=正常；只覆盖本轮到播报的区域，其余保持原值）
        if states:
            self.memory.store_all(states)
            self.get_logger().info(
                f"gauge memory stored: {self.memory.normalized_text()} "
                f"abnormal={self.memory.abnormal_zones()}"
            )
            self._publish_gauge_memory()
        # 固定顺序 A→B→C→D，12 选 4
        plan = []
        for z in ZONES:
            st = states.get(z)
            if st is None:
                continue
            key = f"{z}_{STATE_TO_KEY.get(st, st)}"
            plan.append(key)
        self.get_logger().info(f"broadcast plan (source={source}): {plan}")
        # enabled=False 时只写记忆不播声音（静默模式：debug/CI 用）
        if not self.enabled:
            self.get_logger().info(
                f"[silent mode] gauge memory stored but playback skipped (plan={plan})"
            )
            return
        threading.Thread(target=self._play_plan, args=(plan,), daemon=True).start()

    def _on_memory_reset(self, msg: Bool) -> None:
        # 新一轮比赛开始前清空记忆（发 /inspection/gauge_memory_reset Bool=true）
        if not msg.data:
            return
        self.memory.reset()
        self.get_logger().info("gauge memory reset")
        self._publish_gauge_memory()

    def _publish_gauge_memory(self) -> None:
        try:
            payload = json.dumps(self.memory.to_dict(), ensure_ascii=False)
        except Exception:  # noqa: BLE001
            payload = self.memory.normalized_text()
        self.pub_gauge_memory.publish(String(data=payload))

    def _play_plan(self, plan: list[str]) -> None:
        for key in plan:
            # 同步播放：等上一条播完再播下一条，避免 aplay 抢声卡/打断
            self.speaker.play_blocking(key)
            time.sleep(self.gap_sec)

    # ---- 解析 ----

    def _parse(self, text: str) -> dict[str, str]:
        states: dict[str, str] = {}
        for part in text.split(","):
            if ":" not in part:
                continue
            zone, state = part.split(":", 1)
            zone = normalize_zone(zone)
            if zone:
                states[zone] = state.strip().lower()
        return states


def main() -> None:
    rclpy.init()
    node = VoiceBroadcastNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
