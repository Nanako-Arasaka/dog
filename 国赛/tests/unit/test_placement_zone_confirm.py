#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""放置区字母确认联动逻辑单测（task_manager_node 视觉确认部分）。

依赖 mock rclpy/std_msgs（本机无 ROS2）。只测 FSM 的纯逻辑方法，
不实例化真实节点、不联网、不加载模型。
"""

import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── mock ROS 依赖 ─────────────────────────────
for _name in ["rclpy", "sensor_msgs", "cv_bridge", "std_msgs"]:
    _m = types.ModuleType(_name)
    _m.__path__ = []
    sys.modules[_name] = _m

_mock_node = types.ModuleType("rclpy.node")


class _FakeNode:
    def __init__(self):
        self.pubs = []
        self.subs = []

    def declare_parameter(self, n, d):
        return types.SimpleNamespace(value=d)

    def get_parameter(self, n):
        return types.SimpleNamespace(value="")

    def create_publisher(self, t, topic, q):
        self.pubs.append(topic)
        return types.SimpleNamespace(publish=lambda m: None)

    def create_subscription(self, t, topic, cb, q):
        self.subs.append(topic)
        return None

    def create_client(self, *a, **k):
        return None

    def create_timer(self, *a, **k):
        return None

    def get_logger(self):
        return types.SimpleNamespace(
            info=lambda *a, **k: None,
            warn=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )


_mock_node.Node = _FakeNode
sys.modules["rclpy.node"] = _mock_node

_mock_std_msg = types.ModuleType("std_msgs.msg")


class _FakeString:
    def __init__(self, data=""):
        self.data = data


class _FakeBool:
    def __init__(self, data=False):
        self.data = data


_mock_std_msg.String = _FakeString
_mock_std_msg.Bool = _FakeBool
sys.modules["std_msgs.msg"] = _mock_std_msg

from arm_grasp.arm_grasp.task_manager_node import (  # noqa: E402
    TaskManagerNode,
    PLACEMENT_ZONE_RETRIES,
    PLACEMENT_ZONE_TIMEOUT_SEC,
)


class _FakeTM(TaskManagerNode):
    """最小测试替身：只承载放置确认逻辑需要的属性/方法。"""

    def __init__(self):
        self.abnormal_zones = ["A", "C"]
        self.target_index = 0
        self.max_abnormal_zones = 2
        self.dry_run = True
        self.state = None
        self.calls = []
        self._place_zone_pending = False
        self._place_zone_retries = 0
        self._place_zone_started_at = 0.0
        self._clock = 1000.0
        self.place_visual_confirm = True
        self._logger = types.SimpleNamespace(
            info=lambda *a, **k: None,
            warn=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        self.pub_vision_request = types.SimpleNamespace(
            publish=lambda m: self.calls.append(("request_zone", m.data)))
        self.pub_targets = types.SimpleNamespace(
            publish=lambda m: self.calls.append(("targets", m.data)))

    def get_logger(self):
        return self._logger

    def _now(self):
        return self._clock

    def _start_place(self):
        self.calls.append(("place", self.abnormal_zones[self.target_index]))


def _fresh():
    return _FakeTM()


def test_gauge_memory_drives_abnormal_zones():
    tm = _fresh()
    tm._on_gauge_memory(_FakeString(json.dumps({"abnormal_zones": ["B", "D"]})))
    assert tm.abnormal_zones == ["B", "D"]
    assert ("targets", "B,D") in tm.calls


def test_gauge_memory_bad_json_ignored():
    tm = _fresh()
    tm._on_gauge_memory(_FakeString("not-json{{{"))
    assert tm.abnormal_zones == ["A", "C"]


def test_place_arrival_starts_zone_confirm():
    tm = _fresh()
    tm.dry_run = False
    tm._on_place_arrival()
    assert tm._place_zone_pending is True
    assert ("request_zone", "zone") in tm.calls


def test_zone_match_confirms_place():
    tm = _fresh()
    tm.dry_run = False
    tm._on_place_arrival()
    tm._on_placement_zone(_FakeString("A"))  # 目标 A
    assert ("place", "A") in tm.calls
    assert tm._place_zone_pending is False


def test_zone_mismatch_retries_then_fallback():
    tm = _fresh()
    tm.dry_run = False
    tm._on_place_arrival()
    tm._on_placement_zone(_FakeString("B"))  # != 目标 A
    assert tm._place_zone_retries == 1
    assert tm._place_zone_pending is True
    for _ in range(PLACEMENT_ZONE_RETRIES):
        tm._on_placement_zone(_FakeString("B"))
    # 超过重试上限 → 兜底按航点放置
    assert ("place", "A") in tm.calls
    assert tm._place_zone_pending is False


def test_zone_none_retries():
    tm = _fresh()
    tm.dry_run = False
    tm._on_place_arrival()
    tm._on_placement_zone(_FakeString("none"))
    assert tm._place_zone_retries == 1
    assert tm._place_zone_pending is True


def test_zone_confirm_timeout_fallback():
    tm = _fresh()
    tm.dry_run = False
    tm._on_place_arrival()
    tm._place_zone_started_at = 2000.0
    tm._clock = 2000.0 + PLACEMENT_ZONE_TIMEOUT_SEC + 1
    tm._check_place_zone_timeout()
    assert ("place", "A") in tm.calls
    assert tm._place_zone_pending is False


def test_zone_msg_ignored_when_not_pending():
    tm = _fresh()
    tm._on_placement_zone(_FakeString("C"))
    assert not any(c[0] == "place" for c in tm.calls)


def test_visual_confirm_disabled_places_directly():
    tm = _fresh()
    tm.place_visual_confirm = False
    tm._on_place_arrival()
    assert ("place", "A") in tm.calls
    assert tm._place_zone_pending is False


def test_dry_run_places_directly():
    tm = _fresh()
    tm.dry_run = True  # 默认
    tm._on_place_arrival()
    assert ("place", "A") in tm.calls
