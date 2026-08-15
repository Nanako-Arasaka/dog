#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GaugeMemory 仪表盘结果记忆存储单元测试。"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from nodes.gauge_memory import GaugeMemory, ZONES, normalize_status


def _tmp_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)  # 让 GaugeMemory 自己创建
    return path


def test_normalize_status():
    # 黄/红 = 异常, 绿 = 正常
    assert normalize_status("low") == "abnormal"
    assert normalize_status("high") == "abnormal"
    assert normalize_status("abnormal_low") == "abnormal"
    assert normalize_status("abnormal_high") == "abnormal"
    assert normalize_status("abnormal") == "abnormal"
    assert normalize_status("normal") == "normal"
    assert normalize_status("unknown") == "unknown"
    assert normalize_status("") == "unknown"
    assert normalize_status(None) == "unknown"
    assert normalize_status(" LOW ") == "abnormal"  # 大小写/空白容错


def test_store_zone_and_query():
    m = GaugeMemory()
    assert m.status("A") == "unknown"
    m.store_zone("A", "low")
    m.store_zone("B", "normal")
    m.store_zone("C", "high")
    assert m.status("A") == "abnormal"
    assert m.status("B") == "normal"
    assert m.status("C") == "abnormal"
    assert m.get_zone("A")["raw"] == "low"
    assert m.abnormal_zones() == ["A", "C"]
    assert m.status("D") == "unknown"


def test_store_zone_rejects_invalid():
    m = GaugeMemory()
    m.store_zone("E", "low")  # 非法区域,忽略
    m.store_zone("", "low")
    assert all(m.status(z) == "unknown" for z in ZONES)


def test_store_all_only_overwrites_present_zones():
    m = GaugeMemory()
    m.store_all({"A": "low", "B": "normal"})
    assert m.status("A") == "abnormal"
    assert m.status("B") == "normal"
    assert m.status("C") == "unknown"
    # 第二轮只播到 C: A/B 保持上一轮结果, C 更新
    m.store_all({"C": "high"})
    assert m.status("A") == "abnormal"
    assert m.status("B") == "normal"
    assert m.status("C") == "abnormal"


def test_text_formats():
    m = GaugeMemory()
    m.store_all({"A": "low", "B": "normal", "C": "high", "D": "normal"})
    assert m.summary_text() == "A:low,B:normal,C:high,D:normal"
    assert m.normalized_text() == "A:abnormal,B:normal,C:abnormal,D:normal"


def test_persistence_roundtrip():
    path = _tmp_path()
    try:
        m = GaugeMemory(path)
        m.store_all({"A": "low", "B": "normal", "C": "high"})
        # 新实例从文件恢复
        m2 = GaugeMemory(path)
        assert m2.status("A") == "abnormal"
        assert m2.status("B") == "normal"
        assert m2.status("C") == "abnormal"
        assert m2.status("D") == "unknown"
        assert m2.abnormal_zones() == ["A", "C"]
        # 文件内容结构正确
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert set(data["zones"].keys()) == set(ZONES)
        assert data["abnormal_zones"] == ["A", "C"]
        assert "announced" in data
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_missing_file_is_safe():
    m = GaugeMemory("/nonexistent/dir/never.json")
    assert all(m.status(z) == "unknown" for z in ZONES)


def test_load_corrupt_file_is_safe():
    path = _tmp_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json!!")
        m = GaugeMemory(path)
        assert all(m.status(z) == "unknown" for z in ZONES)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_reset():
    path = _tmp_path()
    try:
        m = GaugeMemory(path)
        m.store_all({"A": "low"})
        assert m.abnormal_zones() == ["A"]
        m.reset()
        assert m.abnormal_zones() == []
        assert m.status("A") == "unknown"
        # 重置后文件也清空
        m2 = GaugeMemory(path)
        assert m2.status("A") == "unknown"
    finally:
        if os.path.exists(path):
            os.unlink(path)
