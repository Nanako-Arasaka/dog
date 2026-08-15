#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仪表盘结果记忆存储模块（纯逻辑，无 ROS 依赖，可独立单测）。

用途
----
比赛规则：机器狗巡检 ABCD 四个箱子，识别区域字母 + 仪表盘指针状态；
随后在夹取区夹取红色长条，放到放置区中「先前巡检为异常」的箱子上。

本模块在**语音播报确定**（扬声器播出 "A 区异常" 等）的同一时刻被调用，
把该区域仪表盘结果存储下来，供后续抓取/放置阶段查询：
  - 黄针（偏低 low）  → 异常 abnormal
  - 红针（偏高 high） → 异常 abnormal
  - 绿针（正常 normal）→ 正常 normal
同时保留原始三态（raw: low/normal/high），便于调试与审计。

特性
----
- JSON 持久化（原子写：tmp + rename），进程重启不丢，现场可随时查看；
- 线程安全（内部锁），可被 ROS 回调 / 播报子线程并发调用；
- 只覆盖实际播报到的区域，未播报区域保持原值（unknown 或上一轮结果）。

数据文件格式（默认 output/gauge_memory.json）：
{
  "updated_at": 1723719600.123,
  "zones": {
    "A": {"raw": "low", "status": "abnormal", "updated_at": 1723719600.100},
    "B": {"raw": "normal", "status": "normal", "updated_at": 1723719600.200}
  },
  "abnormal_zones": ["A"],
  "announced": "A:low,B:normal"
}
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Optional

ZONES = ("A", "B", "C", "D")

# 原始仪表盘状态（三态 + 退化态）→ 二态归一化：黄/红 = 异常，绿 = 正常
_RAW_TO_STATUS = {
    "low": "abnormal",          # 黄针（偏低）
    "high": "abnormal",         # 红针（偏高）
    "abnormal_low": "abnormal",  # 退化态变体
    "abnormal_high": "abnormal",
    "abnormal": "abnormal",
    "normal": "normal",
}

UNKNOWN = "unknown"


def normalize_status(raw: str) -> str:
    """三态/退化态 → abnormal | normal | unknown。"""
    s = (raw or "").strip().lower()
    return _RAW_TO_STATUS.get(s, UNKNOWN)


class GaugeMemory:
    """仪表盘结果记忆：内存态 + JSON 持久化。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.zones: dict[str, dict] = {
            z: {"raw": UNKNOWN, "status": UNKNOWN, "updated_at": 0.0} for z in ZONES
        }
        self.updated_at: float = 0.0
        if self.path:
            self.load()

    # ── 写入 ──────────────────────────────────────────────

    def store_zone(self, zone: str, raw: str, ts: Optional[float] = None) -> None:
        """存储单区域状态（播报到哪区存哪区）。"""
        zone = (zone or "").strip().upper()
        if zone not in ZONES:
            return
        ts = ts if ts is not None else time.time()
        with self._lock:
            self.zones[zone] = {
                "raw": (raw or "").strip().lower(),
                "status": normalize_status(raw),
                "updated_at": ts,
            }
            self.updated_at = ts
            self._save_locked()

    def store_all(self, states: dict[str, str], ts: Optional[float] = None) -> None:
        """批量存储一轮播报确定的状态（只覆盖出现的区域，其余保持原值）。"""
        ts = ts if ts is not None else time.time()
        changed = False
        with self._lock:
            for zone, raw in (states or {}).items():
                zone = (zone or "").strip().upper()
                if zone not in ZONES:
                    continue
                self.zones[zone] = {
                    "raw": (raw or "").strip().lower(),
                    "status": normalize_status(raw),
                    "updated_at": ts,
                }
                changed = True
            if changed:
                self.updated_at = ts
                self._save_locked()

    # ── 查询 ──────────────────────────────────────────────

    def get_zone(self, zone: str) -> dict:
        zone = (zone or "").strip().upper()
        with self._lock:
            return dict(self.zones.get(zone, {"raw": UNKNOWN, "status": UNKNOWN, "updated_at": 0.0}))

    def status(self, zone: str) -> str:
        return self.get_zone(zone)["status"]

    def abnormal_zones(self) -> list[str]:
        """按 A→D 顺序返回异常区域列表（放置阶段消费）。"""
        with self._lock:
            return [z for z in ZONES if self.zones[z]["status"] == "abnormal"]

    def summary_text(self) -> str:
        """原始三态文本，与 /inspection/all_detailed 同格式：A:low,B:normal。"""
        with self._lock:
            return self._announced_unlocked()

    def normalized_text(self) -> str:
        """归一化二态文本，与 /inspection/all 同格式：A:abnormal,B:normal。"""
        with self._lock:
            return ",".join(f"{z}:{self.zones[z]['status']}" for z in ZONES)

    def _announced_unlocked(self) -> str:
        """无锁内部实现：调用者必须已持有 self._lock（避免锁内重入死锁）。"""
        return ",".join(f"{z}:{self.zones[z]['raw']}" for z in ZONES)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "updated_at": self.updated_at,
                "zones": {z: dict(v) for z, v in self.zones.items()},
                "abnormal_zones": [z for z in ZONES if self.zones[z]["status"] == "abnormal"],
                "announced": self._announced_unlocked(),
            }

    def reset(self) -> None:
        """清空记忆（新一轮比赛前调用 /inspection/gauge_memory_reset 触发）。"""
        with self._lock:
            for z in ZONES:
                self.zones[z] = {"raw": UNKNOWN, "status": UNKNOWN, "updated_at": 0.0}
            self.updated_at = 0.0
            self._save_locked()

    # ── 持久化 ────────────────────────────────────────────

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                for z in ZONES:
                    entry = data.get("zones", {}).get(z)
                    if isinstance(entry, dict) and entry.get("raw"):
                        self.zones[z] = {
                            "raw": str(entry["raw"]).strip().lower(),
                            "status": normalize_status(entry["raw"]),
                            "updated_at": float(entry.get("updated_at", 0.0)),
                        }
                self.updated_at = float(data.get("updated_at", 0.0))
        except (OSError, ValueError) as exc:
            print(f"[gauge_memory] load failed ({self.path}): {exc}")

    def _save_locked(self) -> None:
        """落盘（调用者必须已持有 self._lock，内部不得再获取锁）。"""
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            payload = {
                "updated_at": self.updated_at,
                "zones": self.zones,
                "abnormal_zones": [z for z in ZONES if self.zones[z]["status"] == "abnormal"],
                "announced": self._announced_unlocked(),
            }
            fd, tmp = tempfile.mkstemp(
                prefix=".gauge_memory_", dir=os.path.dirname(os.path.abspath(self.path))
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
        except OSError as exc:
            print(f"[gauge_memory] save failed ({self.path}): {exc}")


if __name__ == "__main__":
    # 简单自检
    m = GaugeMemory()
    m.store_all({"A": "low", "B": "normal", "C": "high"})
    print("zones:", m.to_dict()["zones"])
    print("abnormal:", m.abnormal_zones())
    print("normalized:", m.normalized_text())
