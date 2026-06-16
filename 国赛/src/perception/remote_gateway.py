"""RemotePerceptionGateway —— 通过 TCP 连接外接 NVIDIA 算力板。

通信协议（JSON over TCP，每行一条消息）：

  机器狗 → 算力板 (请求):
    {"req": "detect_zone_letters"}
    {"req": "detect_gauges"}
    {"req": "poll_inspection"}

  算力板 → 机器狗 (响应):
    {"type": "zone_letters", "detections": [...], "timestamp": 1.23}
    {"type": "gauges",       "detections": [...], "timestamp": 1.23}
    {"type": "inspection_results", "results": [...], "timestamp": 1.23}
    {"type": "error", "message": "..."}

算力板的职责：
  - 相机取流
  - A/B/C/D 字母识别
  - 仪表盘读数
  - 巡检结果融合

机器狗本地的职责：
  - 发送请求
  - 接收 JSON 结构化结果
  - 缓存最新结果
  - 超时处理
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from core.types import (
    BBox,
    GaugeReading,
    InspectionReading,
    MeterStatus,
    Zone,
    ZoneLetterResult,
)
from perception.gateway import PerceptionGateway


@dataclass
class RemotePerceptionConfig:
    """远程算力板连接配置"""

    host: str = "192.168.1.200"
    port: int = 9800
    timeout_sec: float = 2.0
    reconnect_interval: float = 3.0


class RemotePerceptionGateway(PerceptionGateway):
    """通过 TCP 连接外接 NVIDIA 算力板的感知网关。

    算力板不可直接控制机器狗运动和机械臂。
    机器狗本地只接收结构化 JSON 结果。
    """

    def __init__(self, cfg: RemotePerceptionConfig) -> None:
        self._cfg = cfg
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._connected = False

        # 结果缓存
        self._cache_zone_letters: list[ZoneLetterResult] = []
        self._cache_gauges: list[GaugeReading] = []

        # 巡检轮询
        self._inspection_queue: list[InspectionReading] = []
        self._inspection_cursor: int = 0
        self._inspection_loaded: bool = False

    # ── 连接管理 ──

    def _connect(self) -> bool:
        if self._connected:
            return True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self._cfg.timeout_sec)
            self._sock.connect((self._cfg.host, self._cfg.port))
            self._connected = True
            logging.info("RemotePerception: connected to %s:%d", self._cfg.host, self._cfg.port)
            return True
        except (OSError, socket.timeout) as e:
            logging.warning("RemotePerception: connect failed (%s), will retry", e)
            self._connected = False
            return False

    def _disconnect(self) -> None:
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _request(self, req: dict) -> dict | None:
        """发送请求并读取一行 JSON 响应。"""
        with self._lock:
            if not self._connect():
                return None
            try:
                payload = json.dumps(req, ensure_ascii=False) + "\n"
                self._sock.sendall(payload.encode("utf-8"))  # type: ignore[union-attr]
                raw = self._sock.recv(65536)  # type: ignore[union-attr]
                if not raw:
                    self._disconnect()
                    return None
                return json.loads(raw.decode("utf-8"))
            except (OSError, socket.timeout, json.JSONDecodeError) as e:
                logging.warning("RemotePerception: request failed (%s)", e)
                self._disconnect()
                return None

    # ── 请求 → 结构化结果 映射 ──

    def _parse_zone_letters(self, resp: dict) -> list[ZoneLetterResult]:
        dets = resp.get("detections", [])
        results = []
        for d in dets:
            zone_str = str(d.get("zone", "")).upper()
            if zone_str in ("A", "B", "C", "D"):
                b = d.get("bbox", {})
                results.append(ZoneLetterResult(
                    zone=Zone(zone_str),
                    confidence=float(d.get("confidence", 0)),
                    bbox=BBox(b.get("x1", 0), b.get("y1", 0), b.get("x2", 0), b.get("y2", 0)) if b else None,
                    timestamp=float(resp.get("timestamp", time.time())),
                ))
        return results

    def _parse_gauges(self, resp: dict) -> list[GaugeReading]:
        dets = resp.get("detections", [])
        results = []
        for d in dets:
            zone_str = str(d.get("zone", "")).upper()
            if zone_str not in ("A", "B", "C", "D"):
                continue
            status_str = str(d.get("status", "normal")).lower()
            try:
                status = MeterStatus(status_str)
            except ValueError:
                status = MeterStatus.NORMAL
            results.append(GaugeReading(
                zone=Zone(zone_str),
                status=status,
                confidence=float(d.get("confidence", 0)),
                raw_value=d.get("raw_value"),
                timestamp=float(resp.get("timestamp", time.time())),
            ))
        return results

    def _parse_inspection_results(self, resp: dict) -> list[InspectionReading]:
        dets = resp.get("results", resp.get("detections", []))
        results: list[InspectionReading] = []
        for d in dets:
            zone_str = str(d.get("zone", "")).upper()
            if zone_str not in ("A", "B", "C", "D"):
                continue
            status_str = str(d.get("gauge_status", d.get("status", "normal"))).lower()
            try:
                status = MeterStatus(status_str)
            except ValueError:
                status = MeterStatus.NORMAL
            results.append(InspectionReading(
                zone=Zone(zone_str),
                meter_status=status,
                confidence=float(d.get("confidence", 0)),
                timestamp=float(d.get("timestamp", resp.get("timestamp", time.time()))),
            ))
        return results

    # ── PerceptionGateway 实现 ──

    def detect_zone_letters(self, rgb: np.ndarray | None = None) -> list[ZoneLetterResult]:
        resp = self._request({"req": "detect_zone_letters"})
        if resp is None or resp.get("type") == "error":
            return self._cache_zone_letters
        self._cache_zone_letters = self._parse_zone_letters(resp)
        return self._cache_zone_letters

    def detect_gauges(self, rgb: np.ndarray | None = None) -> list[GaugeReading]:
        resp = self._request({"req": "detect_gauges"})
        if resp is None or resp.get("type") == "error":
            return self._cache_gauges
        self._cache_gauges = self._parse_gauges(resp)
        # 构建巡检播报队列
        self._inspection_queue = []
        for g in self._cache_gauges:
            self._inspection_queue.append(InspectionReading(
                zone=g.zone,
                meter_status=g.status,
                confidence=g.confidence,
                meter_raw_value=g.raw_value,
                timestamp=g.timestamp,
            ))
        self._inspection_cursor = 0
        self._inspection_loaded = True
        return self._cache_gauges

    def poll_inspection(self) -> list[InspectionReading]:
        if not self._inspection_loaded:
            resp = self._request({"req": "poll_inspection"})
            if resp is not None and resp.get("type") != "error":
                self._inspection_queue = self._parse_inspection_results(resp)
                self._inspection_cursor = 0
                self._inspection_loaded = True
        if self._inspection_cursor >= len(self._inspection_queue):
            return []
        item = self._inspection_queue[self._inspection_cursor]
        self._inspection_cursor += 1
        return [item]

    def is_ready(self) -> bool:
        return self._connected or self._connect()
