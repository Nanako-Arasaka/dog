"""巡检识别闭环异常层次结构。"""

from __future__ import annotations


# ── 巡检 ────────────────────────────────────────────────


class InspectionError(Exception):
    """巡检识别异常基类"""
    pass


class InspectionTimeoutError(InspectionError):
    """巡检阶段超时"""
    pass


class InspectionReadError(InspectionError):
    """仪表读数失败（置信度过低/多次重试失败）"""
    pass


class ZoneOCRError(InspectionError):
    """区域字母识别失败"""
    pass


# ── 感知 ────────────────────────────────────────────────


class PerceptionError(Exception):
    """感知层异常基类"""
    pass


class DetectionFailedError(PerceptionError):
    """目标检测失败（无结果/置信度过低）"""
    pass


# ── 硬件 ────────────────────────────────────────────────


class HardwareError(Exception):
    """硬件层异常基类"""
    pass


class CameraError(HardwareError):
    """相机异常"""
    pass


class SpeakerError(HardwareError):
    """语音播报异常"""
    pass
