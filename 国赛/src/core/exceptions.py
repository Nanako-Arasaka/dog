"""领域异常层次结构。

所有比赛相关异常从此处抛出，便于上层统一捕获处理。
"""

from __future__ import annotations


class MissionError(Exception):
    """任务层异常基类"""
    pass


# ── 避障 ────────────────────────────────────────────────


class ObstacleTimeoutError(MissionError):
    """避障阶段超时"""
    pass


class ObstacleBlockedError(MissionError):
    """避障路径被完全阻断"""
    pass


# ── 巡检 ────────────────────────────────────────────────


class InspectionTimeoutError(MissionError):
    """巡检阶段超时"""
    pass


class InspectionReadError(MissionError):
    """仪表读数失败（置信度过低/多次重试失败）"""
    pass


class ZoneOCRError(MissionError):
    """区域字母识别失败"""
    pass


# ── 抓取 ────────────────────────────────────────────────


class PickupFailedError(MissionError):
    """抓取任务整体失败"""
    pass


class DropLimitExceededError(MissionError):
    """掉落次数达到上限"""
    pass


class ArmError(MissionError):
    """机械臂异常"""
    pass


# ── 导航 ────────────────────────────────────────────────


class NavigationLostError(MissionError):
    """导航迷路，重定位失败"""
    pass


class NavigationBlockedError(MissionError):
    """导航路径被阻挡"""
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
