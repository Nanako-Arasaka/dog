"""导航层抽象接口。

提供定位、路径规划、避障策略。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.types import ConeDetection, NavigationStatus, RobotPose
from hardware.camera.interface import CameraGateway


class NavigationGateway(ABC):
    """导航抽象。

    实现：
    - VisionNavigator  (基于 AprilTag + 视觉里程计的定位 + A* 路径规划)
    - MockNavigator    (仿真，始终返回"已到达")
    """

    @abstractmethod
    def localize(self) -> RobotPose:
        """基于视觉的自我定位。

        Returns:
            当前机器人位姿（世界坐标系）。
        """
        ...

    @abstractmethod
    def plan_path(
        self,
        target: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """规划从当前位置到目标的路径。

        Args:
            target: 目标点 (x, y) 世界坐标。

        Returns:
            路径点序列（含起点和终点）。
        """
        ...

    @abstractmethod
    def navigate_to(self, target: tuple[float, float]) -> NavigationStatus:
        """导航到目标点（逐步执行，单 tick 调用一次）。

        每个 main_loop tick 调用，驱动机器人沿规划路径移动。

        Returns:
            MOVING:  仍在移动中。
            ARRIVED: 已到达目标（在容差范围内）。
            BLOCKED: 路径被阻挡。
            LOST:    定位丢失。
        """
        ...

    @abstractmethod
    def compute_avoidance(
        self,
        cones: list[ConeDetection],
    ) -> tuple[int, int]:
        """根据锥桶位置计算避障运动指令。

        Args:
            cones: 检测到的锥桶列表。

        Returns:
            (forward_value, turn_value): 摇杆控制值。
        """
        ...

    @abstractmethod
    def is_at_target(
        self,
        target: tuple[float, float],
        tolerance: float = 0.15,
    ) -> bool:
        """检查是否已到达目标点。"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置导航状态（切换阶段时调用）。"""
        ...


# ── Mock 实现 ────────────────────────────────────────────


class MockNavigator(NavigationGateway):
    """仿真导航器 —— 始终返回"已到达"。"""

    def __init__(self, camera: CameraGateway, dog: object) -> None:
        self._camera = camera
        self._dog = dog
        self._pose = RobotPose()

    def localize(self) -> RobotPose:
        return self._pose

    def plan_path(self, target: tuple[float, float]) -> list[tuple[float, float]]:
        return [(self._pose.x, self._pose.y), target]

    def navigate_to(self, target: tuple[float, float]) -> NavigationStatus:
        self._pose.x, self._pose.y = target
        return NavigationStatus.ARRIVED

    def compute_avoidance(self, cones: list[ConeDetection]) -> tuple[int, int]:
        return (8000, 0)

    def is_at_target(self, target: tuple[float, float], tolerance: float = 0.15) -> bool:
        dx = self._pose.x - target[0]
        dy = self._pose.y - target[1]
        return (dx * dx + dy * dy) < tolerance * tolerance

    def reset(self) -> None:
        pass
