"""机械臂运动学骨架。

提供正运动学 (FK) 和逆运动学 (IK) 的接口与工具方法。
具体求解器由各机械臂实现提供。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.types import ArmPose, JointAngles


class KinematicSolver(ABC):
    """正/逆运动学求解器抽象。"""

    @abstractmethod
    def forward_kinematics(self, angles: JointAngles) -> ArmPose:
        """正运动学：关节角度 → 末端位姿。

        Args:
            angles: 各关节角度（弧度）。

        Returns:
            末端执行器在基座坐标系下的位姿。
        """
        ...

    @abstractmethod
    def inverse_kinematics(
        self,
        target: ArmPose,
        seed: JointAngles | None = None,
    ) -> JointAngles | None:
        """逆运动学：末端位姿 → 关节角度。

        Args:
            target: 目标末端位姿。
            seed:   IK 初始猜测（用于数值求解器），可选。

        Returns:
            一组可行关节角度；无解时返回 None。
        """
        ...


def rotation_matrix_x(angle: float) -> np.ndarray:
    """绕 X 轴旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """绕 Y 轴旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """绕 Z 轴旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """单步 DH 变换矩阵（4x4）。

    Args:
        a:      连杆长度 (mm 或 m，取决于约定)。
        alpha:  连杆扭转角 (rad)。
        d:      关节偏距。
        theta:  关节角度 (rad)。
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,      ca,      d],
        [0,         0,       0,      1],
    ])
