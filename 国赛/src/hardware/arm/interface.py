"""机械臂硬件抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

from app.config import ArmConfig
from core.types import ArmPose, JointAngles


class ArmGateway(ABC):
    """机械臂硬件抽象。

    实现：
    - <具体型号>Arm (通过串口/CAN/以太网协议控制)
    - MockArm         (仿真，返回固定状态，用于 CI/开发)
    """

    @abstractmethod
    def connect(self) -> None:
        """建立与机械臂控制器的连接。"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接，执行安全停止。"""
        ...

    @abstractmethod
    def move_to_pose(self, pose: ArmPose, speed: float = 0.5) -> None:
        """末端执行器移动到目标位姿（笛卡尔空间）。

        Args:
            pose: 目标位姿（基座坐标系）。
            speed: 移动速度比例 0.0~1.0。
        """
        ...

    @abstractmethod
    def move_joints(self, angles: JointAngles, speed: float = 0.5) -> None:
        """直接控制各关节转动到目标角度。

        Args:
            angles: 各关节目标角度（弧度）。
            speed: 移动速度比例。
        """
        ...

    @abstractmethod
    def open_gripper(self) -> None:
        """打开夹爪。"""
        ...

    @abstractmethod
    def close_gripper(self, force: float = 1.0) -> None:
        """闭合夹爪。

        Args:
            force: 夹持力 0.0~1.0（归一化到最大力）。
        """
        ...

    @abstractmethod
    def move_home(self) -> None:
        """回到预定义的"家"位置（安全收起）。"""
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """紧急停止，立即释放力矩（或保持当前位置）。"""
        ...

    @abstractmethod
    def is_moving(self) -> bool:
        """机械臂是否正在运动中。"""
        ...

    @abstractmethod
    def get_current_pose(self) -> ArmPose:
        """获取当前末端位姿估计。"""
        ...

    @property
    @abstractmethod
    def has_object(self) -> bool:
        """夹爪中是否夹持有物体（通过压力/电流判断）。"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否与机械臂保持连接。"""
        ...


# ── Mock 实现 ────────────────────────────────────────────


class MockArm(ArmGateway):
    """仿真机械臂 —— 始终返回成功，用于 CI/开发。"""

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        self._connected = False
        self._moving = False
        self._gripper_open = True
        self._current_pose = ArmPose(x=0.2, y=0.0, z=0.1)

    def connect(self) -> None:
        self._connected = True
        logging.info("MockArm: connected")

    def disconnect(self) -> None:
        self._connected = False
        logging.info("MockArm: disconnected")

    def move_to_pose(self, pose: ArmPose, speed: float = 0.5) -> None:
        self._current_pose = pose

    def move_joints(self, angles: JointAngles, speed: float = 0.5) -> None:
        pass

    def open_gripper(self) -> None:
        self._gripper_open = True

    def close_gripper(self, force: float = 1.0) -> None:
        self._gripper_open = False

    def move_home(self) -> None:
        self._current_pose = ArmPose(x=0.0, y=0.0, z=0.2)

    def emergency_stop(self) -> None:
        pass

    def is_moving(self) -> bool:
        return False

    def get_current_pose(self) -> ArmPose:
        return self._current_pose

    @property
    def has_object(self) -> bool:
        return not self._gripper_open

    @property
    def is_connected(self) -> bool:
        return self._connected
