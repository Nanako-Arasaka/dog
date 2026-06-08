"""机械臂硬件抽象接口。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.config import ArmConfig
from core.types import ArmPose, JointAngles


class ArmGateway(ABC):
    """机械臂硬件抽象。

    实现：
    - MockArm  (仿真)
    - 具体型号 Arm (串口/CAN/以太网协议)

    抓取动作分解（由 Mission 编排，Arm 只执行原子动作）：
      pick()  → 移动到目标上方 → 下降 → 闭合夹爪
      lift()  → 抬起到运输高度
      place() → 移动到放置点上方 → 下降 → 打开夹爪
    """

    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def move_to_pose(self, pose: ArmPose, speed: float = 0.5) -> None: ...
    @abstractmethod
    def move_joints(self, angles: JointAngles, speed: float = 0.5) -> None: ...

    @abstractmethod
    def open_gripper(self) -> None: ...
    @abstractmethod
    def close_gripper(self, force: float = 1.0) -> None: ...

    @abstractmethod
    def move_home(self) -> None:
        """回到安全收起位置。"""
        ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    # ── 复合动作（由 Mission 直接调用）──────────────────

    @abstractmethod
    def pick(self, target: ArmPose) -> bool:
        """抓取动作：接近 → 下降 → 闭合夹爪 → 确认。

        Args:
            target: 目标物体位姿。
        Returns:
            True = 抓取成功（夹爪有物体）。
        """
        ...

    @abstractmethod
    def lift(self) -> None:
        """抬起到运输高度。"""
        ...

    @abstractmethod
    def place(self, target: ArmPose) -> bool:
        """放置动作：移动到目标上方 → 下降 → 打开夹爪 → 确认。

        Args:
            target: 放置点位姿。
        Returns:
            True = 放置成功（夹爪为空）。
        """
        ...

    @abstractmethod
    def is_moving(self) -> bool: ...
    @abstractmethod
    def get_current_pose(self) -> ArmPose: ...

    @property
    @abstractmethod
    def has_object(self) -> bool:
        """夹爪是否夹持有物体（通过压力/电流判断）。
        Mission 用此检测搬运中掉落。"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


# ── Mock 实现 ────────────────────────────────────────────


class MockArm(ArmGateway):
    """仿真机械臂。"""

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        self._connected = False
        self._moving = False
        self._has_object = False
        self._current_pose = ArmPose(x=0.2, y=0.0, z=0.1)

        # 掉落模拟
        self._drop_on_next_pick: int = 0  # 计数器：>0 表示接下来N次 pick 都失败
        self._transport_ticks_until_drop: int = -1

    # ── drop simulation API (test helpers) ──

    def simulate_drop_on_next_pick(self) -> None:
        """让下一次 pick() 返回失败（可累积多次调用）。"""
        self._drop_on_next_pick += 1

    def simulate_drop_during_transport(self, after_ticks: int = 3) -> None:
        """模拟搬运中掉落：after_ticks 次 has_object 检查后自动脱手。"""
        self._transport_ticks_until_drop = after_ticks

    # ── ArmGateway impl ──

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
        self._has_object = False

    def close_gripper(self, force: float = 1.0) -> None:
        self._has_object = True

    def move_home(self) -> None:
        self._current_pose = ArmPose(x=0.0, y=0.0, z=0.2)

    def emergency_stop(self) -> None:
        pass

    def pick(self, target: ArmPose) -> bool:
        self.move_to_pose(target)
        self.close_gripper()
        if self._drop_on_next_pick > 0:
            self._drop_on_next_pick -= 1
            self._has_object = False
            return False
        return True

    def lift(self) -> None:
        self._current_pose = ArmPose(
            x=self._current_pose.x,
            y=self._current_pose.y,
            z=0.3,  # 抬高
        )

    def place(self, target: ArmPose) -> bool:
        self.move_to_pose(target)
        self.open_gripper()
        return not self._has_object

    def is_moving(self) -> bool:
        return False

    def get_current_pose(self) -> ArmPose:
        return self._current_pose

    @property
    def has_object(self) -> bool:
        if self._transport_ticks_until_drop > 0:
            self._transport_ticks_until_drop -= 1
        elif self._transport_ticks_until_drop == 0:
            self._has_object = False
            self._transport_ticks_until_drop = -1
        return self._has_object

    @property
    def is_connected(self) -> bool:
        return self._connected
