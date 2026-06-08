"""阶段处理器抽象。

状态机中的每个阶段对应一个 PhaseHandler。
这种设计使每个阶段可独立测试、可复用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.types import MissionPhase


@dataclass
class PhaseContext:
    """阶段间共享的上下文数据。

    所有阶段处理器通过此对象交换数据。
    """

    # 巡检结果
    inspection_by_zone: dict[str, Any] = field(default_factory=dict)

    # 抓取队列（异常区域字母列表）
    delivery_queue: list[str] = field(default_factory=list)

    # 掉落计数
    drop_count: int = 0

    # 当前阶段的已重试次数
    retry_count: int = 0

    # 阶段进入时间戳
    phase_enter_ts: float = 0.0

    # 自由扩展字段
    extra: dict[str, Any] = field(default_factory=dict)


class PhaseHandler(ABC):
    """单个任务阶段的处理器。

    生命周期：on_enter → [tick × N] → on_exit

    tick() 返回下一个阶段（或 None 表示停留在当前阶段）。
    """

    @abstractmethod
    def on_enter(self, ctx: PhaseContext) -> None:
        """阶段进入时调用一次。"""
        ...

    @abstractmethod
    def tick(self, ctx: PhaseContext) -> MissionPhase | None:
        """每帧调用。返回下一个阶段名即触发阶段切换。

        Returns:
            - MissionPhase 枚举值：切换到该阶段。
            - None：停留在当前阶段。
        """
        ...

    @abstractmethod
    def on_exit(self, ctx: PhaseContext) -> None:
        """阶段退出时调用一次（清理/记录）。"""
        ...

    @property
    @abstractmethod
    def phase(self) -> MissionPhase:
        """此处理器对应的阶段标识。"""
        ...
