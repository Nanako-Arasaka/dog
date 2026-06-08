"""相机硬件抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.config import CameraConfig
from core.types import CameraIntrinsics


class CameraGateway(ABC):
    """相机硬件抽象。

    实现：
    - RealSenseCamera (Intel RealSense D435/D455)
    - MockCamera        (返回固定测试图像，用于 CI/仿真)
    """

    @abstractmethod
    def start(self) -> None:
        """启动相机流。调用后方可获取帧。"""
        ...

    @abstractmethod
    def stop(self) -> None:
        """停止相机流，释放资源。"""
        ...

    @abstractmethod
    def get_rgb_frame(self) -> np.ndarray:
        """获取 RGB 图像。

        Returns:
            (H, W, 3) uint8 ndarray, BGR 或 RGB 取决于实现约定。
        """
        ...

    @abstractmethod
    def get_depth_frame(self) -> np.ndarray:
        """获取深度图。

        Returns:
            (H, W) float32 ndarray，单位为米。无效点 = 0.0。
        """
        ...

    @abstractmethod
    def get_aligned_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """获取对齐后的 RGB + Depth 对。

        Returns:
            (rgb, depth): rgb 和 depth 逐像素对齐，尺寸一致。
        """
        ...

    @property
    @abstractmethod
    def intrinsics(self) -> CameraIntrinsics:
        """相机内参（标定后填入）。"""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """相机是否正在采集。"""
        ...


# ── Mock 实现 ────────────────────────────────────────────


class MockCamera(CameraGateway):
    """返回固定测试图像的相机。

    用于不连真机的开发调试和 CI 测试。
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._running = False
        self._intrinsics = CameraIntrinsics(
            fx=615.0, fy=615.0,
            cx=cfg.width / 2.0, cy=cfg.height / 2.0,
            width=cfg.width, height=cfg.height,
        )

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_rgb_frame(self) -> np.ndarray:
        return _dummy_rgb(self._cfg.height, self._cfg.width)

    def get_depth_frame(self) -> np.ndarray:
        return _dummy_depth(self._cfg.height, self._cfg.width)

    def get_aligned_frames(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_rgb_frame(), self.get_depth_frame()

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def is_running(self) -> bool:
        return self._running


def _dummy_rgb(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _dummy_depth(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.float32) * 1.5
