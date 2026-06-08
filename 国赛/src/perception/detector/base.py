"""检测器基类 —— 所有 CV 检测器的统一接口。

每个检测器负责一个特定识别任务，可独立训练/替换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

# 泛型检测结果
T = TypeVar("T")


class BaseDetector(ABC, Generic[T]):
    """CV 检测器基类。

    所有检测器（锥桶/设备/仪表/OCR/长条）遵循此接口。
    实现：
    - YOLOConeDetector
    - YOLOEquipmentDetector
    - AnalogMeterReader
    - PaddleOCRLocalizer
    - YOLOStripDetector
    """

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型权重。

        Args:
            model_path: 模型文件路径（.pt / .onnx / .pdmodel 等）。
        """
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[T]:
        """对单帧图像执行推理。

        Args:
            image: (H, W, 3) uint8 BGR/RGB 图像。

        Returns:
            检测结果列表。
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已成功加载。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """检测器名称（用于日志/调试）。"""
        ...
