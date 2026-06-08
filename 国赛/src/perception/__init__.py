"""perception —— 感知层。

提供统一的感知接口，内部编排多个检测器。
"""

from perception.detector.base import BaseDetector
from perception.gateway import PerceptionGateway

__all__ = ["PerceptionGateway", "BaseDetector"]
