"""perception —— 感知层。

提供统一的感知接口，内部编排多个检测器。
"""

from perception.detector.base import BaseDetector
from perception.gateway import PerceptionGateway
from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway

__all__ = [
    "PerceptionGateway",
    "RemotePerceptionGateway",
    "RemotePerceptionConfig",
    "BaseDetector",
]
