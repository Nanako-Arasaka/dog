"""hardware.camera —— 相机抽象。"""

from hardware.camera.interface import CameraGateway, MockCamera

__all__ = ["CameraGateway", "MockCamera"]
