#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Astra / Orbbec RGB-D 相机节点 — 真实深度替代伪深度。

RGB 仍通过 V4L2 UVC 读取；深度改为多后端真实获取，按顺序自动探测：

  1. pyorbbec  —— 官方 Orbbec Python SDK（最可靠，推荐现场安装）
                   Jetson 安装：pip install pyorbbecsdk
  2. uvc       —— V4L2 Z16 深度流（RGB 的相邻 video 设备，如 /dev/video1）
                  部分 Orbbec 型号（Astra Pro/Pro Plus 等）UVC 模式暴露 Z16 深度
  3. none      —— 后端全部不可用：默认【不再发布深度】，并循环报错提示。
                   抓取侧 vision_node 会得到 invalid_depth → 不动作（安全，不会用假深度去抓）。

行为变化（相对旧版伪深度 0.5m 常量）：
  - 旧版：深度恒 500mm，抓取 z 完全不可靠；
  - 新版：深度来自真实相机（mm，16UC1），z 方向随距离正确变化。
  - 若仍想用伪深度调试（不建议正式比赛用），显式传 --ros-args -p fake_depth_fallback:=true。

深度不可用时不会崩溃：节点继续发 RGB，深度话题停止发布/或按 fallback 开关处理。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class AstraCameraNode(Node):
    def __init__(self):
        super().__init__('astra_camera_node')

        # 发布 RGB 图像
        self.pub_color = self.create_publisher(Image, '/rgbd_cam/color/image_rect_color', 10)
        # 发布深度图像（真实深度，16UC1，单位 mm）
        self.pub_depth = self.create_publisher(Image, '/rgbd_cam/depth/image_raw', 10)
        # 发布相机内参
        self.pub_info = self.create_publisher(CameraInfo, '/rgbd_cam/color/camera_info', 10)

        # ── 参数 ──
        # 相机节点号（默认 0 = /dev/video0 = Orbbec RGB UVC）
        self.camera_index = self.declare_parameter('camera_index', 0).value
        # 深度后端: auto | pyorbbec | uvc | none
        self.depth_mode = str(self.declare_parameter('depth_mode', 'auto').value).lower()
        # UVC 深度设备号（None → 默认 camera_index + 1）
        self.depth_index = self.declare_parameter('depth_index', -1).value
        self.depth_width = self.declare_parameter('depth_width', 640).value
        self.depth_height = self.declare_parameter('depth_height', 480).value
        # 深度后端全部不可用时的兜底: true=发伪深度(仅调试), false=不发深度(安全)
        self.fake_depth_fallback = bool(self.declare_parameter('fake_depth_fallback', False).value)
        # 相机内参（默认近似值；pyorbbec 后端可能从 SDK 读到更准的值）
        self.camera_fx = float(self.declare_parameter('camera_fx', 570.34).value)
        self.camera_fy = float(self.declare_parameter('camera_fy', 570.34).value)
        self.camera_cx = float(self.declare_parameter('camera_cx', 320.0).value)
        self.camera_cy = float(self.declare_parameter('camera_cy', 240.0).value)

        self.bridge = CvBridge()

        # ── 打开 RGB 摄像头 (Orbbec USB 2.0 Camera) ──
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error('无法打开 Astra RGB 摄像头 (index=%s)', self.camera_index)
            return
        self.get_logger().info('Astra RGB 摄像头已打开 (640x480)')

        # ── 初始化真实深度后端 ──
        self._depth = self._init_depth_backend()
        if self._depth is None:
            if self.fake_depth_fallback:
                self.get_logger().warn('真实深度不可用，回退伪深度(仅调试，正式比赛请勿用)!')
            else:
                self.get_logger().error(
                    '真实深度后端全部不可用：将不发布深度。'
                    '现场请安装 pyorbbecsdk (pip install pyorbbecsdk) '
                    '或配置正确的 depth_index。抓取侧会报 invalid_depth 安全停止。'
                )

        # 每 2 秒重发 camera_info，确保 vision_node 能收到
        self._publish_camera_info()
        self.info_timer = self.create_timer(2.0, self._publish_camera_info)

        # 30fps 图像流
        self.timer = self.create_timer(0.033, self._timer)

    # ──────────────────────────────────────────────
    # 深度后端
    # ──────────────────────────────────────────────
    def _init_depth_backend(self):
        """按 depth_mode 探测可用的真实深度后端，返回取帧函数或 None。"""
        mode = self.depth_mode
        if mode == 'auto' or mode == 'pyorbbec':
            fn = self._try_pyorbbec()
            if fn is not None:
                self.get_logger().info('深度后端: pyorbbecsdk')
                return fn
            if mode == 'pyorbbec':
                self.get_logger().warn('pyorbbecsdk 不可用（未安装或无设备）')
                return None
        if mode == 'auto' or mode == 'uvc':
            fn = self._try_uvc_depth()
            if fn is not None:
                self.get_logger().info('深度后端: uvc Z16 (video%s)', self._resolved_depth_index())
                return fn
            if mode == 'uvc':
                self.get_logger().warn('uvc 深度流不可用')
                return None
        if mode == 'none':
            self.get_logger().warn('depth_mode=none，不发布深度')
        return None

    def _resolved_depth_index(self):
        if isinstance(self.depth_index, int) and self.depth_index >= 0:
            return self.depth_index
        return self.camera_index + 1

    def _try_pyorbbec(self):
        """官方 Orbbec SDK 后端。返回 ()=>np.uint16 深度帧 或 None。"""
        try:
            from pyorbbecsdk import (  # type: ignore
                Context, OBSensorType, Config,
            )
        except ImportError:
            return None
        try:
            ctx = Context()
            device_list = ctx.query_devices()
            if device_list is None or device_list.get_device_count() == 0:
                return None
            dev = device_list.get_device_by_index(0)
            depth_sensor = dev.get_sensor(OBSensorType.DEPTH_SENSOR)
            if depth_sensor is None:
                return None
            profile_list = depth_sensor.get_stream_profile_list()
            profile = profile_list.get_default_video_stream_profile()
            cfg = Config()
            cfg.enable_stream(profile)
            depth_sensor.start(cfg)

            def read():
                frame = depth_sensor.read_frame(timeout_ms=1000)
                if frame is None:
                    return None
                w = frame.get_width()
                h = frame.get_height()
                data = frame.get_data()  # numpy, 16UC1, 单位 mm
                if data is None or data.size == 0:
                    return None
                arr = np.asarray(data, dtype=np.uint16).reshape(h, w)
                return arr

            # 预热一帧，确认真的能出数
            first = read()
            if first is None:
                depth_sensor.stop()
                return None
            # pyorbbec 可用时尝试从 SDK 取内参（部分型号支持）
            try:
                intrinsic = dev.get_calibration_camera_param(profile)
                if intrinsic is not None:
                    fx = float(intrinsic.fx)
                    fy = float(intrinsic.fy)
                    cx = float(intrinsic.cx)
                    cy = float(intrinsic.cy)
                    if fx > 0:
                        self.camera_fx, self.camera_fy = fx, fy
                        self.camera_cx, self.camera_cy = cx, cy
                        self.get_logger().info('已从 SDK 读取内参 fx=%.2f fy=%.2f cx=%.2f cy=%.2f', fx, fy, cx, cy)
            except Exception:
                pass
            return read
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn('pyorbbecsdk 初始化失败: %s', exc)
            return None

    def _try_uvc_depth(self):
        """V4L2 Z16 深度流后端。返回 ()=>np.uint16 深度帧 或 None。"""
        idx = self._resolved_depth_index()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Z16 '))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.depth_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.depth_height)

        def read():
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            if frame.ndim == 2 and frame.dtype == np.uint16:
                return frame
            if frame.ndim == 3 and frame.shape[2] == 3:
                # 驱动把它当 RGB 解了，尝试按内存重新解释为 uint16
                try:
                    raw = np.frombuffer(frame.tobytes(), dtype=np.uint16)
                    return raw.reshape(frame.shape[0], frame.shape[1])
                except Exception:
                    return None
            return None

        # 预热一帧
        first = read()
        if first is None:
            cap.release()
            return None
        return read

    # ──────────────────────────────────────────────
    # 发布
    # ──────────────────────────────────────────────
    def _publish_camera_info(self):
        """发布相机内参（pyorbbec 后端读到 SDK 内参时用真实值，否则用参数）。"""
        info = CameraInfo()
        info.header.frame_id = 'camera_link'
        info.width = 640
        info.height = 480
        info.k = [self.camera_fx, 0.0, self.camera_cx,
                  0.0, self.camera_fy, self.camera_cy,
                  0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.pub_info.publish(info)

    def _timer(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('RGB 图像读取失败', throttle_duration_sec=5)
            return

        # 发布 RGB
        color_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        color_msg.header.stamp = self.get_clock().now().to_msg()
        color_msg.header.frame_id = 'camera_link'
        self.pub_color.publish(color_msg)

        # 发布真实深度（16UC1, mm）
        depth_img = None
        if self._depth is not None:
            try:
                depth_img = self._depth()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn('深度读取失败: %s', exc, throttle_duration_sec=5)
                depth_img = None

        if depth_img is None or depth_img.size == 0:
            if self.fake_depth_fallback:
                # 仅调试兜底：伪深度 0.5m（正式比赛必须关闭此开关）
                h, w = frame.shape[:2]
                depth_img = np.full((h, w), 500, dtype=np.uint16)
            else:
                self.get_logger().warn('深度帧不可用，跳过发布（抓取侧将安全停止）', throttle_duration_sec=5)
                return

        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.frame_id = 'camera_link'
        self.pub_depth.publish(depth_msg)

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    rclpy.init()
    node = AstraCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
