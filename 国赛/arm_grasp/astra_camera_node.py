#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Astra / Orbbec RGB-D 相机节点 — 真实深度替代伪深度。

RGB 仍通过 V4L2 UVC 读取；深度改为多后端真实获取，按顺序自动探测：

  1. pyorbbec  —— 官方 Orbbec Python SDK（最可靠，推荐现场安装）
                   Jetson 安装：pip install pyorbbecsdk
  2. openni    —— OpenNI2 + liborbbec 老驱动（Astra 真深度；绕开 pyorbbecsdk
                   对 iSerial=0 设备的枚举 bug，现场实测可用）
                   依赖: pip install openni + OPENNI2_REDIST(或自动探测 ~/openni2)
  3. uvc       —— V4L2 Z16 深度流（RGB 的相邻 video 设备，如 /dev/video1）
                  部分 Orbbec 型号（Astra Pro/Pro Plus 等）UVC 模式暴露 Z16 深度
  4. realsense —— 转发 RealSense 的 color + aligned depth + camera_info 到
                   /rgbd_cam/*（抓取视觉复用，零额外依赖；前提：RealSense
                   视角覆盖机械臂工作台面，cam2arm 手眼标定按 RealSense 重做）
  5. none      —— 后端全部不可用：默认【不再发布深度】，并循环报错提示。
                   抓取侧 vision_node 会得到 invalid_depth → 不动作（安全，不会用假深度去抓）。

行为变化（相对旧版伪深度 0.5m 常量）：
  - 旧版：深度恒 500mm，抓取 z 完全不可靠；
  - 新版：深度来自真实相机（mm，16UC1），z 方向随距离正确变化。
  - 若仍想用伪深度调试（不建议正式比赛用），显式传 --ros-args -p fake_depth_fallback:=true。

深度不可用时不会崩溃：节点继续发 RGB，深度话题停止发布/或按 fallback 开关处理。
"""

import glob
import os

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
        # 深度后端: auto | pyorbbec | openni | uvc | realsense | none
        self.depth_mode = str(self.declare_parameter('depth_mode', 'auto').value).lower()
        # UVC 深度设备号（None → 默认 camera_index + 1）
        self.depth_index = self.declare_parameter('depth_index', -1).value
        self.depth_width = self.declare_parameter('depth_width', 640).value
        self.depth_height = self.declare_parameter('depth_height', 480).value
        # 深度后端全部不可用时的兜底: true=发伪深度(仅调试), false=不发深度(安全)
        self.fake_depth_fallback = bool(self.declare_parameter('fake_depth_fallback', False).value)
        # 相机内参（默认近似值；openni/pyorbbec 后端会从驱动读到更准的值）
        self.camera_fx = float(self.declare_parameter('camera_fx', 570.34).value)
        self.camera_fy = float(self.declare_parameter('camera_fy', 570.34).value)
        self.camera_cx = float(self.declare_parameter('camera_cx', 320.0).value)
        self.camera_cy = float(self.declare_parameter('camera_cy', 240.0).value)
        # openni2 redist 路径（空=自动探测 ~/openni2/OpenNI-Linux-*/Redist 与 OPENNI2_REDIST）
        self.openni_redist = str(self.declare_parameter('openni_redist', '').value)
        # openni 后端内参（默认=现场 OpenNI2 实测 640x480;get_camera_params 不可用时兜底,可现场标定修改）
        self.openni_fx = float(self.declare_parameter('openni_fx', 945.028).value)
        self.openni_fy = float(self.declare_parameter('openni_fy', 945.028).value)
        self.openni_cx = float(self.declare_parameter('openni_cx', 320.0).value)
        self.openni_cy = float(self.declare_parameter('openni_cy', 400.0).value)
        # 是否把深度注册到 RGB 视角（默认关!实测注册后覆盖率暴跌 ~130 倍,见排障手册 4.6）
        self.openni_registration = bool(self.declare_parameter('openni_registration', False).value)
        # realsense 转发后端的话题
        self.realsense_color_topic = str(self.declare_parameter(
            'realsense_color_topic', '/camera/camera/color/image_raw').value)
        self.realsense_depth_topic = str(self.declare_parameter(
            'realsense_depth_topic', '/camera/camera/aligned_depth_to_color/image_raw').value)
        self.realsense_info_topic = str(self.declare_parameter(
            'realsense_info_topic', '/camera/camera/color/camera_info').value)

        self.bridge = CvBridge()
        self._openni2_state = None  # (device, depth_stream)，防止流对象被 GC 关闭

        # ── realsense 转发模式（不打开本地 V4L2，直接转发 RealSense 话题）──
        if self.depth_mode == 'realsense':
            self._init_realsense_mode()
            return

        # ── 打开 RGB 摄像头 (Orbbec USB 2.0 Camera) ──
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开 Astra RGB 摄像头 (index={self.camera_index})')
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
                    '或配置正确的 depth_index，或改用 depth_mode:=realsense。'
                    '抓取侧会报 invalid_depth 安全停止。'
                )

        # 每 2 秒重发 camera_info，确保 vision_node 能收到
        self._publish_camera_info()
        self.info_timer = self.create_timer(2.0, self._publish_camera_info)

        # 30fps 图像流
        self.timer = self.create_timer(0.033, self._timer)

    # ──────────────────────────────────────────────
    # realsense 转发后端
    # ──────────────────────────────────────────────
    def _init_realsense_mode(self) -> None:
        """转发 RealSense 的 color/aligned-depth/camera_info 到 /rgbd_cam/*。

        前提：RealSense 视角覆盖机械臂工作台面；cam2arm 手眼标定按 RealSense 重做。
        深度统一转成 16UC1(mm)，与 vision_node 的解码约定一致。
        """
        self._depth = None
        self._rs_info: CameraInfo | None = None
        self.create_subscription(Image, self.realsense_color_topic, self._on_rs_color, 10)
        self.create_subscription(Image, self.realsense_depth_topic, self._on_rs_depth, 10)
        self.create_subscription(CameraInfo, self.realsense_info_topic, self._on_rs_info, 10)
        self.info_timer = self.create_timer(2.0, self._publish_rs_info)
        self.get_logger().info(
            f'深度后端: realsense 转发 color={self.realsense_color_topic} '
            f'depth={self.realsense_depth_topic}'
        )

    def _on_rs_color(self, msg: Image) -> None:
        self.pub_color.publish(msg)

    def _on_rs_depth(self, msg: Image) -> None:
        enc = (msg.encoding or '').lower()
        try:
            if enc in ('32fc1', '32fc'):
                arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                arr16 = np.clip(arr * 1000.0, 0.0, 65535.0).astype(np.uint16)
                out = self.bridge.cv2_to_imgmsg(arr16, encoding='16UC1')
            elif enc in ('16uc1', 'mono16'):
                out = msg
            else:
                return
        except Exception:  # noqa: BLE001
            return
        out.header = msg.header
        self.pub_depth.publish(out)

    def _on_rs_info(self, msg: CameraInfo) -> None:
        self._rs_info = msg
        self.pub_info.publish(msg)

    def _publish_rs_info(self) -> None:
        if self._rs_info is not None:
            self.pub_info.publish(self._rs_info)

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
        if mode == 'auto' or mode == 'openni':
            fn = self._try_openni2()
            if fn is not None:
                self.get_logger().info('深度后端: openni2 (liborbbec)')
                return fn
            if mode == 'openni':
                self.get_logger().warn('openni2 不可用（未安装或无设备）')
                return None
        if mode == 'auto' or mode == 'uvc':
            fn = self._try_uvc_depth()
            if fn is not None:
                self.get_logger().info(f'深度后端: uvc Z16 (video{self._resolved_depth_index()})')
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
                        self.get_logger().info(f'已从 SDK 读取内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
            except Exception:
                pass
            return read
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'pyorbbecsdk 初始化失败: {exc}')
            return None

    def _try_openni2(self):
        """OpenNI2 + liborbbec 后端（Astra 真深度）。返回 ()=>np.uint16 深度帧 或 None。

        依赖: pip install openni；OPENNI2_REDIST 环境变量或自动探测 ~/openni2。
        若设备支持 image registration，尝试把深度注册到 RGB 视角（消除基线视差）。
        """
        try:
            from openni import openni2
        except ImportError:
            return None

        # 确保 OPENNI2_REDIST（环境变量 → 参数 → 自动探测 ~/openni2）
        if not os.environ.get("OPENNI2_REDIST"):
            redist = self.openni_redist
            if not redist:
                for base in glob.glob(os.path.expanduser("~/openni2/OpenNI-Linux-*")):
                    candidate = os.path.join(base, "Redist")
                    if os.path.isdir(candidate):
                        redist = candidate
                        break
            if redist and os.path.isdir(redist):
                os.environ["OPENNI2_REDIST"] = redist
                self.get_logger().info(f'OPENNI2_REDIST -> {redist}')

        try:
            openni2.initialize()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'openni2 initialize 失败: {exc}')
            return None
        try:
            dev = openni2.Device.open_any()
        except Exception as exc:  # noqa: BLE001
            openni2.unload()
            self.get_logger().warn(f'openni2 未发现 Astra 设备: {exc}')
            return None

        # 深度注册到 RGB 视角（默认关闭:实测注册后覆盖率暴跌 ~130 倍;
        # 开启可消除 RGB-IR 视差,但仅当覆盖率可接受时用）
        if self.openni_registration:
            try:
                if hasattr(dev, "set_image_registration_mode"):
                    dev.set_image_registration_mode(True)
                    self.get_logger().info('openni 深度已注册到 RGB 视角')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'openni 注册模式设置失败: {exc}')

        try:
            depth = dev.create_depth_stream()
            depth.start()
            frame = depth.read_frame()
            if frame is None:
                depth.stop()
                dev.close()
                openni2.unload()
                return None
            # 从驱动读深度内参（IR 视角；若已注册到 RGB 则接近 RGB 内参）
            # 优先 SDK get_camera_params，但必须过合理性校验（现场实测该 API 曾返回
            # 异常 cx=640），失败则回退显式参数 openni_fx/fy/cx/cy（默认=现场实测值）。
            sdk_intrinsics = None
            try:
                params = depth.get_camera_params()
                sdk_intrinsics = (
                    float(params.fx), float(params.fy), float(params.cx), float(params.cy)
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'get_camera_params 失败: {exc}，回退参数内参')
            if sdk_intrinsics is not None:
                fx, fy, cx, cy = sdk_intrinsics
                w = frame.width
                h = frame.height
                # 主点合理范围:cx 应接近图像中心(cy=400 是 Astra IR 传感器稳定特性,放宽上限)
                plausible = (
                    fx > 100.0 and fy > 100.0
                    and w * 0.2 <= cx <= w * 0.8
                    and h * 0.1 <= cy <= h * 0.95
                )
                if plausible:
                    self.camera_fx, self.camera_fy = fx, fy
                    self.camera_cx, self.camera_cy = cx, cy
                    self.get_logger().info(
                        f'OpenNI2 SDK 内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}'
                    )
                else:
                    self.get_logger().warn(
                        f'OpenNI2 SDK 内参不合理 fx={fx:.1f} cx={cx:.1f} cy={cy:.1f}'
                        f'（图像 {w}x{h}），回退参数内参'
                    )
                    sdk_intrinsics = None
            if sdk_intrinsics is None:
                self.camera_fx = self.openni_fx
                self.camera_fy = self.openni_fy
                self.camera_cx = self.openni_cx
                self.camera_cy = self.openni_cy
                self.get_logger().info(
                    f'OpenNI2 使用参数内参 fx={self.openni_fx:.2f} fy={self.openni_fy:.2f} '
                    f'cx={self.openni_cx:.2f} cy={self.openni_cy:.2f}'
                )

            def read():
                try:
                    f = depth.read_frame()
                    if f is None:
                        return None
                    buf = f.get_buffer_as_uint16()
                    if buf is None or len(buf) == 0:
                        return None
                    return np.frombuffer(buf, dtype=np.uint16).reshape(f.height, f.width)
                except Exception:  # noqa: BLE001
                    return None

            first = read()
            if first is None or first.size == 0:
                depth.stop()
                dev.close()
                openni2.unload()
                return None
            self._openni2_state = (dev, depth)
            return read
        except Exception as exc:  # noqa: BLE001
            try:
                depth.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass
            openni2.unload()
            self.get_logger().warn(f'openni2 初始化失败: {exc}')
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
                self.get_logger().warn(f'深度读取失败: {exc}', throttle_duration_sec=5)
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
        if self._openni2_state is not None:
            try:
                _dev, depth = self._openni2_state
                depth.stop()
            except Exception:  # noqa: BLE001
                pass
            self._openni2_state = None


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
