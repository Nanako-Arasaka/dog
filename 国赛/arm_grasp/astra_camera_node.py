#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Astra RGB-D 相机节点 — USB 方式驱动"""
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
        # 发布深度图像 (伪深度, 等 Orbbec SDK 装好后再替换)
        self.pub_depth = self.create_publisher(Image, '/rgbd_cam/depth/image_raw', 10)
        # 发布相机内参
        self.pub_info = self.create_publisher(CameraInfo, '/rgbd_cam/color/camera_info', 10)

        self.bridge = CvBridge()

        # 打开 RGB 摄像头 (Orbbec USB 2.0 Camera)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self.get_logger().error('无法打开 Astra RGB 摄像头')
            return

        self.get_logger().info('Astra RGB 摄像头已打开 (640x480)')

        # 每 2 秒重发 camera_info，确保 vision_node 能收到
        self._publish_camera_info()
        self.info_timer = self.create_timer(2.0, self._publish_camera_info)

        # 30fps 图像流
        self.timer = self.create_timer(0.033, self._timer)

    def _publish_camera_info(self):
        """发布 Astra 默认相机内参 (640x480)"""
        info = CameraInfo()
        info.header.frame_id = 'camera_link'
        info.width = 640
        info.height = 480
        # Astra 默认内参 (近似值, 需实测校准)
        info.k = [570.34, 0.0, 320.0,
                  0.0, 570.34, 240.0,
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

        # 发布伪深度 (全 0.5m, 等真实深度驱动就绪后替换)
        h, w = frame.shape[:2]
        fake_depth = np.full((h, w), 500, dtype=np.uint16)  # 500mm = 0.5m
        depth_msg = self.bridge.cv2_to_imgmsg(fake_depth, encoding='16UC1')
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
