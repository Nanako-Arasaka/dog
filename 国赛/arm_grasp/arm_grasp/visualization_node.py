#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点5: 可视化节点 (visualization_node)
功能: 订阅各节点状态 → 叠加显示到图像 → 发布可视化结果

订阅: /vision/debug_image, /task/status, /arm/state, /inspection/target_zones
发布: /visualization/display (Image)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisualizationNode(Node):

    def __init__(self):
        super().__init__('visualization_node')
        self.bridge = CvBridge()

        self.task_state = 'WAITING'
        self.arm_state = 'idle'
        self.grasp_cnt = 0
        self.drop_cnt = 0
        self.max_drops = 3
        self.targets = []
        self.debug_img = None

        self.create_subscription(Image, '/vision/debug_image', self._cb_img, 10)
        self.create_subscription(String, '/task/status', self._cb_task, 10)
        self.create_subscription(String, '/arm/state', self._cb_arm, 10)
        self.create_subscription(String, '/inspection/target_zones', self._cb_tgt, 10)

        self.pub_disp = self.create_publisher(Image, '/visualization/display', 10)
        self.create_timer(0.1, self._timer)
        self.get_logger().info('[可视化节点] 就绪')

    def _cb_img(self, msg):
        try:
            self.debug_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            pass

    def _cb_task(self, msg):
        try:
            for p in msg.data.split('|'):
                p = p.strip()
                if p.startswith('状态:'):
                    self.task_state = p.split(':')[1]
                elif p.startswith('抓取:'):
                    gc = p.split(':')[1].split('/')
                    self.grasp_cnt = int(gc[0])
                elif p.startswith('掉落:'):
                    dc = p.split(':')[1].split('/')
                    self.drop_cnt = int(dc[0])
                    self.max_drops = int(dc[1])
                elif p.startswith('目标:'):
                    tz = p.split(':')[1]
                    self.targets = [x for x in tz.strip("[]'").split(',') if x]
        except:
            pass

    def _cb_arm(self, msg):
        self.arm_state = msg.data

    def _cb_tgt(self, msg):
        t = msg.data.strip()
        self.targets = [z.strip() for z in t.split(',')] if t else []

    def _timer(self):
        if self.debug_img is not None:
            d = self.debug_img.copy()
        else:
            d = np.zeros((480, 640, 3), np.uint8) + 30
        self._draw(d)
        try:
            self.pub_disp.publish(self.bridge.cv2_to_imgmsg(d, 'bgr8'))
        except:
            pass

    def _draw(self, img):
        colors = {'WAITING': (0, 255, 255), 'GRASPING': (0, 165, 255),
                  'PLACING': (255, 100, 0), 'COMPLETED': (0, 255, 0), 'ERROR': (0, 0, 255)}
        o = img.copy()
        cv2.rectangle(o, (5, 5), (380, 155), (0, 0, 0), -1)
        cv2.addWeighted(o, 0.7, img, 0.3, 0, img)

        f, y = cv2.FONT_HERSHEY_SIMPLEX, 22
        c = colors.get(self.task_state, (255, 255, 255))
        cv2.putText(img, f'Task: {self.task_state}', (12, y), f, 0.5, c, 2)
        y += 22
        cv2.putText(img, f'Arm: {self.arm_state}', (12, y), f, 0.5,
                   (0, 255, 0) if self.arm_state == 'idle' else (0, 165, 255), 2)
        y += 22
        cv2.putText(img, f'Grasp: {self.grasp_cnt}/2', (12, y), f, 0.5, (255, 255, 255), 2)
        y += 22
        cv2.putText(img, f'Drop: {self.drop_cnt}/{self.max_drops}', (12, y), f, 0.5,
                   (0, 0, 255) if self.drop_cnt > 0 else (255, 255, 255), 2)
        y += 22
        tgt = ','.join(self.targets) if self.targets else 'None'
        cv2.putText(img, f'Target: {tgt}', (12, y), f, 0.5, (255, 255, 255), 2)
        y += 22
        score = self.grasp_cnt * 25 - min(self.drop_cnt * 5, 10)
        cv2.putText(img, f'Score: {score}/50', (12, y), f, 0.5, (0, 255, 255), 2)


def main():
    rclpy.init()
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
