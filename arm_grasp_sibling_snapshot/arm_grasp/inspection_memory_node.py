#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点3: 巡检记忆节点 (inspection_memory_node)
功能: 接收巡检结果 → 记录A/B/C/D区域状态 → 发布异常区域列表

订阅: /inspection/all (String) "A:abnormal,B:normal,C:abnormal,D:normal"
      /inspection/reset (Bool)
发布: /inspection/target_zones (String) "A,C"
      /inspection/all_zones (String)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class InspectionMemoryNode(Node):

    def __init__(self):
        super().__init__('inspection_memory_node')
        self.zones = {}

        self.create_subscription(String, '/inspection/all', self._cb_all, 10)
        self.create_subscription(Bool, '/inspection/reset', self._cb_rst, 10)

        self.pub_target = self.create_publisher(String, '/inspection/target_zones', 10)
        self.pub_all = self.create_publisher(String, '/inspection/all_zones', 10)

        self.get_logger().info('[巡检记忆节点] 就绪，等待巡-检结果...')

    def _cb_all(self, msg):
        try:
            for p in msg.data.split(','):
                z, s = p.strip().split(':')
                z, s = z.strip().upper(), s.strip().lower()
                if z in 'ABCD' and s in ('normal', 'abnormal'):
                    self.zones[z] = s
            self.get_logger().info(f'巡检结果: {self.zones}')
            self._publish()
        except Exception as e:
            self.get_logger().error(f'解析失败: {e}')

    def _cb_rst(self, msg):
        if msg.data:
            self.zones.clear()
            self.get_logger().info('记忆已重置')
            self._publish()

    def _publish(self):
        abnormal = sorted([z for z, s in self.zones.items() if s == 'abnormal'])
        self.pub_target.publish(String(data=','.join(abnormal)))

        parts = []
        for z in 'ABCD':
            parts.append(f'{z}:{self.zones.get(z, "unknown")}')
        self.pub_all.publish(String(data=','.join(parts)))

        self.get_logger().info(f'异常区域: {abnormal}')


def main():
    rclpy.init()
    node = InspectionMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
