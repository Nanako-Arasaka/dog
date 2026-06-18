#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 串口桥接节点 —— 订阅官方 topic，调用官方 SDK 驱动 STM32 控制板。

订阅: /ros_robot_controller/bus_servo/set_position (ServosPosition)
服务: /ros_robot_controller/init_finish (Trigger)
"""

import rclpy
from rclpy.node import Node
from ros_robot_controller_msgs.msg import ServosPosition
from std_srvs.srv import Trigger
import sys
import os
import time

# 添加 home 目录到 Python 路径，导入官方底层 SDK
sys.path.insert(0, os.path.expanduser('~'))
from ros_robot_controller_sdk import Board


class SerialBridgeNode(Node):
    """ROS2 ↔ STM32 串口桥接"""

    PUMP_ANGLE_UP = 200   # 气泵角度: 上
    PUMP_ANGLE_DOWN = 500  # 气泵角度: 下

    def __init__(self):
        super().__init__('serial_bridge_node')

        device = self.declare_parameter('device', '/dev/ttyUSB0').value
        baudrate = self.declare_parameter('baudrate', 1000000).value

        self.get_logger().info(f'连接串口: {device} @ {baudrate}')
        try:
            self.board = Board(device=device, baudrate=baudrate, timeout=2)
            self.board.enable_reception(True)
            time.sleep(0.5)
            self.get_logger().info('✅ 串口连接成功，STM32 控制板就绪')
        except Exception as e:
            self.get_logger().error(f'❌ 串口连接失败: {e}')
            self.board = None

        # 订阅: 总线舵机位置控制
        self.create_subscription(
            ServosPosition,
            '/ros_robot_controller/bus_servo/set_position',
            self._set_position_cb,
            10
        )

        # 服务: 初始化完成信号
        self.create_service(Trigger, '/ros_robot_controller/init_finish',
                            self._init_finish_cb)

        # 订阅: 蜂鸣器
        from std_msgs.msg import String
        self.create_subscription(
            String,
            '/ros_robot_controller/buzzer/set_buzzer',
            self._buzzer_cb,
            10
        )

        self.get_logger().info('✅ 串口桥接节点就绪')

    # ── 初始化服务 ──────────────────────────────

    def _init_finish_cb(self, request, response):
        response.success = self.board is not None
        response.message = 'ok' if self.board else 'serial not connected'
        return response

    # ── 舵机控制 ────────────────────────────────

    def _set_position_cb(self, msg):
        if self.board is None:
            self.get_logger().warn('串口未连接，忽略舵机指令')
            return

        positions = []
        for sp in msg.position:
            p = max(0, min(1000, int(sp.position)))
            positions.append([int(sp.id), p])

        if positions:
            try:
                self.board.bus_servo_set_position(float(msg.duration), positions)
                self.get_logger().debug(
                    f'舵机: dur={msg.duration:.1f}s, pos={[[p[0], p[1]] for p in positions]}')
            except Exception as e:
                self.get_logger().error(f'舵机命令失败: {e}')

    # ── 蜂鸣器 ──────────────────────────────────

    def _buzzer_cb(self, msg):
        """格式: "freq,duty,delay,repeat" 如 "1900,0.1,0.05,1" """
        if self.board is None:
            return
        try:
            parts = msg.data.split(',')
            freq = int(parts[0]) if len(parts) > 0 else 1900
            duty = float(parts[1]) if len(parts) > 1 else 0.1
            delay = float(parts[2]) if len(parts) > 2 else 0.05
            repeat = int(parts[3]) if len(parts) > 3 else 1
            self.board.set_buzzer(freq, duty, delay, repeat)
        except Exception as e:
            self.get_logger().error(f'蜂鸣器命令失败: {e}')


def main():
    rclpy.init()
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
