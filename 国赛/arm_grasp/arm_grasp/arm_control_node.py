#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点2: 机械臂控制节点 (arm_control_node)
功能: 接收指令 → 控制JetArm舵机 → 执行抓取/放置/回零 → 发布反馈
比赛规则: 长条100×50×50mm, 50×100mm底面竖放, 从上方抓取

订阅: /arm/command (String), /arm/emergency_stop (Bool)
发布: /arm/feedback (String), /arm/state (String)
      /ros_robot_controller/bus_servo/set_position (ServosPosition)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition
import numpy as np
import yaml
import os
import time


class ArmControlNode(Node):
    """JetArm 机械臂控制"""

    # 舵机 ID
    SERVO = {'base': 1, 'shoulder': 2, 'elbow': 3,
             'wrist1': 4, 'wrist2': 5, 'wrist3': 6, 'gripper': 10}

    # 关节顺序
    JOINT_ORDER = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']

    def __init__(self):
        super().__init__('arm_control_node')

        # ── 加载配置 ──────────────────────────
        config_path = self.declare_parameter('config_path', '').value
        cfg = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

        # 关节限位
        lims = cfg.get('joint_limits', {})
        self.limits = {
            'base':    lims.get('base', [0, 1000]),
            'shoulder': lims.get('shoulder', [200, 800]),
            'elbow':   lims.get('elbow', [200, 800]),
            'wrist1':  lims.get('wrist1', [200, 800]),
            'wrist2':  lims.get('wrist2', [200, 800]),
            'wrist3':  lims.get('wrist3', [200, 800]),
            'gripper': lims.get('gripper', [100, 400]),
        }

        # 抓取策略
        g = cfg.get('grasp_strategy', {})
        self.approach_h = g.get('approach_height', 0.20)
        self.pre_grasp_offset = g.get('pre_grasp_offset', 0.10)
        self.grasp_depth = g.get('grasp_depth_offset', 0.02)
        self.lift_h = g.get('lift_height', 0.20)
        self.depart_h = g.get('depart_height', 0.25)

        # 高台安全高度 (摄像头不能撞平台, 否则机器狗会翻)
        pf = cfg.get('platform_config', {})
        self.platform_z = pf.get('height', 0.22)  # 平台上表面距底座高度

        # 姿态偏置: 避免大臂绷直
        pb = cfg.get('posture_bias', {})
        self.shoulder_back_bias = pb.get('shoulder_back', 80)
        self.elbow_up_bias = pb.get('elbow_up', 40)

        # 夹爪
        grp = cfg.get('gripper', {})
        self.gripper_open = grp.get('open', 400)
        self.gripper_close = grp.get('close', 100)

        # 连杆参数 (m) — JetArm 实测
        self.L1 = 0.18   # 上臂 (肩→肘)
        self.L2 = 0.16   # 前臂 (肘→腕)
        self.L3 = 0.18   # 腕部到爪子最前端
        self.shoulder_h = 0.06  # 肩关节距基座底部高度

        # 状态
        self.state = 'idle'
        self.holding = False
        self.emergency = False

        # 发布: 舵机命令 → ros_robot_controller 桥接节点
        self.serial_pub = self.create_publisher(
            ServosPosition, '/ros_robot_controller/bus_servo/set_position', 10)

        # 订阅
        self.create_subscription(String, '/arm/command', self._cmd_cb, 10)
        self.create_subscription(Bool, '/arm/emergency_stop', self._stop_cb, 10)

        # 发布
        self.fb_pub = self.create_publisher(String, '/arm/feedback', 10)
        self.state_pub = self.create_publisher(String, '/arm/state', 10)

        self.get_logger().info('[机械臂节点] JetArm 就绪')

    # ── 命令分发 ─────────────────────────────

    def _cmd_cb(self, msg):
        if self.emergency:
            self._fb('emergency', False, '急停状态')
            return

        parts = msg.data.split('|')
        cmd = parts[0].strip()

        # 解析参数 (格式: cmd|x|y|z|angle|duration|cx|cy|base)
        x = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        y = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        z = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
        angle = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
        dur = float(parts[5]) if len(parts) > 5 and parts[5] else 2.0
        cx = int(parts[6]) if len(parts) > 6 and parts[6] else 320
        cy = int(parts[7]) if len(parts) > 7 and parts[7] else 240
        base = int(parts[8]) if len(parts) > 8 and parts[8] else 512

        self.get_logger().info(f'▶ {cmd} x={x:.3f} y={y:.3f} z={z:.3f} '
                               f'angle={angle:.0f}° dur={dur:.1f}s')

        ok = False
        if cmd == 'move_to':
            ok = self._move_to(x, y, z, angle, dur)
        elif cmd == 'grasp':
            ok = self._grasp(x, y, z, angle, dur)
        elif cmd == 'direct_grasp':
            ok = self._direct_grasp(x, y, z, angle, dur, cx, cy, base)
        elif cmd == 'place':
            ok = self._place(x, y, z, dur)
        elif cmd == 'home':
            ok = self._home()
        elif cmd == 'open_gripper':
            ok = self._set_gripper(self.gripper_open)
        elif cmd == 'center_base':
            ok = self._center_base(cx)
        elif cmd == 'close_gripper':
            ok = self._set_gripper(self.gripper_close)
        elif cmd == 'stop':
            ok = self._stop()

        self._fb(cmd, ok, '完成' if ok else '失败')

    def _stop_cb(self, msg):
        if msg.data:
            self.emergency = True
            self._stop()
            self.state_pub.publish(String(data='error'))
            self.get_logger().warn('!!! 急停 !!!')

    # ── 底层舵机控制 ─────────────────────────

    def _clamp(self, joint_name, pos):
        """关节限位"""
        lo, hi = self.limits.get(joint_name, [0, 1000])
        return max(lo, min(hi, int(pos)))

    def _servo(self, sid, pos, duration=1.0):
        """发送单舵机命令"""
        p = self._clamp(self._id_to_name(sid), pos)
        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position = [ServoPosition(id=sid, position=p)]
        self.serial_pub.publish(msg)

    def _servos(self, id_pos_pairs, duration=2.0):
        """批量发送舵机命令 (所有舵机同步运动)"""
        msg = ServosPosition()
        msg.duration = float(duration)
        msg.position = []
        for sid, pos in id_pos_pairs:
            p = self._clamp(self._id_to_name(sid), pos)
            msg.position.append(ServoPosition(id=sid, position=p))
        self.serial_pub.publish(msg)
        self.get_logger().debug(f'舵机: {[(x.id, x.position) for x in msg.position]}')

    def _joints(self, positions, duration=2.0):
        """批量移动 6 个关节 (positions 按 JOINT_ORDER 顺序)"""
        pairs = []
        for i, name in enumerate(self.JOINT_ORDER):
            if i < len(positions):
                pairs.append((self.SERVO[name], positions[i]))
        self._servos(pairs, duration)
        time.sleep(duration + 0.3)

    def _id_to_name(self, sid):
        for name, id_ in self.SERVO.items():
            if id_ == sid:
                return name
        return 'base'

    # ── 逆运动学 (IK) ────────────────────────

    def _ik(self, x, y, z, angle=0.0):
        """
        JetArm 6DOF IK — 两遍求解, 补偿18cm腕长
        输入: (x,y,z) = 爪尖目标在基座坐标系(m)
        """
        base = 500 + int(np.arctan2(y, x) * 500.0 / np.pi)
        d_xy = np.sqrt(x**2 + y**2)
        h = z - self.shoulder_h

        # 第一遍: 腕长=0, 估算小臂方向
        d1 = min(d_xy, self.L1 + self.L2 - 0.005)
        s1, e1 = self._solve_2link(d1, h)
        fa = s1 + e1

        # 第二遍: 扣掉腕长投影, 让爪尖正好到目标
        d2 = d_xy - self.L3 * np.cos(fa)
        h2 = h    - self.L3 * np.sin(fa)
        d2 = max(0.02, min(d2, self.L1 + self.L2 - 0.005))

        s2, e2 = self._solve_2link(d2, h2)

        shoulder = 500 + int(s2 * 500.0 / np.pi)
        elbow    = 500 + int(e2 * 500.0 / np.pi)
        wrist1   = 500 - int((s2 + e2) * 500.0 / np.pi)

        shoulder = self._clamp('shoulder', shoulder)
        elbow    = self._clamp('elbow', elbow)
        wrist1   = self._clamp('wrist1', wrist1)
        wrist2 = 522
        wrist3 = 500

        self.get_logger().info(
            f'IK: ({x:.3f},{y:.3f},{z:.3f}) d={d_xy:.3f} h={h:.3f} '
            f'→ d2={d2:.3f} h2={h2:.3f} [{base},{shoulder},{elbow},{wrist1}]')

        return [base, shoulder, elbow, wrist1, wrist2, wrist3]

    def _solve_2link(self, d, h):
        """2连杆逆运动学, 返回 (肩角, 肘角) 单位rad"""
        cos_e = (d**2 + h**2 - self.L1**2 - self.L2**2) / (2.0 * self.L1 * self.L2)
        cos_e = max(-1.0, min(1.0, cos_e))
        e = np.arccos(cos_e)
        s = np.arctan2(h, max(d, 0.001)) - \
            np.arctan2(self.L2 * np.sin(e), self.L1 + self.L2 * np.cos(e))
        return s, e

    # ── 动作 ─────────────────────────────────

    def _move_to(self, x, y, z, angle, dur):
        """移动到目标位置"""
        self._set_state('moving')
        try:
            j = self._ik(x, y, z, angle)
            self._joints(j, max(dur, 1.0))
            self._set_state('idle')
            return True
        except Exception as e:
            self.get_logger().error(f'移动失败: {e}')
            self._set_state('error')
            return False

    def _grasp(self, x, y, z, angle, dur):
        """
        抓取流程 (比赛标准):
        1. 开夹爪
        2. 移动到物体上方 (approach)
        3. 垂直下降到抓取点
        4. 闭合夹爪
        5. 提升并悬空保持 3 秒
        6. 提升到搬运高度
        """
        self._set_state('grasping')
        try:
            # 1. 开夹爪
            self._set_gripper(self.gripper_open)
            time.sleep(0.3)

            # 2. 预抓取: 物体上方 (至少0.20m高，保证大臂朝前)
            az = max(z + self.pre_grasp_offset, 0.20)
            self._move_to(x, y, az, angle, dur)
            time.sleep(0.3)

            # 3. 下降到抓取点 (稍深一点确保夹紧)
            gz = z + self.grasp_depth
            self._move_to(x, y, gz, angle, dur * 0.6)
            time.sleep(0.2)

            # 4. 闭合夹爪
            self._set_gripper(self.gripper_close)
            time.sleep(0.5)
            self.holding = True

            # 5. 提升一点 (保持垂直 → 验证悬空)
            lz = z + self.lift_h
            self._move_to(x, y, lz, angle, dur * 0.6)
            time.sleep(0.3)

            # 6. 悬空验证 3 秒；验证结束后继续夹紧，直到收到 place 指令才释放。
            self.get_logger().info('★★★ 悬空验证 3 秒，之后保持夹紧等待放置指令 ★★★')
            time.sleep(3.0)
            self._set_gripper(self.gripper_close)

            # 7. 搬运高度
            dz = z + self.depart_h
            self._move_to(x, y, dz, angle, dur * 0.6)
            time.sleep(0.3)

            self._set_state('idle')
            self.get_logger().info('✓ 抓取完成，夹爪保持闭合')
            return True
        except Exception as e:
            self.get_logger().error(f'抓取失败: {e}')
            self.holding = False
            self._set_state('error')
            return False

    def _direct_grasp(self, x, y, z, angle, dur, cx=320, cy=240, base_target=512):
        """
        ★ 直抓: 使用task_manager指定的底座位置, 调肩肘腕去抓
        """
        self._set_state('grasping')
        try:
            # 1. 开夹爪
            self._set_gripper(self.gripper_open)
            time.sleep(0.3)

            # ═══ 底座使用传入值(默认512), 调肩肘腕去抓 ═══

            shoulder_target = 500 - int(x * 700) - 25
            shoulder_target = max(200, min(500, shoulder_target))

            elbow_rise = int((500 - shoulder_target) * 0.8)
            elbow_target = 200 + elbow_rise
            elbow_target = max(200, min(600, elbow_target))

            wrist1_adjust = int((240 - cy) * 0.3)
            wrist1_target = 530 + wrist1_adjust
            if wrist1_target < 480:
                elbow_target += int((480 - wrist1_target) * 0.6)
            wrist1_target = max(470, min(550, wrist1_target))

            self.get_logger().info(
                f'直抓: 底座={base_target}(不动) '
                f'肩={shoulder_target} 肘={elbow_target} 腕1={wrist1_target}')
            self._joints([base_target, shoulder_target, elbow_target,
                          wrist1_target, 522, 500], max(dur, 2.0))
            time.sleep(0.5)

            # 2. 夹紧
            self._set_gripper(self.gripper_close)
            time.sleep(0.5)
            self.holding = True

            # 2.5 大幅度抬高手臂+手腕, 防止剐蹭平台
            self.get_logger().info('直抓: 大幅抬高手臂防剐蹭...')
            self._joints([base_target, 500, 350, 520, 522, 500], 1.2)
            time.sleep(0.3)

            # 3. 拎回home
            self._joints([512, 500, 200, 350, 522, 500], 2.0)
            time.sleep(0.5)

            # 4. 悬空验证；验证结束后继续夹紧，直到收到 place 指令才释放。
            self.get_logger().info('★★★ 悬空验证 5 秒，之后保持夹紧等待放置指令 ★★★')
            time.sleep(5.0)
            self._set_gripper(self.gripper_close)

            self._set_state('idle')
            self.get_logger().info('✓ 直抓完成，夹爪保持闭合')
            return True
        except Exception as e:
            self.get_logger().error(f'直抓失败: {e}')
            self.holding = False
            self._set_state('error')
            return False

    def _place(self, x, y, z, dur):
        """
        放置流程:
        1. 移动到放置区上方
        2. 下降到放置高度
        3. 开夹爪释放
        4. 提升回上方
        """
        self._set_state('placing')
        try:
            # 1. 到放置区上方
            az = z + self.pre_grasp_offset
            self._move_to(x, y, az, 0, dur)
            time.sleep(0.3)

            # 2. 下降
            self._move_to(x, y, z + 0.03, 0, dur * 0.6)
            time.sleep(0.3)

            # 3. 释放
            self._set_gripper(self.gripper_open)
            time.sleep(0.5)
            self.holding = False

            # 4. 退回上方
            self._move_to(x, y, az, 0, dur * 0.6)
            time.sleep(0.3)

            self._set_state('idle')
            self.get_logger().info('✓ 放置完成')
            return True
        except Exception as e:
            self.get_logger().error(f'放置失败: {e}')
            self._set_state('error')
            return False

    def _home(self):
        """回零 → 摄像头朝桌面姿态 (肩500=不过分前倾, 不挡视野)"""
        self._set_state('moving')
        self._joints([512, 500, 200, 350, 522, 500], 2.0)
        self._set_gripper(self.gripper_open)
        time.sleep(0.5)
        self._set_state('idle')
        self.get_logger().info('✓ 回零完成')
        return True

    def _center_base(self, cx):
        """只转底座使目标水平居中（供闭环居中循环调用）"""
        base_adjust = int((cx - 320) * 0.5)
        base_target = 512 + base_adjust
        base_target = max(200, min(800, base_target))
        self.get_logger().info(f'底座居中: cx={cx} → base={base_target}')
        self._joints([base_target, 500, 200, 350, 522, 500], 1.2)
        return True

    def _set_gripper(self, pos):
        """控制夹爪"""
        p = self._clamp('gripper', pos)
        self._servo(self.SERVO['gripper'], p, 0.5)
        time.sleep(0.3)
        return True

    def _stop(self):
        """急停: 所有关节回中位"""
        for n in self.JOINT_ORDER:
            self._servo(self.SERVO[n], 500, 0.1)
        return True

    # ── 辅助 ─────────────────────────────────

    def _set_state(self, s):
        self.state = s
        self.state_pub.publish(String(data=s))

    def _fb(self, cmd, ok, msg):
        s = 'success' if ok else 'fail'
        self.fb_pub.publish(String(data=f'{cmd}|{s}|{msg}'))
        self.get_logger().info(f'反馈: {cmd} → {s}')


def main():
    rclpy.init()
    node = ArmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
