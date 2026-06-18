#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点4: 任务管理节点 (task_manager_node)
功能: 协调视觉 + 机械臂 → 完成国赛长条抓取搬运全流程

新增功能:
  ★ 边缘微调: 物体太靠左/右时, 小幅转底座使其靠近中央
  ★ 抓取验证: 抓取后用视觉二次检测, 通过z坐标变化判断是否真的夹到
  ★ 失败重试: 未夹到则小幅度转底座后重新抓取, 最多3次

订阅: /vision/grasp_pose, /inspection/target_zones, /placement/recognized_zone,
      /arm/feedback, /task/start, /task/reset
发布: /arm/command, /task/status, /vision/detect_request
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class TaskManagerNode(Node):
    """国赛任务调度器"""

    # 状态机
    WAIT   = 'WAITING'
    DETECT = 'DETECTING'    # 等待视觉检测结果
    CENTER = 'CENTERING'    # 底座微调中 (边缘修正/重试)
    GRASP  = 'GRASPING'     # 等待抓取反馈
    VERIFY = 'VERIFYING'    # 等待二次视觉确认是否夹到
    WAIT_PLACE = 'WAITING_PLACE_ZONE'  # 已抓取, 等待识别到目标放置区字母
    PLACE  = 'PLACING'
    DIRECT = 'DIRECT_GRASP'
    DONE   = 'COMPLETED'
    ERR    = 'ERROR'

    EDGE_MARGIN  = 80        # 像素, cx距离左右边少于此值触发微调
    EDGE_ADJUST  = 60        # 像素, 首次边缘微调时的cx修正量
    RETRY_DEGREE_MIN = 40     # 像素, 最小底座旋转量 (~4°)
    RETRY_DEGREE_MAX = 150    # 像素, 最大底座旋转量 (~15°)
    GRASP_Z_TOL = 0.03      # 米, 抓取后z变化超过此值视为物体被拎起
    MAX_GRASP_RETRIES = 10   # 最多重试抓取次数 (10×5°=50°后回home放弃)

    def __init__(self):
        super().__init__('task_manager_node')

        cfg = {}
        try:
            import yaml
            config_path = self.declare_parameter('config_path', '').value
            if config_path and __import__('os').path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
        except:
            pass

        scoring = cfg.get('scoring', {})
        self.hold_time = scoring.get('hold_time', 3.0)
        self.max_drops = scoring.get('max_drops', 3)
        self.total_grasps = scoring.get('total_grasps', 2)
        self.zones_cfg = cfg.get('placement_zones', {})

        dg = cfg.get('direct_grasp', {})
        self.look_pose = {
            'x': dg.get('look_x', 0.20),
            'y': dg.get('look_y', 0.0),
            'z': dg.get('look_z', 0.15),
            'angle': dg.get('look_angle', 0.0),
            'dur': dg.get('look_duration', 2.0),
        }

        # ── 任务状态 ──────────────────────────
        self.state = self.WAIT
        self.targets = []
        self.zone_idx = 0
        self.grasp_cnt = 0
        self.drop_cnt = 0
        self.current_pose = None
        self.current_cmd = ''
        self.direct_mode = False
        self.direct_retries = 0
        self.max_direct_retries = 3
        self._direct_color = 'red'
        self._grasp_retries = 0        # 当前物体抓取重试次数
        self._pre_x = 0.0              # 抓取前物体坐标 (用于z轴判断)
        self._pre_y = 0.0
        self._pre_z = 0.0
        self._pre_cx = 320             # 抓取前物体像素 cx
        self._verify_cx = 320          # VERIFY时检测到的cx, 用于判断旋转方向
        self._rolling_back = False     # 旋转方向错误时回退中
        self._desired_base = 512       # 期望的底座位置, 传给arm_control
        self._rollback_count = 0       # 回退次数, 防止死循环
        self._last_seen_cx = None      # 每次视觉成功时更新, 供回退用

        # ── 订阅 ──────────────────────────────
        self.create_subscription(String, '/vision/grasp_pose', self._cb_vision, 10)
        self.create_subscription(String, '/inspection/target_zones', self._cb_inspection, 10)
        self.create_subscription(String, '/placement/recognized_zone', self._cb_place_zone, 10)
        self.create_subscription(String, '/arm/feedback', self._cb_feedback, 10)
        self.create_subscription(String, '/task/start', self._cb_start, 10)
        self.create_subscription(Bool, '/task/reset', self._cb_reset, 10)
        self.create_subscription(String, '/task/direct_grasp', self._cb_direct_grasp, 10)

        # ── 发布 ──────────────────────────────
        self.pub_cmd = self.create_publisher(String, '/arm/command', 10)
        self.pub_status = self.create_publisher(String, '/task/status', 10)
        self.pub_detect = self.create_publisher(String, '/vision/detect_request', 10)

        self.create_timer(1.0, self._timer)

        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('  任务管理节点就绪')
        self.get_logger().info('  等待巡检结果 (/inspection/target_zones)...')
        self.get_logger().info('═══════════════════════════════════════')

    # ══════════════════════════════════════════════════════════
    #  回调
    # ══════════════════════════════════════════════════════════

    def _cb_inspection(self, msg):
        if self.state not in (self.WAIT,):
            return
        zs = msg.data.strip()
        self.targets = [z.strip() for z in zs.split(',') if z.strip()] if zs else []
        self.get_logger().info(f'巡检结果: 异常区域 = {self.targets}')
        if self.targets:
            self._start_task()
        else:
            self.get_logger().info('无异常区域，无需抓取')
            self._set_state(self.DONE)

    def _cb_vision(self, msg):
        """接收视觉检测结果"""
        if self.state not in (self.DETECT, self.VERIFY):
            return

        data = msg.data.strip()

        if data in ('none', 'invalid_depth', 'low_conf', ''):
            self.get_logger().warn(f'视觉检测失败: {data}')
            self._handle_vision_fail()
            return

        try:
            parts = data.split('|')
            if parts[0] != 'grasp' or len(parts) < 5:
                self.get_logger().error(f'视觉格式错误: {data}')
                self._handle_vision_fail()
                return

            cx_val = int(parts[6]) if len(parts) > 6 else 320
            cy_val = int(parts[7]) if len(parts) > 7 else 240
            z_val  = float(parts[3])

            pose = {
                'x': float(parts[1]),
                'y': float(parts[2]),
                'z': z_val,
                'angle': float(parts[4]),
                'conf': float(parts[5]) if len(parts) > 5 else 0.5,
                'cx': cx_val,
                'cy': cy_val,
            }
            self.get_logger().info(
                f'★ 位姿: x={pose["x"]:.3f} y={pose["y"]:.3f} z={pose["z"]:.3f} '
                f'angle={pose["angle"]:.0f}° 像素({pose["cx"]},{pose["cy"]})')

            # ★ 每次成功检测都记录, 供回退用
            self._last_seen_cx = cx_val
            self._rollback_count = 0  # 检测成功, 清零回退计数

            # ── 分支: VERIFY (二次确认) ────────
            if self.state == self.VERIFY:
                self._verify_grasp(pose)
                return

            # ── 分支: DETECT → 重试定向/边缘检查/直接抓 ──
            self.current_pose = pose

            if self.direct_mode:
                # 首次检测: 边缘微调 (重试时跳过, 避免和retry旋转冲突)
                if self._grasp_retries > 0:
                    self.get_logger().info('重试中, 跳过边缘微调, 直接抓取')
                    self._do_grasp()
                    return

                m = self.EDGE_MARGIN
                if cx_val < m:
                    adj_cx = cx_val - self.EDGE_ADJUST
                    self.get_logger().info(
                        f'物体偏左 (cx={cx_val}<{m}), 小幅左转底座 (→cx={adj_cx})')
                    self._send_center_base(adj_cx)
                    return
                elif cx_val > 640 - m:
                    adj_cx = cx_val + self.EDGE_ADJUST
                    self.get_logger().info(
                        f'物体偏右 (cx={cx_val}>{640-m}), 小幅右转底座 (→cx={adj_cx})')
                    self._send_center_base(adj_cx)
                    return
                else:
                    self.get_logger().info(
                        f'物体在安全区内 (cx={cx_val},cy={cy_val}), 直接抓取')

            self._do_grasp()

        except Exception as e:
            self.get_logger().error(f'解析视觉数据失败: {e}')
            self._handle_vision_fail()

    def _cb_feedback(self, msg):
        """接收机械臂反馈"""
        if self.state not in (self.GRASP, self.PLACE, self.CENTER):
            return

        try:
            parts = msg.data.split('|')
            cmd = parts[0].strip() if len(parts) > 0 else ''
            status = parts[1].strip() if len(parts) > 1 else 'fail'
            ok = (status == 'success')
        except:
            return

        self.get_logger().info(f'机械臂反馈: {cmd} → {"✓ 成功" if ok else "✗ 失败"}')
        self._handle_feedback(cmd, ok)

    def _cb_start(self, msg):
        if self.state == self.WAIT and self.targets:
            self._start_task()

    def _cb_place_zone(self, msg):
        """狗识别到放置区字母后, 只有匹配当前异常目标才放下红条。

        期望话题:
          /placement/recognized_zone std_msgs/String

        可接受格式:
          "A"
          "zone_A"
          "A:0.92"
        """
        if self.state != self.WAIT_PLACE:
            return

        seen_zone = self._normalize_zone(msg.data)
        target_zone = self._current_target_zone()
        if seen_zone is None:
            self.get_logger().warn(f'放置区识别格式无效: {msg.data}')
            return

        if seen_zone != target_zone:
            self.get_logger().info(
                f'看到 {seen_zone} 区, 当前红条目标是 {target_zone} 区, 继续等待')
            return

        self.get_logger().info(
            f'✓ 已确认到达 {target_zone} 区放置箱, 执行红条放置')
        self._do_place()

    def _cb_direct_grasp(self, msg):
        """
        ★ 直抓模式: 检测到红色长条 → 边缘微调(如需) → 抓取 → 二次验证 → 失败重试
        用法: ros2 topic pub /task/direct_grasp std_msgs/msg/String "data: 'red'"
        """
        # 防止重复指令在验证/重试期间重置状态
        if self.direct_mode and self.state not in (self.DONE, self.ERR, self.WAIT):
            self.get_logger().warn(
                f'直抓已在进行中 (state={self.state}), 忽略重复指令')
            return

        color = msg.data.strip().lower() if msg.data else 'red'
        if color not in ('red', 'green'):
            color = 'red'

        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info(f'  ★★★ 直抓: {color}色物体 ★★★')
        self.get_logger().info('  流程: 边缘微调 → 抓取 → 二次确认 → 失败重试')
        self.get_logger().info('═══════════════════════════════════════')

        self.direct_mode = True
        self.direct_retries = 0
        self._grasp_retries = 0
        self._pre_x = 0.0
        self._pre_y = 0.0
        self._pre_z = 0.0
        self._pre_cx = 320
        self._verify_cx = 320
        self._rolling_back = False
        self._desired_base = 512
        self._last_seen_cx = None
        self._rollback_count = 0
        self.grasp_cnt = 0
        self.current_pose = None
        self.current_cmd = ''
        self._direct_color = color

        self._set_state(self.DETECT)
        self.pub_detect.publish(String(data=color))

    def _cb_reset(self, msg):
        if msg.data:
            self._reset()

    # ══════════════════════════════════════════════════════════
    #  核心流程
    # ══════════════════════════════════════════════════════════

    def _start_task(self):
        self.state = self.DETECT
        self.zone_idx = 0
        self.grasp_cnt = 0
        self.drop_cnt = 0
        self.get_logger().info(f'  ★★★ 开始抓取任务: {len(self.targets)} 个区域 ★★★')
        self._request_vision()

    def _request_vision(self):
        self._set_state(self.DETECT)
        zone = self.targets[self.zone_idx] if self.zone_idx < len(self.targets) else self.targets[0]
        self.get_logger().info(f'→ 请求视觉检测 ({zone}区)...')
        self.pub_detect.publish(String(data='red'))

    def _send_center_base(self, cx_target):
        """发送底座微调命令, 同时记录期望的底座位置"""
        cmd = f'center_base|||0|0|1.0|{cx_target}|0'
        self.pub_cmd.publish(String(data=cmd))
        self.current_cmd = 'center_base'
        self._desired_base = int(512 + (cx_target - 320) * 0.5)
        self._set_state(self.CENTER)

    def _do_grasp(self):
        """执行抓取"""
        if self.current_pose is None:
            return

        # 记录抓取前物体坐标, 供验证"原位是否消失"和重试定向
        self._pre_x = self.current_pose['x']
        self._pre_y = self.current_pose['y']
        self._pre_z = self.current_pose['z']
        self._pre_cx = self.current_pose.get('cx', 320)

        if self.direct_mode:
            self.get_logger().info(
                f'█████ 直抓: pre_pos=({self._pre_x:.3f},{self._pre_y:.3f},{self._pre_z:.3f}) █████')
        else:
            zone = self.targets[self.zone_idx] if self.zone_idx < len(self.targets) else self.targets[0]
            self.get_logger().info(f'█████ 第{self.grasp_cnt+1}/{self.total_grasps}次抓取→{zone}区 █████')

        p = self.current_pose
        if self.direct_mode:
            base = getattr(self, '_desired_base', 512)
            cmd = (f'direct_grasp|{p["x"]:.4f}|{p["y"]:.4f}|{p["z"]:.4f}|'
                   f'{p["angle"]:.1f}|3.0|{p["cx"]}|{p["cy"]}|{base}')
            self.current_cmd = 'direct_grasp'
        else:
            cmd = f'grasp|{p["x"]:.4f}|{p["y"]:.4f}|{p["z"]:.4f}|{p["angle"]:.1f}|3.0'
            self.current_cmd = 'grasp'
        self.pub_cmd.publish(String(data=cmd))
        self._set_state(self.GRASP)

    # ══════════════════════════════════════════════════════════
    #  抓取验证 (二次视觉确认)
    # ══════════════════════════════════════════════════════════

    def _verify_grasp(self, pose):
        """
        z轴判断法: 只对比抓取前后物体z坐标的变化
        - z明显变化 → 物体被拎起来了 → 成功
        - z几乎不变 → 物体还在桌上 → 空抓 → 当场转底座重试
        (不用xy判断, 因为视觉对同一物体的xy检测有噪声)
        """
        dz = abs(pose['z'] - self._pre_z)

        self.get_logger().info(
            f'验证: pre_z={self._pre_z:.3f} post_z={pose["z"]:.3f} '
            f'Δz={dz:.3f}m (阈值={self.GRASP_Z_TOL:.3f}m)')

        if dz > self.GRASP_Z_TOL:
            self.grasp_cnt += 1
            self.get_logger().info('═══════════════════════════════════════')
            self.get_logger().info('  ★★★★★ 抓取成功! z轴变化,物体已被拎起 ★★★★★')
            self.get_logger().info('═══════════════════════════════════════')
            self._set_state(self.DONE)
        else:
            self._grasp_retries += 1
            self.get_logger().warn(
                f'✗ 空抓! 物体仍在原位 '
                f'(重试 {self._grasp_retries}/{self.MAX_GRASP_RETRIES})')

            if self._grasp_retries > self.MAX_GRASP_RETRIES:
                self.get_logger().error(
                    f'重试 {self.MAX_GRASP_RETRIES} 次均空抓, 回home放弃')
                self._send_center_base(320)  # 回home底座
                self._set_state(self.ERR)
            else:
                # ★ 定向转底座: 根据物体离中心距离动态计算旋转量
                cx_now = pose['cx']
                self._verify_cx = cx_now
                pixel_off = abs(cx_now - 320)
                degree_offset = max(self.RETRY_DEGREE_MIN,
                                    min(self.RETRY_DEGREE_MAX,
                                        int(40 + pixel_off * 0.35)))
                approx_deg = degree_offset * 0.1
                if cx_now > 320:
                    adj_cx = 320 - degree_offset
                    direction = '右'
                else:
                    adj_cx = 320 + degree_offset
                    direction = '左'
                self.get_logger().info(
                    f'→ 重试: 向{direction}转底座 ~{approx_deg:.1f}° '
                    f'(|cx-320|={pixel_off}px, offset={degree_offset}px, '
                    f'adj_cx={adj_cx})')
                self._send_center_base(adj_cx)

    # ══════════════════════════════════════════════════════════
    #  反馈处理
    # ══════════════════════════════════════════════════════════

    def _handle_feedback(self, cmd, ok):
        # ── 底座微调完成 ──────────────────────
        if cmd == 'center_base':
            if ok:
                self.get_logger().info('底座微调完成, 重新检测...')
                self._set_state(self.DETECT)
                self.pub_detect.publish(String(data=self._direct_color))
            else:
                self.get_logger().error('底座微调失败')
                # 微调失败不阻塞, 直接用原坐标抓
                if self.current_pose is not None:
                    self._do_grasp()
                else:
                    self._set_state(self.ERR)
            return

        # ── 直抓完成 → 进入验证 ────────────────
        if cmd == 'direct_grasp':
            if ok:
                self.get_logger().info('臂已归位, 二次视觉确认...')
                self._set_state(self.VERIFY)
                self.pub_detect.publish(String(data=self._direct_color))
            else:
                self.direct_retries += 1
                self.get_logger().warn(
                    f'直抓执行失败 ({self.direct_retries}/{self.max_direct_retries})')
                if self.direct_retries < self.max_direct_retries:
                    self._set_state(self.DETECT)
                    self.pub_detect.publish(String(data=self._direct_color))
                else:
                    self.get_logger().error('直抓多次执行失败, 放弃')
                    self._set_state(self.ERR)
            return

        if cmd == 'grasp':
            if ok:
                self.grasp_cnt += 1
                self.get_logger().info(f'✓ 第{self.grasp_cnt}次抓取成功!')
                if self.direct_mode:
                    self.get_logger().info('直抓完成')
                    self._set_state(self.DONE)
                else:
                    self._wait_for_place_zone()
            else:
                self.drop_cnt += 1
                self.get_logger().warn(f'✗ 抓取掉落! ({self.drop_cnt}/{self.max_drops})')
                if self.direct_mode:
                    self.get_logger().error('直抓失败')
                    self._set_state(self.ERR)
                else:
                    self._check_drop()

        elif cmd == 'place':
            if ok:
                self.get_logger().info(f'✓ 放置成功! ({self.grasp_cnt}/{self.total_grasps})')
                self._advance()
            else:
                self.drop_cnt += 1
                self.get_logger().warn(f'✗ 放置掉落! ({self.drop_cnt}/{self.max_drops})')
                self._check_drop()

    # ══════════════════════════════════════════════════════════
    #  辅助
    # ══════════════════════════════════════════════════════════

    def _normalize_zone(self, text):
        value = (text or '').strip().upper()
        if ':' in value:
            value = value.split(':', 1)[0].strip()
        if value.startswith('ZONE_'):
            value = value[-1]
        if value in ('A', 'B', 'C', 'D'):
            return value
        return None

    def _current_target_zone(self):
        if not self.targets:
            return None
        index = self.zone_idx if self.zone_idx < len(self.targets) else 0
        return self._normalize_zone(self.targets[index])

    def _wait_for_place_zone(self):
        target_zone = self._current_target_zone()
        if target_zone is None:
            self.get_logger().error('没有当前异常目标区, 无法决定红条放置位置')
            self._set_state(self.ERR)
            return

        self.get_logger().info(
            f'红条已抓取, 等待狗识别到 {target_zone} 区放置箱 '
            '(/placement/recognized_zone)...')
        self._set_state(self.WAIT_PLACE)

    def _do_place(self):
        zone = self._current_target_zone()
        zone_info = self.zones_cfg.get(zone, {}).get('position', [0.2, 0.0, 0.0])
        px, py, pz = zone_info[0], zone_info[1], zone_info[2]
        self.get_logger().info(f'→ 在 {zone}区放下红条: ({px:.2f}, {py:.2f}, {pz:.2f})')
        cmd = f'place|{px:.4f}|{py:.4f}|{pz:.4f}|0|3.0'
        self.pub_cmd.publish(String(data=cmd))
        self.current_cmd = 'place'
        self._set_state(self.PLACE)

    def _advance(self):
        if self.grasp_cnt >= self.total_grasps:
            self._set_state(self.DONE)
            self._print_score()
        else:
            self.zone_idx += 1
            if self.zone_idx >= len(self.targets):
                self.zone_idx = 0
            self._request_vision()

    def _check_drop(self):
        if self.drop_cnt >= self.max_drops:
            self._set_state(self.ERR)
            self.get_logger().error(f'掉落 {self.max_drops} 次，比赛结束!')
        else:
            self.get_logger().info('重新尝试...')
            self._request_vision()

    def _handle_vision_fail(self):
        if self.state == self.VERIFY:
            # 验证时检测失败 → 可能是物体被夹爪完全遮挡 → 判定成功
            self.grasp_cnt += 1
            self.get_logger().info('═══════════════════════════════════════')
            self.get_logger().info('  ★★★★★ 抓取成功! (物体被遮挡/消失) ★★★★★')
            self.get_logger().info('═══════════════════════════════════════')
            self._set_state(self.DONE)
            return

        if self.direct_mode:
            # 视觉丢失 → 立即回退到上次看到物体的位置
            rollback_cx = self._last_seen_cx if self._last_seen_cx is not None else self._pre_cx
            if self._rollback_count < 3:
                self._rollback_count += 1
                self.get_logger().warn(
                    f'视觉丢失! 立即回退底座 (→cx={rollback_cx}, '
                    f'来源:{"视觉记忆" if self._last_seen_cx else "抓取前位置"}, '
                    f'第{self._rollback_count}次)')
                self._send_center_base(rollback_cx)
                return
            else:
                self.get_logger().error('回退3次仍找不到物体, 放弃')
                self._set_state(self.ERR)
                return
        if self.grasp_cnt > 0:
            self._advance()
        else:
            self.get_logger().warn('视觉检测失败，等待下次请求...')
            self._set_state(self.WAIT)

    def _reset(self):
        self.state = self.WAIT
        self.targets = []
        self.zone_idx = 0
        self.grasp_cnt = 0
        self.drop_cnt = 0
        self.current_pose = None
        self.current_cmd = ''
        self.direct_mode = False
        self.direct_retries = 0
        self._grasp_retries = 0
        self._pre_x = 0.0
        self._pre_y = 0.0
        self._pre_z = 0.0
        self._pre_cx = 320
        self._verify_cx = 320
        self._rolling_back = False
        self._desired_base = 512
        self._last_seen_cx = None
        self._rollback_count = 0
        self._direct_color = 'red'
        self.get_logger().info('任务已重置')

    def _set_state(self, s):
        self.state = s

    def _timer(self):
        score = self.grasp_cnt * 25 - min(self.drop_cnt * 5, 10)
        info = (f'状态:{self.state} | '
                f'抓取:{self.grasp_cnt}/{self.total_grasps} | '
                f'掉落:{self.drop_cnt}/{self.max_drops} | '
                f'目标:{",".join(self.targets) if self.targets else "无"} | '
                f'估分:{score}/50')
        self.pub_status.publish(String(data=info))

    def _print_score(self):
        score = self.grasp_cnt * 25 - min(self.drop_cnt * 5, 10)
        self.get_logger().info('═══════════════════════════════════════')
        self.get_logger().info('  ★★★★★ 全部任务完成! ★★★★★')
        self.get_logger().info(f'  抓取成功: {self.grasp_cnt} 次')
        self.get_logger().info(f'  掉落: {self.drop_cnt} 次')
        self.get_logger().info(f'  预估得分: {score}/50')
        self.get_logger().info('═══════════════════════════════════════')


def main():
    rclpy.init()
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
