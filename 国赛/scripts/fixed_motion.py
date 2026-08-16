#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定运动脚本 —— 不依赖航点/导航, 纯定位闭环执行固定动作序列。

动作序列(默认, 可用 --sequence 改):
  ① 前进 4.0 m
  ② 原地右转 90°
  ③ 前进 2.5 m
  ④ 原地右转 90°
  ⑤ 前进 0.5 m
  ⑥ 原地右转 90°

控制链路:
  /camera_pose_fused (PoseStamped) + /localization/ok (Bool)
      → 本脚本闭环(前进按距离、转向按 heading)
      → /motion/nav_cmd (Twist) → motion_mux → UDP 5005
      → lite2_motion_receiver → 狗

前提(底层桥已运行, 不需要 navigator / walker):
  watchdog + motion_mux + lite2_motion_receiver, 且 SLAM 定位 OK。

用法:
  python3 scripts/fixed_motion.py                          # 默认序列
  python3 scripts/fixed_motion.py --sequence "4,90r,2.5,90r,0.5,90r"
                                  --turn-sign -1.0 --forward-axis z

序列语法: 数字 = 前进米; 数字+r = 右转角度(如 90r = 右转90°); 逗号分隔。

符号约定(与 waypoint_navigator / calibrate_turn_sign 同一套):
  - heading = atan2(f_z, f_x) 前向向量投影(xz 地面系, forward_axis=z)
  - turn_sign: 实机证据(2026-08-16) +wz 使 heading 减小 → 默认 -1.0;
    若现场发现转向方向相反, 改 --turn-sign 1.0。
  - turn_dir: 右转在 heading 空间的增量方向, 默认 -1(heading 减小);
    若现场发现"右转"实际往左转, 改 --turn-dir 1。
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from rclpy.node import Node
from std_msgs.msg import Bool


def wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def heading_from_pose(pose: Pose, ground_plane: str = "xz", forward_axis: str = "z") -> float:
    """从四元数用前向向量投影提取地面朝向(与 waypoint_navigator 同实现)。"""
    q = pose.orientation
    if forward_axis == "x":
        fx = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        fy = 2.0 * (q.x * q.y + q.w * q.z)
        fz = 2.0 * (q.x * q.z - q.w * q.y)
    else:
        fx = 2.0 * (q.x * q.z + q.w * q.y)
        fy = 2.0 * (q.y * q.z - q.w * q.x)
        fz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    if ground_plane == "xz":
        return math.atan2(fz, fx)
    return math.atan2(fy, fx)


def parse_sequence(text: str) -> list[tuple[str, float]]:
    """'4,90r,2.5,90r,0.5,90r' → [('forward',4.0),('turn',90.0),...]"""
    seq: list[tuple[str, float]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token[-1] in ("r", "R"):
            seq.append(("turn", float(token[:-1])))
        else:
            seq.append(("forward", float(token)))
    return seq


class FixedMotion(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("fixed_motion")
        self.args = args
        self.seq = parse_sequence(args.sequence)
        if not self.seq:
            raise ValueError("sequence 为空")

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, 10)
        self.create_subscription(Bool, args.loc_topic, self._on_loc, 10)
        self.create_timer(0.1, self._tick)

        self.pose: Pose | None = None
        self.loc_ok = False
        self.idx = -1                 # 当前动作下标, -1 = 等待开始
        self.phase = "wait"           # wait | forward | turn | done
        self.failed = False
        self.start_x = 0.0
        self.start_z = 0.0
        self.start_heading = 0.0
        self.target_heading = 0.0
        self.goal_m = 0.0
        self.turn_deg = 0.0
        self.phase_started = 0.0
        self.last_log = 0.0
        self.get_logger().info(f"动作序列: {self.seq}")

    # ---------- 回调 ----------
    def _on_pose(self, msg: PoseStamped) -> None:
        self.pose = msg.pose

    def _on_loc(self, msg: Bool) -> None:
        self.loc_ok = bool(msg.data)

    # ---------- 辅助 ----------
    def _heading(self) -> float:
        return heading_from_pose(self.pose, self.args.ground_plane, self.args.forward_axis)

    def _publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def _log_progress(self, text: str) -> None:
        now = time.monotonic()
        if now - self.last_log >= 1.0:
            self.get_logger().info(text)
            self.last_log = now

    def _begin_step(self, idx: int) -> None:
        self.idx = idx
        self.phase_started = time.monotonic()
        p = self.pose.position
        self.start_x, self.start_z = float(p.x), float(p.z)
        self.start_heading = self._heading()
        kind, value = self.seq[idx]
        if kind == "forward":
            self.phase = "forward"
            self.goal_m = value
            self.get_logger().info(
                f"🚶 [{idx + 1}/{len(self.seq)}] 前进 {value:.2f} m")
        else:
            self.phase = "turn"
            self.turn_deg = value
            self.target_heading = self.start_heading + self.args.turn_dir * math.radians(value)
            self.get_logger().info(
                f"🔁 [{idx + 1}/{len(self.seq)}] 右转 {value:.0f}° "
                f"(heading {math.degrees(self.start_heading):+.1f}° → "
                f"{math.degrees(self.target_heading):+.1f}°)")

    def _next_step(self) -> None:
        # 动作间保持 nav_cmd 新鲜(motion_mux max_cmd_age_sec=0.6, 超时判 stale
        # 会发 stop): 等待期间持续发布零速, 避免边界竞争。
        settle_end = time.monotonic() + 0.4
        while time.monotonic() < settle_end and rclpy.ok():
            self._publish_stop()
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.idx + 1 >= len(self.seq):
            self.phase = "done"
            self.get_logger().info("🎉 全部动作执行完成! DONE")
            self._publish_stop()
            return
        self._begin_step(self.idx + 1)

    def _fail(self, reason: str) -> None:
        self.get_logger().error(f"❌ {reason} — 停止并退出")
        self.failed = True
        self.phase = "done"
        self._publish_stop()

    # ---------- 控制 ----------
    def _tick(self) -> None:
        if self.phase == "done":
            self._publish_stop()
            return
        if not self.loc_ok or self.pose is None:
            self._publish_stop()
            return
        if self.idx < 0:
            self._begin_step(0)
            return

        # 每步超时保护
        if self.args.step_timeout > 0 and time.monotonic() - self.phase_started > self.args.step_timeout:
            self.get_logger().warn(f"⏱️ 第 {self.idx + 1} 步超时({self.args.step_timeout:.0f}s)")
            if self.args.skip_on_timeout:
                self.get_logger().warn("  → 跳过继续")
                self._next_step()
                return
            self._fail(f"第 {self.idx + 1} 步超时")
            return

        if self.phase == "forward":
            self._tick_forward()
        else:
            self._tick_turn()

    def _tick_forward(self) -> None:
        p = self.pose.position
        dx = float(p.x) - self.start_x
        dz = float(p.z) - self.start_z
        dist = math.hypot(dx, dz)
        remaining = self.goal_m - dist
        yaw_err = wrap(self.start_heading - self._heading())

        if remaining <= self.args.goal_tol:
            self.get_logger().info(f"✅ 前进 {self.goal_m:.2f} m 完成(实测 {dist:.2f} m)")
            self._publish_stop()
            self._next_step()
            return

        twist = Twist()
        if abs(yaw_err) > self.args.heading_limit:
            # 朝向偏太多: 先原地纠偏再走
            twist.angular.z = self.args.turn_sign * clamp(
                self.args.kp_angular * yaw_err, -self.args.max_wz, self.args.max_wz)
        else:
            twist.linear.x = clamp(
                self.args.kp_linear * remaining, 0.0, self.args.max_vx)
            twist.angular.z = self.args.turn_sign * clamp(
                self.args.kp_angular * yaw_err, -self.args.max_wz, self.args.max_wz)
        self.cmd_pub.publish(twist)
        self._log_progress(
            f"前进 {self.goal_m:.2f} m: 已走 {dist:.2f} m, 余 {remaining:.2f} m")

    def _tick_turn(self) -> None:
        current = self._heading()
        yaw_err = wrap(self.target_heading - current)
        if abs(yaw_err) <= self.args.yaw_tol:
            self.get_logger().info(
                f"✅ 右转 {self.turn_deg:.0f}° 完成(heading {math.degrees(current):+.1f}°)")
            self._publish_stop()
            self._next_step()
            return

        twist = Twist()
        twist.angular.z = self.args.turn_sign * clamp(
            self.args.kp_angular * yaw_err, -self.args.max_wz, self.args.max_wz)
        self.cmd_pub.publish(twist)
        self._log_progress(
            f"右转 {self.turn_deg:.0f}°: 当前 {math.degrees(current):+.1f}°, "
            f"目标 {math.degrees(self.target_heading):+.1f}°")


def main() -> None:
    parser = argparse.ArgumentParser(description="固定运动: 前进/右转序列(不依赖航点)")
    parser.add_argument("--sequence", default="4,90r,2.5,90r,0.5,90r",
                        help="动作序列: 数字=前进米, 数字r=右转角度, 逗号分隔")
    parser.add_argument("--pose-topic", default="/camera_pose_fused")
    parser.add_argument("--cmd-topic", default="/motion/nav_cmd")
    parser.add_argument("--loc-topic", default="/localization/ok")
    parser.add_argument("--ground-plane", default="xz", choices=["xz", "xy"])
    parser.add_argument("--forward-axis", default="z", choices=["z", "x"])
    parser.add_argument("--turn-sign", type=float, default=-1.0,
                        help="实机证据=-1.0; 若转向方向相反改 1.0")
    parser.add_argument("--turn-dir", type=float, default=-1.0,
                        help="右转方向: -1=heading 减小(默认), 1=增大")
    # 参数取值依据(receiver 链路硬约束, 见 lite2_motion_receiver.py):
    #   - normalized_deadband=0.05: |cmd| < 0.05 被置 0 → 到位容差必须 > deadband/kp,
    #     否则误差收敛到 deadband 内时命令为 0, 狗停住永远到不了容差 → 超时。
    #     故 kp_linear=1.5 → deadband 距离 0.033m, goal_tol=0.10 覆盖(留惯性余量);
    #         kp_angular=2.0 → deadband 角度 0.025rad, yaw_tol=0.06 覆盖。
    #   - ensure_effective_speed: 非零命令最小抬升到 vx≈0.2 / wz≈0.29(归一化),
    #     因此实际速度是"恒速+突停"而非平滑减速, 靠容差+惯性滑行到位。
    parser.add_argument("--kp-linear", type=float, default=1.5)
    parser.add_argument("--kp-angular", type=float, default=2.0)
    parser.add_argument("--max-vx", type=float, default=0.25)
    parser.add_argument("--max-wz", type=float, default=0.5)
    parser.add_argument("--goal-tol", type=float, default=0.10,
                        help="前进到位容差 m(必须 > deadband/kp_linear ≈ 0.033)")
    parser.add_argument("--yaw-tol", type=float, default=0.06,
                        help="转向到位容差 rad≈3.4°(必须 > deadband/kp_angular ≈ 0.025)")
    parser.add_argument("--heading-limit", type=float, default=0.5,
                        help="前进中允许的朝向偏差 rad, 超过先原地纠偏")
    parser.add_argument("--step-timeout", type=float, default=60.0,
                        help="每步超时 s, 0=不限制")
    parser.add_argument("--skip-on-timeout", action="store_true",
                        help="超时跳过当前步继续")
    parser.add_argument("--wait-loc", type=float, default=30.0,
                        help="等待定位 OK 最长 s")
    args = parser.parse_args()

    rclpy.init()
    node = FixedMotion(args)
    try:
        # 等定位
        deadline = time.monotonic() + args.wait_loc
        while rclpy.ok() and time.monotonic() < deadline and not (node.loc_ok and node.pose is not None):
            rclpy.spin_once(node, timeout_sec=0.2)
        if not (node.loc_ok and node.pose is not None):
            node.get_logger().error("定位未就绪, 无法执行 — 检查 watchdog/SLAM")
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(1)
        node.get_logger().info("✅ 定位 OK, 开始执行固定动作序列")

        while rclpy.ok() and not node.failed and node.phase != "done":
            node._tick()
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    finally:
        node._publish_stop()
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("[fixed_motion] 退出")


if __name__ == "__main__":
    main()
