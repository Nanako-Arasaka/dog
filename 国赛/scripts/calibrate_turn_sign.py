#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动标定 turn_sign / forward_axis —— 消除上机前的两个符号约定猜测。

原理(闭环符号直接测量, 不依赖任何文档约定):
  A. 转向符号: 命令 +wz 原地转 N 秒, 看 heading(atan2(f_z,f_x)) 变化方向。
       heading 增大 → +wz 使 heading 增大 → 收敛需 turn_sign=+1.0
       heading 减小 → turn_sign=-1.0
     (turn_sign 反了 = 闭环发散 = 现场看到的"原地打转不前进")
  B. 前向轴: 命令 +vx 直走 N 秒, 位移方位 ≈ heading → 当前 axis 正确;
       偏 ±90° → 四元数是另一种约定, 换另一根轴。

前提: watchdog + motion_mux 已运行, 狗已站立/移动模式/行走步态,
     周围留 0.5m 空间(狗会原地转 ~30°、向前走 ~0.4m)。

输出末行机器可读: CALIB turn_sign=+1.0 forward_axis=z   (失败字段为 ?)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from std_msgs.msg import Bool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nodes"))
from waypoint_navigator import heading_from_pose  # noqa: E402


def wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def circular_mean(values: list[float]) -> float:
    sx = sum(math.cos(v) for v in values)
    sy = sum(math.sin(v) for v in values)
    return math.atan2(sy, sx)


class TurnCalibrator(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("turn_sign_calibrator")
        self.args = args
        self.loc_ok = False
        # (t, x, z, heading)
        self.samples: list[tuple[float, float, float, float]] = []
        self.cmd_pub = self.create_publisher(Twist, "/motion/nav_cmd", 10)
        self.create_subscription(PoseStamped, "/camera_pose_fused", self._on_pose, 10)
        self.create_subscription(Bool, "/localization/ok", self._on_ok, 10)

    def _on_ok(self, msg: Bool) -> None:
        self.loc_ok = bool(msg.data)

    def _on_pose(self, msg: PoseStamped) -> None:
        heading = heading_from_pose(msg.pose, "xz", self.args.forward_axis)
        p = msg.pose.position
        self.samples.append((time.monotonic(), float(p.x), float(p.z), heading))

    def spin_for(self, sec: float) -> None:
        end = time.monotonic() + sec
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def window(self, sec: float) -> list[tuple[float, float, float, float]]:
        now = time.monotonic()
        return [s for s in self.samples if now - s[0] <= sec]

    def wait_localization(self) -> bool:
        print(f"[calib] 等待定位 OK(最多 {self.args.wait_loc:.0f}s)...", flush=True)
        deadline = time.monotonic() + self.args.wait_loc
        while time.monotonic() < deadline and rclpy.ok():
            self.spin_for(0.2)
            if self.loc_ok and len(self.window(1.0)) >= 5:
                return True
        return False

    def drive(self, vx: float, wz: float, sec: float) -> None:
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        end = time.monotonic() + sec
        while time.monotonic() < end and rclpy.ok():
            self.cmd_pub.publish(cmd)
            self.spin_for(0.05)
        self.cmd_pub.publish(Twist())

    def calibrate_turn(self) -> str:
        self.spin_for(1.0)
        base = self.window(1.0)
        if len(base) < 3:
            print("[calib] A FAILED: 基准位姿不足", flush=True)
            return "?"
        h0 = circular_mean([s[3] for s in base])
        print(f"[calib] A: 基准 heading={math.degrees(h0):+.1f}°, "
              f"发 +wz={self.args.test_wz} 原地转 {self.args.turn_sec:.1f}s ...", flush=True)
        self.drive(0.0, self.args.test_wz, self.args.turn_sec)
        self.spin_for(self.args.settle)
        recent = self.window(1.0)
        if len(recent) < 3:
            print("[calib] A FAILED: 转向后定位丢失", flush=True)
            return "?"
        h1 = circular_mean([s[3] for s in recent])
        delta = wrap(h1 - h0)
        print(f"[calib] A: heading {math.degrees(h0):+.1f}° -> {math.degrees(h1):+.1f}° "
              f"(Δ={math.degrees(delta):+.1f}°)", flush=True)
        if abs(delta) < math.radians(self.args.min_delta_deg):
            print("[calib] A FAILED: 转角过小(狗没转?), 检查站立/移动模式", flush=True)
            return "?"
        sign = "+1.0" if delta > 0 else "-1.0"
        print(f"[calib] A: turn_sign={sign}", flush=True)
        return sign

    def calibrate_axis(self) -> str:
        base = self.window(1.0)
        if len(base) < 3:
            print("[calib] B FAILED: 基准位姿不足", flush=True)
            return "?"
        h = circular_mean([s[3] for s in base])
        x0 = sum(s[1] for s in base) / len(base)
        z0 = sum(s[2] for s in base) / len(base)
        print(f"[calib] B: 发 +vx={self.args.test_vx} 直走 {self.args.walk_sec:.1f}s ...", flush=True)
        self.drive(self.args.test_vx, 0.0, self.args.walk_sec)
        self.spin_for(self.args.settle)
        recent = self.window(1.0)
        if len(recent) < 3:
            print("[calib] B FAILED: 走动后定位丢失", flush=True)
            return "?"
        x1 = sum(s[1] for s in recent) / len(recent)
        z1 = sum(s[2] for s in recent) / len(recent)
        dist = math.hypot(x1 - x0, z1 - z0)
        bearing = math.atan2(z1 - z0, x1 - x0)
        err = wrap(bearing - h)
        print(f"[calib] B: 位移 {dist:.2f}m 方位 {math.degrees(bearing):+.1f}°, "
              f"heading {math.degrees(h):+.1f}°, 偏差 {math.degrees(err):+.1f}°", flush=True)
        if dist < self.args.min_walk_m:
            print("[calib] B FAILED: 位移过小(狗没走?)", flush=True)
            return "?"
        if abs(err) <= math.pi / 4.0:
            print(f"[calib] B: forward_axis={self.args.forward_axis} 正确", flush=True)
            return self.args.forward_axis
        other = "x" if self.args.forward_axis == "z" else "z"
        if (abs(wrap(err - math.pi / 2.0)) <= math.pi / 4.0
                or abs(wrap(err + math.pi / 2.0)) <= math.pi / 4.0):
            print(f"[calib] B: 偏差≈±90°, forward_axis 应为 {other}", flush=True)
            return other
        print("[calib] B FAILED: 偏差既非 0 也非 ±90°, 结果不可信", flush=True)
        return "?"


def main() -> None:
    parser = argparse.ArgumentParser(description="自动标定 turn_sign / forward_axis")
    parser.add_argument("--forward-axis", default="z", choices=["z", "x"],
                        help="待验证的当前前向轴约定(默认 z=光学相机)")
    parser.add_argument("--test-wz", type=float, default=0.25, help="转向测试角速度 rad/s")
    parser.add_argument("--turn-sec", type=float, default=2.0, help="转向测试时长 s")
    parser.add_argument("--test-vx", type=float, default=0.18, help="直走测试线速度 m/s")
    parser.add_argument("--walk-sec", type=float, default=2.0, help="直走测试时长 s")
    parser.add_argument("--settle", type=float, default=1.5, help="每次动作后等定位稳定 s")
    parser.add_argument("--wait-loc", type=float, default=30.0, help="等待定位 OK 最长 s")
    parser.add_argument("--min-delta-deg", type=float, default=12.0, help="转向有效最小角度 deg")
    parser.add_argument("--min-walk-m", type=float, default=0.12, help="直走有效最小位移 m")
    args = parser.parse_args()

    rclpy.init()
    node = TurnCalibrator(args)
    turn_sign = "?"
    axis = "?"
    try:
        if not node.wait_localization():
            print("[calib] FAILED: 定位未就绪, 无法标定", flush=True)
        else:
            print("[calib] ⚠️ 狗即将原地慢转 + 向前走, 确保周围 0.5m 无障碍!", flush=True)
            node.spin_for(2.0)
            turn_sign = node.calibrate_turn()
            axis = node.calibrate_axis()
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"CALIB turn_sign={turn_sign} forward_axis={axis}", flush=True)


if __name__ == "__main__":
    main()
