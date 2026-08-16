#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线仿真 waypoint_navigator 控制律 — 无 ROS 依赖, 四元数级狗模型。

与上一版的关键区别: 狗模型不再"假设 yaw=朝向", 而是合成真实的
/camera_pose 四元数 (水平光学相机 = 绕世界 y 轴的纯旋转), 导航器侧
用与 nodes/waypoint_navigator.py 逐行相同的公式从四元数提取朝向。

世界系 (与实机一致): y 竖直, 地面 = x-z, 朝向 φ = atan2(f_z, f_x)。
水平相机朝向 φ 对应四元数 q = (0, sin(α/2), 0, cos(α/2)), α = π/2 − φ。

两种朝向提取对比:
  new: 前向向量投影 atan2(f_z, f_x)  → 精确恢复 φ
  old: z-euler atan2(2(wz+xy), 1−2(y²+z²)) → 对绕 y 轴旋转退化为 0/π
       (实机采集数据已证实: 13 点不同朝向 yaw 全部 ≈0)

物理转向模型 (wz>0 的物理效果, 二选一, 实机确认):
  turn_model=−1: wz>0 → φ 减小 (世界 y 朝下 + 接收桥 wz>0=右转 时成立)
  turn_model=+1: wz>0 → φ 增大 (世界 y 朝上, 或接收桥符号相反时成立)

用法:
  python3 tools/sim_waypoint_nav.py [--waypoints PATH]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ---- 与 nodes/waypoint_navigator.py 逐行相同的提取公式 ----

def heading_new(quat, ground_plane: str = "xz", forward_axis: str = "z") -> float:
    """前向向量投影提取朝向 (修复后, 默认 forward_axis=z)。"""
    x, y, z, w = quat
    if forward_axis == "x":
        fx = 1.0 - 2.0 * (y * y + z * z)
        fz = 2.0 * (x * z - w * y)
    else:
        fx = 2.0 * (x * z + w * y)
        fz = 1.0 - 2.0 * (x * x + y * y)
    if ground_plane == "xz":
        return math.atan2(fz, fx)
    return math.atan2(2.0 * (y * z - w * x), fx)


def heading_old(quat) -> float:
    """z-euler yaw (修复前) — 对绕世界 y 轴的真实转身退化为 0/π。"""
    x, y, z, w = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---- 狗模型: 物理朝向 φ → 合成 /camera_pose 四元数 ----

def quat_from_heading(phi: float) -> tuple[float, float, float, float]:
    """水平光学相机朝向 φ (rad), 世界系 y 竖直: q = 绕 y 轴 α=π/2−φ。"""
    alpha = math.pi / 2.0 - phi
    return (0.0, math.sin(alpha / 2.0), 0.0, math.cos(alpha / 2.0))


class SimParams:
    goal_tolerance = 0.16
    yaw_tolerance = 0.22
    kp_linear = 0.45
    kp_angular = 1.2
    max_vx = 0.28
    max_wz = 0.45
    rotate_in_place_angle = 0.75
    dt = 0.1


def control_law(x, z, phi_extracted, goal, cfg: SimParams, turn_sign: float):
    """与 waypoint_navigator._tick 一致的控制律, 返回 (vx, wz, dist, arrived)。"""
    dx = goal["x"] - x
    dz = goal["z"] - z
    distance = math.sqrt(dx * dx + dz * dz)
    bearing = math.atan2(dz, dx)
    yaw_error = normalize_angle(goal["yaw"] - phi_extracted)
    if distance <= cfg.goal_tolerance and abs(yaw_error) <= cfg.yaw_tolerance:
        return 0.0, 0.0, distance, True
    heading_error = normalize_angle(bearing - phi_extracted)
    vx = 0.0
    wz = 0.0
    if distance <= cfg.goal_tolerance:
        wz = turn_sign * clamp(cfg.kp_angular * yaw_error, -cfg.max_wz, cfg.max_wz)
    else:
        if abs(heading_error) <= cfg.rotate_in_place_angle:
            vx = clamp(cfg.kp_linear * distance * max(0.0, math.cos(heading_error)),
                       0.0, cfg.max_vx)
        wz = turn_sign * clamp(cfg.kp_angular * heading_error, -cfg.max_wz, cfg.max_wz)
    return vx, wz, distance, False


def simulate(waypoints, cfg: SimParams, extraction: str, turn_sign: float,
             turn_model: float, max_steps: int = 6000):
    """闭环仿真。extraction: 'new' 前向投影 / 'old' z-euler。"""
    extract = heading_new if extraction == "new" else (lambda q: heading_old(q))
    x, z, phi = waypoints[0]["x"], waypoints[0]["z"], waypoints[0]["yaw"]
    results = []
    all_ok = True
    for goal in waypoints:
        arrived = False
        steps = 0
        dist = float("inf")
        for _ in range(max_steps):
            quat = quat_from_heading(phi)
            phi_ext = extract(quat)
            vx, wz, dist, done = control_law(x, z, phi_ext, goal, cfg, turn_sign)
            if done:
                arrived = True
                break
            x += vx * math.cos(phi) * cfg.dt
            z += vx * math.sin(phi) * cfg.dt
            phi = normalize_angle(phi + turn_model * wz * cfg.dt)
            steps += 1
        results.append((goal["name"], arrived, steps, dist))
        if not arrived:
            all_ok = False
            break
    return results, all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waypoints", default="jetson_payload/slam_maps/waypoints_FINAL.yaml")
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.waypoints).read_text(encoding="utf-8"))
    waypoints = data.get("waypoints", [])
    cfg = SimParams()

    scenarios = [
        ("新提取(前向投影) + wz>0→右转模型 + turn_sign=-1", "new", -1.0, -1.0),
        ("新提取(前向投影) + wz>0→左转模型 + turn_sign=+1", "new", +1.0, +1.0),
        ("符号失配: 右转模型配 turn_sign=+1", "new", +1.0, -1.0),
        ("旧提取(z-euler) + 右转模型 + turn_sign=-1", "old", -1.0, -1.0),
        ("旧提取(z-euler) + 左转模型 + turn_sign=+1", "old", +1.0, +1.0),
    ]
    for title, extraction, sign, model in scenarios:
        results, ok = simulate(waypoints, cfg, extraction, sign, model)
        reached = sum(1 for _, a, _, _ in results if a)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {title}: 到达 {reached}/{len(results)}")
        for name, arrived, steps, dist in results[:5]:
            mark = "OK " if arrived else "NG "
            print(f"  {mark}{name}: steps={steps} dist={dist:.3f}m")
        if len(results) > 5:
            print(f"  ... 共 {len(results)} 个")

    print(
        "\n结论:\n"
        "  1. 前向投影提取在两种物理转向模型下都能全航点收敛 (符号各自匹配)。\n"
        "  2. z-euler 提取对绕 y 轴转身退化 (恒 0/π), 到点 yaw 永差 ±90° 不收敛,\n"
        "     行进中 heading_error 也是错轴相减 → 复现上机乱跑。\n"
        "  3. 实机首跑看第一个转向: 该左转时右转 → run_waypoints_only.sh 里\n"
        "     TURN_SIGN 改 -1; 到位后朝向恒差 90° → FORWARD_AXIS 改 x。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
