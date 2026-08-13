#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AprilTag 世界坐标现场标定工具。

在 Jetson 上运行（SLAM + RealSense 已启动、定位正常）：

  python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml

流程：
  1. 读取 tags.yaml 模板（id / name / size），占位坐标无所谓
  2. 逐个 tag：把狗开到能看到该 tag 的位置，等 SLAM 位姿稳定后按 Enter
  3. 工具自动采集 N 帧「SLAM 相机位姿 + tag 检测」，解出 tag 在世界系的 6DoF 位姿
  4. 离群剔除 + 多帧平均后写回 tags.yaml（原文件自动备份为 .bak_时间戳）

标定后验证（建议执行，检查兜底定位精度）：

  python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml --verify

原理（与 nodes/tag_localizer_node.py 完全对偶）：
  检测给出 tag 在相机系位姿：p_cam   = R_cam_tag @ p_tag + t_cam_tag
  SLAM 给出相机在世界系位姿：p_world = R_world_cam @ p_cam + t_world_cam
  消去 p_cam 得 tag 世界位姿：
      R_w_tag = R_world_cam @ R_cam_tag
      p_w_tag = t_world_cam + R_world_cam @ t_cam_tag
  节点运行时用逆运算从 tag 恢复相机位姿（兜底）。

无 ROS 自检（验证几何函数正确性）：

  python3 tools/calibrate_tags.py --self-test
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ================================================================ 纯几何函数（无 ROS 依赖，可单测）

def euler_zyx_to_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX 顺序欧拉角（度）→ 3x3 旋转矩阵，R = Rz @ Ry @ Rx。与节点实现一致。"""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_to_euler_zyx_deg(R: np.ndarray) -> tuple[float, float, float]:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) → (roll_deg, pitch_deg, yaw_deg)。"""
    sp = max(-1.0, min(1.0, -float(R[2, 0])))
    pitch = math.asin(sp)
    if abs(sp) > 0.99999:  # 万向节死锁：roll/yaw 只观测量到 (roll ∓ yaw)，约定 roll=0
        roll = 0.0
        if sp > 0:  # pitch = +90°
            yaw = math.atan2(-R[0, 1], R[0, 2])
        else:       # pitch = -90°
            yaw = math.atan2(-R[0, 1], -R[0, 2])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def yaw_of_rotation(R: np.ndarray) -> float:
    """相机近似水平时，用 ZYX yaw 即可（pitch 远离 ±90°）。"""
    return math.atan2(R[1, 0], R[0, 0])


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def average_rotations(Rs: list[np.ndarray]) -> np.ndarray:
    """旋转矩阵平均：算术平均 + SVD 投影回 SO(3)。样本散布小时足够精确。"""
    M = np.mean(np.asarray(Rs), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def angular_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1 @ R2.T) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def compute_tag_world_pose(R_world_cam, t_world_cam, R_cam_tag, t_cam_tag):
    """SLAM 相机位姿 + 一次检测 → tag 在世界系的 6DoF 位姿。"""
    R_w_tag = R_world_cam @ R_cam_tag
    p_w_tag = t_world_cam + R_world_cam @ t_cam_tag
    return R_w_tag, p_w_tag


def camera_pose_from_tag(R_w_tag, p_w_tag, R_cam_tag, t_cam_tag):
    """已标定 tag 位姿 + 一次检测 → 相机在世界系位姿（tag_localizer_node 的运行时方程）。"""
    R_world_cam = R_w_tag @ R_cam_tag.T
    t_world_cam = p_w_tag - R_world_cam @ t_cam_tag
    return R_world_cam, t_world_cam


# ================================================================ 自检

def _small_random_rotation(rng: np.random.Generator, angle_rad: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def self_test() -> int:
    rng = np.random.default_rng(42)

    # 1) 欧拉角往返（含贴墙 pitch=±90 附近的邻域）
    for _ in range(500):
        r = float(rng.uniform(-45, 45))
        p = float(rng.uniform(-85, 85))
        y = float(rng.uniform(-180, 180))
        R = euler_zyx_to_rotation(r, p, y)
        r2, p2, y2 = rotation_to_euler_zyx_deg(R)
        R2 = euler_zyx_to_rotation(r2, p2, y2)
        assert np.allclose(R, R2, atol=1e-9), f"euler roundtrip failed: {(r, p, y)} -> {(r2, p2, y2)}"
    print("[1/4] 欧拉角 ZYX 往返一致 ✓")

    # 2) 标定方程往返：真值 tag 位姿 + 相机位姿 → 合成检测 → 还原 tag 位姿
    R_wt = euler_zyx_to_rotation(0.0, 90.0, 37.0)          # 典型贴墙 tag（pitch=90）
    p_wt = np.array([2.5, -1.3, 0.45]).reshape(3, 1)
    R_wc = euler_zyx_to_rotation(0.0, 0.0, -140.0)         # 狗在 2m 外斜对 tag
    t_wc = np.array([1.1, 0.4, 0.30]).reshape(3, 1)
    R_ct = R_wc.T @ R_wt
    t_ct = R_wc.T @ (p_wt - t_wc)
    R_wt2, p_wt2 = compute_tag_world_pose(R_wc, t_wc, R_ct, t_ct)
    assert np.allclose(R_wt, R_wt2, atol=1e-9), "calibration rotation mismatch"
    assert np.allclose(p_wt, p_wt2, atol=1e-9), "calibration position mismatch"
    print("[2/4] 标定方程往返还原 tag 位姿 ✓")

    # 3) 运行时逆方程（节点 _camera_world_pose）与标定方程互逆
    R_wc2, t_wc2 = camera_pose_from_tag(R_wt, p_wt, R_ct, t_ct)
    assert np.allclose(R_wc, R_wc2, atol=1e-9), "inverse rotation mismatch"
    assert np.allclose(t_wc, t_wc2, atol=1e-9), "inverse position mismatch"
    print("[3/4] 节点运行时逆方程一致 ✓")

    # 4) 旋转平均抗噪
    noisy = [R_wt @ _small_random_rotation(rng, math.radians(1.0)) for _ in range(30)]
    err = angular_error_deg(average_rotations(noisy), R_wt)
    assert err < 0.5, f"rotation averaging error too large: {err:.3f} deg"
    print(f"[4/4] 旋转平均抗噪 ✓（30 帧 1° 噪声 → 残差 {err:.3f}°）")

    print("\nself-test 全部通过")
    return 0


# ================================================================ tags.yaml 模板读写（无 ROS 依赖）

def load_template(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    family = str(data.get("tag_family", "tag36h11"))
    default_size = float(data.get("default_size_m", 0.20))
    metas: dict[int, dict] = {}
    for entry in data.get("tags", []):
        tid = int(entry["id"])
        metas[tid] = {
            "name": str(entry.get("name", f"tag_{tid}")),
            "size_m": float(entry.get("size_m", default_size)),
            "note": str(entry.get("note", "")),
        }
    return data, family, default_size, metas


def write_tags_yaml(path: Path, data: dict, calibrated: dict[int, dict], dry_run: bool) -> Optional[Path]:
    out = copy.deepcopy(data)
    for entry in out.get("tags", []):
        tid = int(entry["id"])
        if tid in calibrated:
            c = calibrated[tid]
            entry["world"] = {
                "x": round(c["x"], 4),
                "y": round(c["y"], 4),
                "z": round(c["z"], 4),
                "yaw_deg": round(c["yaw"], 2),
                "pitch_deg": round(c["pitch"], 2),
                "roll_deg": round(c["roll"], 2),
            }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    header = (
        f"# 由 tools/calibrate_tags.py 于 {datetime.now():%Y-%m-%d %H:%M:%S} 现场标定生成\n"
        f"# 世界系 = ORB-SLAM3 建图时的世界系（与 /camera_pose 一致）\n"
        f"# 手动改动前请先备份；重新标定直接再跑一次本工具即可\n"
    )
    body = yaml.safe_dump(out, allow_unicode=True, sort_keys=False, default_flow_style=None)
    if dry_run:
        print("\n--- dry-run：以下为将写入的内容 ---")
        print(header + body)
        return None
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    path.write_text(header + body, encoding="utf-8")
    return backup


# ================================================================ ROS 标定主流程

def run_ros(args, data: dict, family: str, default_size: float, metas: dict[int, dict], verify: bool) -> int:
    sys.path.insert(0, str(REPO_ROOT / "nodes"))
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image
    import tag_localizer_node as tln  # 复用检测后端与图像转换

    backend, downgrade_note = tln.make_backend(family)
    if downgrade_note:
        print(f"[警告] {downgrade_note}")

    class CalibNode(Node):
        def __init__(self) -> None:
            super().__init__("calibrate_tags")
            self.lock = threading.Lock()
            self.slam_history: deque = deque(maxlen=400)  # (recv_time, R 3x3, t 3x1)
            self.latest_gray: Optional[np.ndarray] = None
            self.gray_time = 0.0
            self.K: Optional[np.ndarray] = None

        def ingest_pose(self, pose) -> None:
            q = pose.orientation
            R = quaternion_to_rotation(q.x, q.y, q.z, q.w)
            t = np.array([pose.position.x, pose.position.y, pose.position.z]).reshape(3, 1)
            with self.lock:
                self.slam_history.append((time.time(), R, t))

        def on_pose_stamped(self, msg: PoseStamped) -> None:
            self.ingest_pose(msg.pose)

        def on_odom(self, msg: Odometry) -> None:
            self.ingest_pose(msg.pose.pose)

        def on_image(self, msg: Image) -> None:
            gray = tln.TagLocalizerNode._image_to_gray(msg)
            if gray is None:
                return
            with self.lock:
                self.latest_gray = gray
                self.gray_time = time.time()

        def on_info(self, msg: CameraInfo) -> None:
            with self.lock:
                self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    rclpy.init()
    node = CalibNode()
    if args.pose_type == "odometry":
        node.create_subscription(Odometry, args.pose_topic, node.on_odom, 10)
    else:
        node.create_subscription(PoseStamped, args.pose_topic, node.on_pose_stamped, 10)
    node.create_subscription(Image, args.color_topic, node.on_image, 10)
    node.create_subscription(CameraInfo, args.camera_info_topic, node.on_info, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    tag_sizes = {tid: m["size_m"] for tid, m in metas.items()}

    def latest_slam(max_age: float):
        with node.lock:
            if not node.slam_history:
                return None
            t_recv, R, tvec = node.slam_history[-1]
        if time.time() - t_recv > max_age:
            return None
        return R, tvec

    def wait_slam_available(timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        print("等待 SLAM 位姿与相机内参...", end="", flush=True)
        while time.time() < deadline:
            with node.lock:
                has_k = node.K is not None
            if latest_slam(1.0) is not None and has_k:
                print(" OK")
                return True
            time.sleep(0.2)
            print(".", end="", flush=True)
        print(" 超时")
        return False

    def slam_stable() -> tuple[bool, str]:
        with node.lock:
            hist = list(node.slam_history)
        if len(hist) < args.slam_stable_samples:
            return False, f"样本不足 {len(hist)}/{args.slam_stable_samples}"
        recent = hist[-args.slam_stable_samples:]
        if time.time() - recent[-1][0] > 1.0:
            return False, "位姿不新鲜"
        for prev, cur in zip(recent, recent[1:]):
            dist = float(np.linalg.norm(cur[2] - prev[2]))
            dyaw = abs(angle_diff(yaw_of_rotation(cur[1]), yaw_of_rotation(prev[1])))
            if dist > args.slam_stable_pos or dyaw > args.slam_stable_yaw:
                return False, f"仍在运动 dist={dist:.3f}m dyaw={math.degrees(dyaw):.1f}°"
        _, R, tvec = recent[-1]
        return True, f"x={tvec[0,0]:.3f} y={tvec[1,0]:.3f} yaw={math.degrees(yaw_of_rotation(R)):.1f}°"

    def wait_slam_stable(timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        last_report = 0.0
        while time.time() < deadline:
            ok, detail = slam_stable()
            if ok:
                print(f"  SLAM 稳定：{detail}")
                return True
            if time.time() - last_report > 1.0:
                print(f"  等待 SLAM 稳定（{detail}）...")
                last_report = time.time()
            time.sleep(0.1)
        print("  SLAM 稳定等待超时")
        return False

    def detect(gray: np.ndarray):
        with node.lock:
            K = node.K
        if K is None:
            return []
        camera_params = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        dets = backend.detect_poses(gray, camera_params, tag_sizes, default_size)
        return [d for d in dets if d.margin >= args.min_decision_margin]

    def collect_pairs(tid: int, n: int, timeout: float):
        """采集 (SLAM 相机位姿, tag 检测) 配对样本。"""
        pairs = []
        deadline = time.time() + timeout
        last_gray_time = -1.0
        no_tag_frames = 0
        while len(pairs) < n and time.time() < deadline:
            with node.lock:
                gray = node.latest_gray
                gray_t = node.gray_time
            if gray is None or time.time() - gray_t > 0.5 or gray_t == last_gray_time:
                time.sleep(0.02)
                continue
            slam = latest_slam(0.3)
            if slam is None:
                time.sleep(0.02)
                continue
            last_gray_time = gray_t
            dets = detect(gray)
            det = max((d for d in dets if d.tag_id == tid), key=lambda d: d.quality, default=None)
            if det is None:
                no_tag_frames += 1
                if no_tag_frames == 30:
                    print(f"  （连续 {no_tag_frames} 帧未见 tag {tid}，请确认相机正对 tag、距离适中）")
                time.sleep(0.02)
                continue
            no_tag_frames = 0
            pairs.append((slam[0], slam[1], det))
            print(f"\r  已采集 {len(pairs)}/{n}", end="", flush=True)
            time.sleep(0.01)
        print()
        return pairs

    def summarize_tag_samples(pairs):
        samples = []
        for R_wc, t_wc, det in pairs:
            R_wt, p_wt = compute_tag_world_pose(R_wc, t_wc, det.R_cam_tag, det.t_cam_tag)
            samples.append((R_wt, p_wt))
        positions = np.hstack([p for _, p in samples])
        med = np.median(positions, axis=1).reshape(3, 1)
        dists = np.linalg.norm(positions - med, axis=0)
        thresh = max(0.03, 3.0 * float(dists.std()))
        keep = [i for i, d in enumerate(dists) if d <= thresh]
        if len(keep) < max(3, len(samples) // 2):
            keep = list(range(len(samples)))  # 剔除过多说明阈值不适配，退回全量
        R_avg = average_rotations([samples[i][0] for i in keep])
        p_avg = np.mean(np.hstack([samples[i][1] for i in keep]), axis=1).reshape(3, 1)
        spread = float(np.linalg.norm(np.hstack([samples[i][1] for i in keep]) - p_avg, axis=0).max())
        return R_avg, p_avg, len(keep), len(samples), spread

    def prompt(text: str) -> str:
        try:
            return input(text).strip().lower()
        except EOFError:
            return "q"

    # ------------------------------------------------------------ 主循环
    if not wait_slam_available():
        print("未收到 SLAM 位姿或相机内参，请确认话题名与系统状态")
        return 2

    tag_ids = sorted(metas.keys()) if args.ids == "all" else [int(x) for x in args.ids.split(",") if x.strip()]
    unknown = [t for t in tag_ids if t not in metas]
    if unknown:
        print(f"[错误] 以下 id 不在 tags.yaml 模板中：{unknown}")
        return 2

    calibrated: dict[int, dict] = {}
    verify_report: dict[int, dict] = {}

    for tid in tag_ids:
        meta = metas[tid]
        print(f"\n========== tag {tid}（{meta['name']}）==========")
        while True:
            ans = prompt(f"把狗开到能看到 tag {tid} 的位置、SLAM 定位正常后按 Enter（q 退出）：")
            if ans == "q":
                print("已中止")
                return 3
            if not wait_slam_stable():
                continue

            if verify:
                world = next(e.get("world", {}) for e in data.get("tags", []) if int(e["id"]) == tid)
                R_wt = euler_zyx_to_rotation(
                    float(world.get("roll_deg", 0.0)),
                    float(world.get("pitch_deg", 0.0)),
                    float(world.get("yaw_deg", 0.0)),
                )
                p_wt = np.array([float(world.get("x", 0.0)), float(world.get("y", 0.0)), float(world.get("z", 0.0))]).reshape(3, 1)
                if abs(p_wt).max() < 1e-6:
                    print("  该 tag 尚未标定（占位坐标），跳过验证")
                    break
                pairs = collect_pairs(tid, args.verify_samples, args.collect_timeout)
                if len(pairs) < max(3, args.verify_samples // 3):
                    print(f"  有效样本太少（{len(pairs)}），请调整位置后重试")
                    continue
                pos_errs, yaw_errs = [], []
                for R_wc, t_wc, det in pairs:
                    R_hat, t_hat = camera_pose_from_tag(R_wt, p_wt, det.R_cam_tag, det.t_cam_tag)
                    pos_errs.append(float(np.linalg.norm(t_hat - t_wc)))
                    yaw_errs.append(abs(angle_diff(yaw_of_rotation(R_hat), yaw_of_rotation(R_wc))))
                mp, my = float(np.mean(pos_errs)), float(np.mean(yaw_errs))
                passed = mp <= args.verify_max_pos_err and my <= math.radians(args.verify_max_yaw_err_deg)
                verify_report[tid] = {"pos": mp, "yaw_deg": math.degrees(my), "n": len(pairs), "pass": passed}
                print(
                    f"  验证结果：平均位置误差 {mp*100:.1f}cm，平均 yaw 误差 {math.degrees(my):.2f}°"
                    f"（{len(pairs)} 帧）→ {'通过' if passed else '超差，建议重标定该 tag'}"
                )
                break

            pairs = collect_pairs(tid, args.samples, args.collect_timeout)
            if len(pairs) < max(5, args.samples // 3):
                print(f"  有效样本太少（{len(pairs)}），请调整位置后重试")
                continue
            R_avg, p_avg, kept, total, spread = summarize_tag_samples(pairs)
            roll, pitch, yaw = rotation_to_euler_zyx_deg(R_avg)
            print(
                f"  结果（{kept}/{total} 帧有效，最大散布 {spread*100:.1f}cm）：\n"
                f"    x={p_avg[0,0]:.4f}  y={p_avg[1,0]:.4f}  z={p_avg[2,0]:.4f}\n"
                f"    yaw={yaw:.2f}°  pitch={pitch:.2f}°  roll={roll:.2f}°"
            )
            if spread > 0.10:
                print("  [警告] 样本散布偏大（>10cm），可能采集中狗在移动或 SLAM 漂移，建议重采")
            ans = prompt("  Enter=接受  r=重采  s=跳过该 tag  q=退出：")
            if ans == "q":
                print("已中止")
                return 3
            if ans == "r":
                continue
            if ans == "s":
                break
            calibrated[tid] = {
                "x": float(p_avg[0, 0]), "y": float(p_avg[1, 0]), "z": float(p_avg[2, 0]),
                "yaw": yaw, "pitch": pitch, "roll": roll,
            }
            if args.yes:
                break
            break

    # ------------------------------------------------------------ 收尾
    if verify:
        print("\n========== 验证汇总 ==========")
        for tid, rep in verify_report.items():
            flag = "PASS" if rep["pass"] else "FAIL"
            print(f"  tag {tid}: 位置 {rep['pos']*100:.1f}cm / yaw {rep['yaw_deg']:.2f}° ({rep['n']}帧) [{flag}]")
        failed = [tid for tid, rep in verify_report.items() if not rep["pass"]]
        if failed:
            print(f"\n有 {len(failed)} 个 tag 超差：{failed}。建议对这些 tag 重新标定（去掉 --verify）。")
            return 1
        print("\n全部通过。可将 config/guosai_final.yaml 的 tag_localizer.enabled 置为 true。")
        return 0

    if not calibrated:
        print("没有标定任何 tag，tags.yaml 不变")
        return 0
    backup = write_tags_yaml(args.tags_yaml, data, calibrated, args.dry_run)
    if backup:
        print(f"\n已写入 {args.tags_yaml}（原文件备份：{backup.name}），共标定 {len(calibrated)} 个 tag")
        print("下一步：python3 tools/calibrate_tags.py --tags-yaml "
              f"{args.tags_yaml} --verify   # 验证兜底精度")
    return 0


# ================================================================ CLI

def main() -> int:
    parser = argparse.ArgumentParser(description="AprilTag 世界坐标现场标定 / 验证工具")
    parser.add_argument("--tags-yaml", default=str(REPO_ROOT / "config" / "tags.yaml"))
    parser.add_argument("--ids", default="all", help="要标定的 tag id，逗号分隔；默认 all")
    parser.add_argument("--samples", type=int, default=30, help="每个 tag 采集帧数")
    parser.add_argument("--collect-timeout", type=float, default=45.0, help="单 tag 采集超时（秒）")
    parser.add_argument("--pose-topic", default="/camera_pose")
    parser.add_argument("--pose-type", default="pose_stamped", choices=["pose_stamped", "odometry"])
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--slam-stable-samples", type=int, default=10)
    parser.add_argument("--slam-stable-pos", type=float, default=0.03, help="SLAM 稳定判据：相邻位姿位移上限（m）")
    parser.add_argument("--slam-stable-yaw", type=float, default=0.10, help="SLAM 稳定判据：相邻 yaw 变化上限（rad）")
    parser.add_argument("--min-decision-margin", type=float, default=12.0)
    parser.add_argument("--verify", action="store_true", help="验证模式：用已标定坐标反推相机位姿，与 SLAM 对比")
    parser.add_argument("--verify-samples", type=int, default=15)
    parser.add_argument("--verify-max-pos-err", type=float, default=0.10, help="验证通过阈值：平均位置误差（m）")
    parser.add_argument("--verify-max-yaw-err-deg", type=float, default=5.0, help="验证通过阈值：平均 yaw 误差（度）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将写入的内容，不落盘")
    parser.add_argument("--yes", action="store_true", help="采完自动接受，不二次确认")
    parser.add_argument("--self-test", action="store_true", help="无 ROS 几何自检")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    tags_path = Path(args.tags_yaml).expanduser().resolve()
    if not tags_path.exists():
        print(f"[错误] 找不到 {tags_path}")
        return 2
    data, family, default_size, metas = load_template(tags_path)
    if not metas:
        print(f"[错误] {tags_path} 里没有 tags 条目")
        return 2
    args.tags_yaml = tags_path
    return run_ros(args, data, family, default_size, metas, verify=args.verify)


if __name__ == "__main__":
    sys.exit(main())
