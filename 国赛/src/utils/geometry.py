"""坐标变换工具函数。"""

from __future__ import annotations

import math

import numpy as np

from core.types import Point3D


def pixel_to_camera_3d(
    u: float, v: float, depth: float,
    fx: float, fy: float, cx: float, cy: float,
) -> Point3D:
    """将像素坐标 + 深度值转换为相机坐标系 3D 点。

    Args:
        u, v:     像素坐标。
        depth:    深度值（米）。
        fx, fy:   焦距（像素）。
        cx, cy:   主点偏移。

    Returns:
        (x, y, z) 相机坐标系 (m)。
    """
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)


def camera_to_world(
    point_cam: Point3D,
    camera_pose: tuple[float, float, float, float, float, float],
) -> Point3D:
    """将相机坐标系 3D 点变换到世界坐标系。

    Args:
        point_cam:   (x, y, z) 相机坐标系。
        camera_pose: (tx, ty, tz, roll, pitch, yaw) 相机在世界坐标系中的位姿。

    Returns:
        (x, y, z) 世界坐标系。
    """
    tx, ty, tz, roll, pitch, yaw = camera_pose
    # 旋转矩阵 ZYX 顺序
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

    p = np.array(point_cam)
    world = R @ p + np.array([tx, ty, tz])
    return (float(world[0]), float(world[1]), float(world[2]))


def transform_point(
    point: Point3D,
    T: np.ndarray,
) -> Point3D:
    """用 4x4 齐次变换矩阵变换 3D 点。

    Args:
        point: (x, y, z)。
        T:     (4, 4) 变换矩阵。

    Returns:
        变换后的 (x, y, z)。
    """
    p = np.array([point[0], point[1], point[2], 1.0])
    res = T @ p
    return (float(res[0]), float(res[1]), float(res[2]))


def quaternion_to_euler(
    qw: float, qx: float, qy: float, qz: float,
) -> tuple[float, float, float]:
    """四元数 → 欧拉角 (roll, pitch, yaw)。"""
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)
