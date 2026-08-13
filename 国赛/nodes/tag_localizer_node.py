#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AprilTag 绝对定位节点（SLAM 丢失兜底）。

订阅 RealSense RGB + camera_info，用官方 AprilTag 库（tag36h11）检测 tag，
结合每个 tag 的已知世界坐标反推「相机在世界坐标系」的位姿，发布到独立话题
/tag_localizer/pose。watchdog 仲裁时把它作为 /camera_pose 丢失时的兜底源。

后端优先级：
  1. `apriltag` 官方库（精度最高、检测距离最远、角度最鲁棒）—— tag36h11
  2. OpenCV `cv2.aruco` 的 DICT_APRILTAG_36h11（零额外依赖的降级）

发布话题：
  /tag_localizer/pose       PoseStamped   相机在 SLAM 世界系的位姿
  /tag_localizer/status     String        "ok:id=N,conf=..." 或 "none"
  /tag_localizer/seen_tags  String        逗号分隔的可见 tag ID
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


@dataclass
class TagSpec:
    id: int
    size_m: float
    name: str
    note: str
    R_w: np.ndarray
    p_w: np.ndarray


@dataclass
class Detection:
    tag_id: int
    R_cam_tag: np.ndarray   # 3x3：tag 在相机系下的旋转
    t_cam_tag: np.ndarray   # 3x1：tag 在相机系下的位置
    quality: float          # 0~1，越高越可信
    margin: float = float("inf")  # decision_margin；OpenCV 降级后端无此概念，默认 inf 不被 margin 过滤


def euler_zyx_to_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX 顺序欧拉角 → 3x3 旋转矩阵（R = Rz @ Ry @ Rx）。"""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 旋转矩阵 → 四元数 (x, y, z, w)。"""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (x, y, z, w)


def load_tags(path_text: str) -> tuple[str, float, dict[int, TagSpec]]:
    path = Path(os.path.expandvars(path_text or ""))
    if not path.exists():
        raise FileNotFoundError(f"tags.yaml not found: {path_text}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    family = str(data.get("tag_family", "tag36h11"))
    default_size = float(data.get("default_size_m", 0.20))
    out: dict[int, TagSpec] = {}
    for entry in data.get("tags", []):
        world = entry.get("world", {})
        tid = int(entry["id"])
        R_w = euler_zyx_to_rotation(
            float(world.get("roll_deg", 0.0)),
            float(world.get("pitch_deg", 0.0)),
            float(world.get("yaw_deg", 0.0)),
        )
        p_w = np.array(
            [
                float(world.get("x", 0.0)),
                float(world.get("y", 0.0)),
                float(world.get("z", 0.0)),
            ]
        ).reshape(3, 1)
        out[tid] = TagSpec(
            id=tid,
            size_m=float(entry.get("size_m", default_size)),
            name=str(entry.get("name", f"tag_{tid}")),
            note=str(entry.get("note", "")),
            R_w=R_w,
            p_w=p_w,
        )
    return family, default_size, out


class AprilTagBackend:
    """统一检测接口。子类返回带位姿的 Detection 列表。"""

    family: str = "tag36h11"

    def __init__(self, family: str) -> None:
        self.family = family

    def detect_poses(
        self,
        gray: np.ndarray,
        camera_params: tuple[float, float, float, float],
        tag_sizes: dict[int, float],
        default_size: float,
    ) -> list[Detection]:
        raise NotImplementedError


class AprilTagLibBackend(AprilTagBackend):
    """官方 apriltag 库后端（精度最高）。"""

    def __init__(self, family: str) -> None:
        super().__init__(family)
        import apriltag  # noqa: F401

        self._apriltag = apriltag
        options = apriltag.DetectorOptions(
            families=family,
            nthreads=4,
            quad_decimate=1.0,      # 全分辨率，精度优先
            quad_sigma=0.0,
            refine_edges=True,      # 亚像素边缘细化
            decode_sharpening=0.25,
            debug=False,
        )
        self.detector = apriltag.Detector(options)

    def detect_poses(self, gray, camera_params, tag_sizes, default_size):
        # 关键：Python apriltag 库只有在 detect() 传入 camera_params + tag_size 时
        # 才会解算位姿（填充 pose_t/pose_R）；只给 detector 设属性无效。
        raw = None
        for attempt in (
            lambda: self.detector.detect(gray, camera_params=camera_params, tag_size=default_size),
            lambda: self.detector.detect(gray, camera_params, default_size),
            lambda: self.detector.detect(gray),
        ):
            try:
                raw = attempt()
                break
            except TypeError:
                # 老版本绑定不支持位姿参数，逐级降级
                continue
        if raw is None:
            return []
        out: list[Detection] = []
        for det in raw:
            tid = int(det.tag_id)
            size = tag_sizes.get(tid, default_size)
            pose_t, pose_R = self._detection_pose(det, camera_params, size, default_size)
            if pose_t is None:
                continue
            margin = float(getattr(det, "decision_margin", 0.0))
            out.append(
                Detection(
                    tag_id=tid,
                    R_cam_tag=pose_R,
                    t_cam_tag=pose_t,
                    quality=min(1.0, max(0.0, margin / 40.0)),
                    margin=margin,
                )
            )
        return out

    @staticmethod
    def _detection_pose(det, camera_params, size, default_size):
        """按 tag 自身尺寸解算位姿。返回 (t_cam_tag 3x1, R_cam_tag 3x3) 或 (None, None)。"""
        est = getattr(det, "estimate_tag_pose", None)
        if callable(est):
            # 不同版本绑定签名可能是 (fx, fy, cx, cy, size) 或 ((fx, fy, cx, cy), size)
            for args in ((*camera_params, size), (camera_params, size)):
                try:
                    a, b = est(*args)
                except Exception:  # noqa: BLE001
                    continue
                a = np.asarray(a, dtype=float)
                b = np.asarray(b, dtype=float)
                if a.size == 3 and b.size == 9:
                    return a.reshape(3, 1), b.reshape(3, 3)
                if a.size == 9 and b.size == 3:
                    return b.reshape(3, 1), a.reshape(3, 3)
        pose_t = getattr(det, "pose_t", None)
        pose_R = getattr(det, "pose_R", None)
        if pose_t is None or pose_R is None:
            return None, None
        # detect() 用 default_size 解算；尺寸不同的 tag 平移按线性缩放，旋转不受尺寸影响
        scale = (size / default_size) if default_size > 0 else 1.0
        return (
            np.asarray(pose_t, dtype=float).reshape(3, 1) * scale,
            np.asarray(pose_R, dtype=float).reshape(3, 3),
        )


class OpenCVAprilTagBackend(AprilTagBackend):
    """OpenCV ArUco 的 AprilTag 字典降级后端（零额外依赖）。"""

    def __init__(self, family: str) -> None:
        super().__init__(family)
        import cv2

        self._cv2 = cv2
        name = "DICT_APRILTAG_" + family.replace("tag", "").upper()
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        except AttributeError:
            raise RuntimeError(f"OpenCV does not support {name}, need opencv>=4.6")
        # 新版 OpenCV（>=4.10/5.x）移除了 DetectorParameters_create
        factory = getattr(cv2.aruco, "DetectorParameters_create", None)
        self.params = factory() if factory else cv2.aruco.DetectorParameters()
        # 新版 OpenCV 同时移除了函数式 detectMarkers，优先用面向对象的 ArucoDetector
        self._detector = (
            cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros(5)

    def detect_poses(self, gray, camera_params, tag_sizes, default_size):
        cv2 = self._cv2
        K = self._camera_matrix
        if K is None:
            fx, fy, cx, cy = camera_params
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        out: list[Detection] = []
        if ids is None:
            return out
        flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        for i, tid in enumerate(ids.flatten()):
            tid = int(tid)
            size = tag_sizes.get(tid, default_size)
            half = size / 2.0
            # ArUco 角点顺序：左上、右上、右下、左下（tag 本地系 +Z 指向正面）
            obj_pts = np.array(
                [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
                dtype=np.float32,
            )
            img_pts = np.asarray(corners[i], dtype=np.float32).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, self._dist_coeffs, flags=flags)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            out.append(
                Detection(
                    tag_id=tid,
                    R_cam_tag=R,
                    t_cam_tag=np.asarray(tvec, dtype=float).reshape(3, 1),
                    quality=0.8,
                    margin=float("inf"),  # ArUco 无 decision_margin 概念，跳过 margin 过滤
                )
            )
        return out

    def set_camera(self, K: np.ndarray, dist: np.ndarray) -> None:
        self._camera_matrix = K
        self._dist_coeffs = dist


def make_backend(family: str) -> tuple[AprilTagBackend, Optional[str]]:
    """优先官方 apriltag 库，失败降级 OpenCV AprilTag。

    返回 (backend, downgrade_reason)；用官方库时 downgrade_reason 为 None。
    """
    try:
        return AprilTagLibBackend(family), None
    except Exception as exc:  # noqa: BLE001
        backend = OpenCVAprilTagBackend(family)
        return backend, f"官方 apriltag 库不可用（{exc}），已降级 OpenCV ArUco"


class TagLocalizerNode(Node):
    def __init__(self) -> None:
        super().__init__("tag_localizer_node")
        self.declare_parameter("tags_yaml", "")
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("pose_topic", "/tag_localizer/pose")
        self.declare_parameter("status_topic", "/tag_localizer/status")
        self.declare_parameter("seen_tags_topic", "/tag_localizer/seen_tags")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("detect_hz", 10.0)
        self.declare_parameter("min_decision_margin", 12.0)

        self.tags_yaml = str(self.get_parameter("tags_yaml").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.seen_tags_topic = str(self.get_parameter("seen_tags_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.detect_interval = 1.0 / max(1.0, float(self.get_parameter("detect_hz").value))
        self.min_decision_margin = float(self.get_parameter("min_decision_margin").value)

        self.family, self.default_tag_size, self.tags = load_tags(self.tags_yaml)
        self.backend, downgrade_note = make_backend(self.family)

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs = np.zeros(5)
        self.last_detect_time = 0.0

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.seen_pub = self.create_publisher(String, self.seen_tags_topic, 10)

        self.create_subscription(Image, self.color_topic, self._on_image, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, 10)

        if downgrade_note:
            self.get_logger().warn(downgrade_note)
        self.get_logger().info(
            f"tag_localizer ready backend={type(self.backend).__name__} "
            f"family={self.family} tags={len(self.tags)} default_size={self.default_tag_size:.3f}m"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=float).reshape(-1)
        if isinstance(self.backend, OpenCVAprilTagBackend):
            self.backend.set_camera(self.camera_matrix, self.dist_coeffs)
        # 官方 apriltag 库的内参通过 detect(camera_params=...) 逐帧传入
        # （给 detector 设属性无效），见 _detect_with_sizes

    def _on_image(self, msg: Image) -> None:
        if self.camera_matrix is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_detect_time < self.detect_interval:
            return
        self.last_detect_time = now

        gray = self._image_to_gray(msg)
        if gray is None:
            return

        # 官方库后端：给每个 tag 设尺寸后由库解算（尺寸可能不同）
        detections = self._detect_with_sizes(gray)
        if not detections:
            self.status_pub.publish(String(data="none"))
            self.seen_pub.publish(String(data=""))
            return

        # 选质量最高 / 最近的一个作为主位姿（只在已标定的 tag 里选）
        detections.sort(key=lambda d: (d.quality, -float(np.linalg.norm(d.t_cam_tag))), reverse=True)
        known = [d for d in detections if d.tag_id in self.tags]
        if not known:
            self.status_pub.publish(String(data=f"unknown_tag:{detections[0].tag_id}"))
            self.seen_pub.publish(String(data=",".join(str(d.tag_id) for d in detections)))
            return
        best = known[0]
        spec = self.tags[best.tag_id]

        R_world_cam, t_world_cam = self._camera_world_pose(spec, best.R_cam_tag, best.t_cam_tag)
        qx, qy, qz, qw = rotation_to_quaternion(R_world_cam)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose = Pose(
            position=Point(x=float(t_world_cam[0, 0]), y=float(t_world_cam[1, 0]), z=float(t_world_cam[2, 0])),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
        self.pose_pub.publish(pose_msg)

        seen = ",".join(str(d.tag_id) for d in detections)
        self.status_pub.publish(String(data=f"ok:id={best.tag_id},q={best.quality:.2f}"))
        self.seen_pub.publish(String(data=seen))

    def _detect_with_sizes(self, gray: np.ndarray) -> list[Detection]:
        if self.camera_matrix is None:
            return []
        fx, fy = float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])
        cx, cy = float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2])
        tag_sizes = {tid: spec.size_m for tid, spec in self.tags.items()}
        detections = self.backend.detect_poses(
            gray, (fx, fy, cx, cy), tag_sizes, self.default_tag_size
        )
        # 过滤低质量检出（OpenCV 降级后端 margin=inf，不受该过滤影响）
        return [d for d in detections if d.margin >= self.min_decision_margin]

    @staticmethod
    def _camera_world_pose(spec: TagSpec, R_cam_tag: np.ndarray, t_cam_tag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # p_cam = R_cam_tag @ p_tag + t_cam_tag   (tag 点 → 相机系)
        # p_world = R_w @ p_tag + p_w_tag          (tag 点 → 世界系)
        # 消去 p_tag：相机在世界系
        R_cam_tag_T = R_cam_tag.T
        R_world_cam = spec.R_w @ R_cam_tag_T
        t_world_cam = spec.p_w - R_world_cam @ t_cam_tag
        return R_world_cam, t_world_cam

    @staticmethod
    def _image_to_gray(msg: Image) -> Optional[np.ndarray]:
        import cv2

        enc = msg.encoding.lower()
        try:
            if enc in ("rgb8", "rgb"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if enc in ("bgr8", "bgr"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            if enc in ("mono8", "gray", "grey"):
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            if enc in ("16uc1",):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                return (arr >> 2).astype(np.uint8)
        except Exception:  # noqa: BLE001
            return None
        return None


def main() -> None:
    rclpy.init()
    node = TagLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
