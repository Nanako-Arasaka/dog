#!/usr/bin/env python3
"""Probe aligned RGB-D topics from realsense-ros.

Run this on the Jetson after starting realsense2_camera with depth alignment.
It does not require pyrealsense2; it uses ROS2 topics and cv_bridge.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe ROS2 aligned depth topics.")
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw", help="Color Image topic.")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw", help="Aligned depth Image topic.")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info", help="Color CameraInfo topic.")
    parser.add_argument("--roi", type=int, default=40, help="Center ROI size in pixels.")
    parser.add_argument("--save-dir", type=Path, default=Path("debug/ros2_aligned_depth_probe"), help="Debug image output directory.")
    parser.add_argument("--save-every", type=int, default=30, help="Save one debug frame every N pairs. 0 disables saving.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frame pairs. 0 means run until Ctrl-C.")
    parser.add_argument("--no-display", action="store_true", help="Do not open OpenCV windows.")
    return parser.parse_args()


def convert_depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "16UC1":
        return depth.astype(np.float32) / 1000.0
    if encoding == "32FC1":
        return depth.astype(np.float32)
    raise ValueError(f"Unsupported depth encoding: {encoding}. Expected 16UC1 or 32FC1.")


def center_depth_stats(depth_m: np.ndarray, roi_size: int) -> dict[str, float | int | None]:
    h, w = depth_m.shape[:2]
    half = max(1, roi_size // 2)
    cx, cy = w // 2, h // 2
    roi = depth_m[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    valid = valid[(valid >= 0.20) & (valid <= 5.0)]
    if valid.size == 0:
        return {"valid_count": 0, "valid_ratio": 0.0, "min_m": None, "median_m": None, "max_m": None}
    return {
        "valid_count": int(valid.size),
        "valid_ratio": float(valid.size / max(1, roi.size)),
        "min_m": float(np.min(valid)),
        "median_m": float(np.median(valid)),
        "max_m": float(np.max(valid)),
    }


def depth_vis(depth_m: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, 3.0)
    depth_u8 = (clipped / 3.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def draw_overlay(color_bgr: np.ndarray, roi_size: int, stats: dict[str, float | int | None]) -> np.ndarray:
    overlay = color_bgr.copy()
    h, w = overlay.shape[:2]
    half = max(1, roi_size // 2)
    cx, cy = w // 2, h // 2
    cv2.rectangle(
        overlay,
        (max(0, cx - half), max(0, cy - half)),
        (min(w - 1, cx + half), min(h - 1, cy + half)),
        (0, 255, 255),
        2,
    )
    median = stats["median_m"]
    text = "center depth: invalid" if median is None else f"center depth: {median:.3f} m"
    cv2.putText(overlay, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"valid_ratio: {stats['valid_ratio']:.2f}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


class Ros2AlignedDepthProbe(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ros2_aligned_depth_probe")
        self.args = args
        self.bridge = CvBridge()
        self.latest_color: Image | None = None
        self.latest_depth: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.frame_count = 0
        self.last_print = 0.0
        self.args.save_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(Image, args.rgb_topic, self.on_color, 10)
        self.create_subscription(Image, args.depth_topic, self.on_depth, 10)
        self.create_subscription(CameraInfo, args.info_topic, self.on_info, 10)
        self.create_timer(0.03, self.process_latest)

        self.get_logger().info(f"RGB topic: {args.rgb_topic}")
        self.get_logger().info(f"Depth topic: {args.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {args.info_topic}")

    def on_color(self, msg: Image) -> None:
        self.latest_color = msg

    def on_depth(self, msg: Image) -> None:
        self.latest_depth = msg

    def on_info(self, msg: CameraInfo) -> None:
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(
                "CameraInfo K="
                + json.dumps(
                    {
                        "fx": msg.k[0],
                        "fy": msg.k[4],
                        "cx": msg.k[2],
                        "cy": msg.k[5],
                        "width": msg.width,
                        "height": msg.height,
                    },
                    ensure_ascii=False,
                )
            )

    def process_latest(self) -> None:
        if self.latest_color is None or self.latest_depth is None:
            return

        color_msg = self.latest_color
        depth_msg = self.latest_depth
        self.latest_color = None
        self.latest_depth = None

        color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = convert_depth_to_meters(depth, depth_msg.encoding)

        if color_bgr.shape[:2] != depth_m.shape[:2]:
            self.get_logger().warn(
                f"RGB and depth shapes differ: color={color_bgr.shape[:2]}, depth={depth_m.shape[:2]}. "
                "This is not aligned-to-color data."
            )

        stats = center_depth_stats(depth_m, self.args.roi)
        now = time.time()
        if now - self.last_print >= 1.0:
            print(
                json.dumps(
                    {
                        "frame": self.frame_count,
                        "rgb_shape": list(color_bgr.shape),
                        "depth_shape": list(depth_m.shape),
                        "depth_encoding": depth_msg.encoding,
                        "center_roi": stats,
                    },
                    ensure_ascii=False,
                )
            )
            self.last_print = now

        overlay = draw_overlay(color_bgr, self.args.roi, stats)
        depth_color = depth_vis(depth_m)

        if self.args.save_every > 0 and self.frame_count % self.args.save_every == 0:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(self.args.save_dir / f"color_{stamp}_{self.frame_count:06d}.jpg"), color_bgr)
            cv2.imwrite(str(self.args.save_dir / f"aligned_depth_vis_{stamp}_{self.frame_count:06d}.jpg"), depth_color)
            cv2.imwrite(str(self.args.save_dir / f"overlay_{stamp}_{self.frame_count:06d}.jpg"), overlay)

        if not self.args.no_display:
            cv2.imshow("ros2 color + center depth", overlay)
            cv2.imshow("ros2 aligned depth vis", depth_color)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                rclpy.shutdown()

        self.frame_count += 1
        if self.args.max_frames > 0 and self.frame_count >= self.args.max_frames:
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Ros2AlignedDepthProbe(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
