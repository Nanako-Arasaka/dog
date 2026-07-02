#!/usr/bin/env python3
"""Probe Intel RealSense D435i color + aligned depth.

This script is for Jetson-side debugging. It does not run YOLO and does not
control the robot. It verifies that depth is aligned to the RGB frame and that
pixel depth values are reasonable before bbox + depth localization is added.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe RealSense aligned depth.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS.")
    parser.add_argument("--roi", type=int, default=40, help="Center ROI size in pixels.")
    parser.add_argument("--save-dir", type=Path, default=Path("debug/realsense_probe"), help="Debug image output directory.")
    parser.add_argument("--save-every", type=int, default=30, help="Save one debug frame every N frames. 0 disables saving.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means run until Ctrl-C.")
    parser.add_argument("--no-display", action="store_true", help="Do not open OpenCV windows.")
    return parser.parse_args()


def depth_stats(depth_m: np.ndarray, roi_size: int) -> dict[str, float | int | None]:
    h, w = depth_m.shape[:2]
    half = max(1, roi_size // 2)
    cx, cy = w // 2, h // 2
    roi = depth_m[max(0, cy - half) : min(h, cy + half), max(0, cx - half) : min(w, cx + half)]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    valid = valid[(valid >= 0.20) & (valid <= 5.0)]
    if valid.size == 0:
        return {
            "valid_count": 0,
            "valid_ratio": 0.0,
            "min_m": None,
            "median_m": None,
            "max_m": None,
        }
    return {
        "valid_count": int(valid.size),
        "valid_ratio": float(valid.size / max(1, roi.size)),
        "min_m": float(np.min(valid)),
        "median_m": float(np.median(valid)),
        "max_m": float(np.max(valid)),
    }


def make_depth_vis(depth_m: np.ndarray) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, 3.0)
    depth_u8 = (clipped / 3.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def draw_overlay(color_bgr: np.ndarray, depth_m: np.ndarray, roi_size: int, stats: dict[str, float | int | None]) -> np.ndarray:
    overlay = color_bgr.copy()
    h, w = overlay.shape[:2]
    half = max(1, roi_size // 2)
    cx, cy = w // 2, h // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w - 1, cx + half), min(h - 1, cy + half)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
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


def main() -> None:
    args = parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "Missing pyrealsense2. Install librealsense/pyrealsense2 on the Jetson, "
            "then run this script there."
        ) from exc

    args.save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    print("RealSense aligned depth probe started.")
    print(f"depth_scale={depth_scale}")
    print(
        "color_intrinsics="
        + json.dumps(
            {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            },
            ensure_ascii=False,
        )
    )
    print("Put the cone at 0.5m, 1.0m, and 1.5m in the center ROI and compare median_m.")

    frame_id = 0
    last_print = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                print("Missing color or depth frame.")
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            stats = depth_stats(depth_m, args.roi)
            now = time.time()
            if now - last_print >= 1.0:
                print(
                    json.dumps(
                        {
                            "frame": frame_id,
                            "color_shape": list(color_bgr.shape),
                            "depth_shape": list(depth_m.shape),
                            "depth_dtype": str(depth_raw.dtype),
                            "center_roi": stats,
                        },
                        ensure_ascii=False,
                    )
                )
                last_print = now

            overlay = draw_overlay(color_bgr, depth_m, args.roi, stats)
            depth_vis = make_depth_vis(depth_m)

            if args.save_every > 0 and frame_id % args.save_every == 0:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(args.save_dir / f"color_{stamp}_{frame_id:06d}.jpg"), color_bgr)
                cv2.imwrite(str(args.save_dir / f"aligned_depth_vis_{stamp}_{frame_id:06d}.jpg"), depth_vis)
                cv2.imwrite(str(args.save_dir / f"overlay_{stamp}_{frame_id:06d}.jpg"), overlay)

            if not args.no_display:
                cv2.imshow("color + center depth", overlay)
                cv2.imshow("aligned depth vis", depth_vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
