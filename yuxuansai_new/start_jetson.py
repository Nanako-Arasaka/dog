#!/usr/bin/env python3
"""Jetson 端：摄像头采集 → 圆检测裁剪 → TensorRT/PyTorch 推理 → UDP 发送结果。"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from perception import (
    CLASS_LABEL_ZH,
    DashboardCameraDetector,
    HoughCircleCropper,
    SwitchConfirm,
    draw_result,
    make_unknown_crop,
    open_camera_with_fallback,
)


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Jetson 端感知发送程序（UDP）")
    parser.add_argument("--target-ip", default="192.168.1.3", help="机器狗主控 IP")
    parser.add_argument("--target-port", type=int, default=5005, help="机器狗主控 UDP 端口")
    parser.add_argument("--camera-device", default=os.environ.get("DASHBOARD_CAM_DEVICE", ""), help="摄像头设备路径")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引")
    parser.add_argument("--camera-width", type=int, default=640, help="摄像头宽度")
    parser.add_argument("--camera-height", type=int, default=480, help="摄像头高度")
    parser.add_argument("--camera-fps", type=int, default=30, help="摄像头 FPS")
    parser.add_argument("--send-hz", type=float, default=15.0, help="UDP 发送频率")
    parser.add_argument("--input-size", type=int, default=160, help="推理输入尺寸")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--hough-interval", type=int, default=3, help="霍夫圆执行间隔帧数")
    parser.add_argument("--hough-sleep-ms", type=float, default=1.0, help="霍夫未命中时休眠毫秒")
    parser.add_argument("--log-interval", type=int, default=30, help="每发送多少帧打印一次日志")
    parser.add_argument(
        "--engine-path",
        default=str(root_dir / "checkpoints" / "model_fp16_160.engine"),
        help="TensorRT 引擎路径",
    )
    parser.add_argument(
        "--model-path",
        default=str(root_dir / "checkpoints" / "final_dashboard_resnet20_v2.pth"),
        help="PyTorch 权重路径（当禁用 TensorRT 时使用）",
    )
    parser.add_argument("--class-names", default="high,normal,low", help="类别顺序，逗号分隔")
    parser.add_argument("--preprocess-mode", default="resize_center_crop",
                        choices=["resize", "resize_center_crop"], help="前处理模式")
    parser.add_argument("--disable-tensorrt", action="store_true", help="禁用 TensorRT，改用 PyTorch CUDA 推理")
    parser.add_argument("--switch-confirm-frames", type=int, default=2, help="类别切换确认帧数")
    parser.add_argument("--cls-confirm-window", type=int, default=2, help="分类确认窗口大小")
    parser.add_argument("--visualize", action="store_true", help="显示可视化窗口（按 q 退出）")
    parser.add_argument("--show-crop", action="store_true", help="可视化时显示裁剪窗口")
    parser.add_argument("--font-path", default="", help="中文字体路径（PIL），留空自动查找")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.send_hz <= 0:
        raise ValueError("--send-hz 必须大于 0")

    engine_path = args.engine_path
    if not args.disable_tensorrt and not Path(engine_path).exists():
        fallback = Path(__file__).resolve().parent / "checkpoints" / "model_fp16.engine"
        if fallback.exists():
            print(f"未找到 {engine_path}，回退到 {fallback}")
            engine_path = str(fallback)

    class_names = [item.strip() for item in str(args.class_names).split(",") if item.strip()]
    if not class_names:
        class_names = ["high", "normal", "low"]

    cropper = HoughCircleCropper(
        detect_interval=args.hough_interval,
        miss_sleep_ms=args.hough_sleep_ms,
    )
    detector = DashboardCameraDetector(
        model_path=args.model_path,
        device="cuda",
        use_tensorrt=not args.disable_tensorrt,
        engine_path=engine_path,
        input_size=args.input_size,
        confidence_threshold=args.confidence_threshold,
        cropper=cropper,
        class_names=class_names,
        preprocess_mode=args.preprocess_mode,
        cls_confirm_window=args.cls_confirm_window,
    )

    if args.camera_device:
        cap = cv2.VideoCapture(args.camera_device, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"无法通过设备路径打开摄像头: {args.camera_device}，改用索引 {args.camera_index}")
            cap = open_camera_with_fallback(args.camera_index)
    else:
        cap = open_camera_with_fallback(args.camera_index)

    if cap is None or not cap.isOpened():
        raise RuntimeError("无法打开摄像头。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.target_ip, args.target_port)
    print(f"UDP 发送目标: {target[0]}:{target[1]}")
    print(f"发送频率上限: {args.send_hz:.2f} Hz")
    print(f"切换确认帧数: {max(1, args.switch_confirm_frames)}")
    print("开始推理并发送...")
    switch_confirm = SwitchConfirm(confirm_frames=args.switch_confirm_frames)

    send_interval = 1.0 / args.send_hz
    next_send_time = time.perf_counter()
    send_count = 0
    start_time = next_send_time
    fps_tick = next_send_time
    fps_count = 0
    fps_value = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.perf_counter()
            if now < next_send_time:
                continue

            _, class_name, confidence, probabilities, crop_img, detected, cx, cy = detector.predict(frame)
            stable_class, stable_detected, stable_confidence = switch_confirm.update(
                class_name, detected, confidence)
            class_zh = CLASS_LABEL_ZH.get(stable_class, "未知")
            payload = {
                "class": class_zh,
                "class_en": stable_class,
                "confidence": round(float(stable_confidence), 4),
                "detected": bool(stable_detected),
                "cx": int(cx) if cx is not None else -1,
                "cy": int(cy) if cy is not None else -1,
            }
            sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), target)
            send_count += 1

            while next_send_time <= now:
                next_send_time += send_interval

            if args.visualize:
                fps_count += 1
                if now - fps_tick >= 1.0:
                    fps_value = fps_count / max(1e-6, now - fps_tick)
                    fps_count = 0
                    fps_tick = now

                result = draw_result(
                    frame,
                    class_name=stable_class,
                    confidence=stable_confidence,
                    probabilities=probabilities,
                    class_names=detector.class_names,
                    fps=fps_value,
                    detected=stable_detected,
                    font_path=args.font_path,
                )
                cv2.imshow("Dashboard Detection", result)
                if args.show_crop:
                    crop_view = crop_img if crop_img is not None else make_unknown_crop()
                    cv2.imshow("Dashboard Crop", crop_view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("收到退出按键，正在退出...")
                    break

            if args.log_interval > 0 and send_count % args.log_interval == 0:
                elapsed = max(1e-6, now - start_time)
                actual_hz = send_count / elapsed
                print(f"[{send_count}] send_hz={actual_hz:.2f} payload={payload}")
    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")
    finally:
        cap.release()
        sock.close()
        if args.visualize:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
