#!/usr/bin/env python3
"""本地推理入口（不使用网络桥接）。"""
import argparse
from pathlib import Path


def parse_args():
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="本地摄像头推理（TRT / PyTorch）")
    parser.add_argument("--mode", choices=["trt", "torch"], default="trt", help="推理后端")
    parser.add_argument(
        "--engine-path",
        default=str(root_dir / "checkpoints" / "resnet18_dashboard.trt"),
        help="TensorRT 引擎路径",
    )
    parser.add_argument(
        "--model-path",
        default=str(root_dir / "checkpoints" / "model_best.pth"),
        help="PyTorch 权重路径",
    )
    parser.add_argument("--camera-device", default="/dev/video5", help="摄像头设备路径")
    parser.add_argument("--camera-index", type=int, default=5, help="摄像头索引（备用）")
    parser.add_argument("--width", type=int, default=640, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=480, help="摄像头高度")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "trt":
        from detect_dashboard_trt import run_dashboard

        run_dashboard(
            camera_device=args.camera_device,
            camera_index=args.camera_index,
            engine_path=args.engine_path,
            width=args.width,
            height=args.height,
        )
    else:
        from Dashboard_detec2t import run_dashboard

        run_dashboard(
            camera_device=args.camera_device,
            camera_index=args.camera_index,
            model_path=args.model_path,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
