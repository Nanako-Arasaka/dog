#!/usr/bin/env python3
"""导出 PyTorch checkpoint 为 ONNX 文件。"""
import argparse
from pathlib import Path

import torch

from perception.model import Resnet18_dashboard


def export_to_onnx(checkpoint_path, onnx_save_path, input_size=(3, 160, 160),
                   opset_version=11, dynamic_batch=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_classes = checkpoint["num_classes"] if isinstance(checkpoint, dict) and "num_classes" in checkpoint else 3

    model = Resnet18_dashboard(num_classes=num_classes, pretrained=False, dropout=0.5)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, *input_size)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model, dummy_input, onnx_save_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes, opset_version=int(opset_version),
        do_constant_folding=True, verbose=False,
        export_params=True, external_data=False,
    )
    print(f"ONNX 模型已保存至: {onnx_save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint 路径")
    parser.add_argument("--onnx", default="checkpoints/model_160.onnx", help="输出 ONNX 路径")
    parser.add_argument("--input-size", type=int, default=160, help="输入尺寸（H=W）")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset 版本")
    parser.add_argument("--static-batch", action="store_true", help="禁用动态 batch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(
        Path(args.checkpoint), Path(args.onnx),
        input_size=(3, args.input_size, args.input_size),
        opset_version=args.opset,
        dynamic_batch=not args.static_batch,
    )
