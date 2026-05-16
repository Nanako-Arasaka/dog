#!/usr/bin/env python3
"""构建 TensorRT 引擎（默认 FP32，可选 FP16）。"""
import argparse
from pathlib import Path

import tensorrt as trt


def build_engine(onnx_path, engine_path, input_size=160, fp16=False, workspace_size=1 << 30):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("解析 ONNX 失败：")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用 FP16 优化")

    profile = builder.create_optimization_profile()
    input_size = int(input_size)
    input_name = network.get_input(0).name
    profile.set_shape(input_name,
                      (1, 3, input_size, input_size),
                      (1, 3, input_size, input_size),
                      (4, 3, input_size, input_size))
    config.add_optimization_profile(profile)

    print(f"正在构建 TensorRT 引擎 (input={input_size}x{input_size})，请稍候...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("引擎构建失败")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT 引擎已保存至: {engine_path}")
    return engine_path


def parse_args():
    parser = argparse.ArgumentParser(description="构建 TensorRT 引擎（默认 FP32）")
    parser.add_argument("--onnx", default="checkpoints/model_160.onnx", help="ONNX 路径")
    parser.add_argument("--engine", default="checkpoints/model_fp32_160.engine", help="输出引擎路径")
    parser.add_argument("--input-size", type=int, default=160, help="输入尺寸（H=W）")
    parser.add_argument("--workspace", type=int, default=1 << 30, help="workspace 字节数")
    parser.add_argument("--fp16", action="store_true", help="启用 FP16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_engine(
        Path(args.onnx), Path(args.engine),
        input_size=args.input_size, fp16=args.fp16, workspace_size=args.workspace,
    )
