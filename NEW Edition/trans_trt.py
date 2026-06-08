# -*- coding: utf-8 -*-
import tensorrt as trt
import os

def build_engine(onnx_file_path, trt_file_path, fp16_mode=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("❌ 解析 ONNX 文件失败")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # 创建 Optimization Profile（关键步骤）
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = (1, 3, 224, 224)
    profile.set_shape(input_name, input_shape, input_shape, input_shape)  # min/opt/max 设为相同
    config.add_optimization_profile(profile)

    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("❌ 构建 TensorRT 引擎失败")

    with open(trt_file_path, 'wb') as f:
        f.write(engine.serialize())
    print("✅ 成功生成 TensorRT 引擎文件：" + trt_file_path)

if __name__ == "__main__":
    onnx_file = "resnet18_dashboard.onnx"
    trt_file = "resnet18_dashboard.trt"
    if not os.path.exists(onnx_file):
        print("❌ ONNX 文件不存在，请先执行 export_to_onnx.py")
    else:
        build_engine(onnx_file, trt_file)

