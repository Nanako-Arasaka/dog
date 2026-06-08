import torch
from dashboard_model import Resnet18_dashboard  # 引入你定义的模型结构
import os

# 初始化模型
model = Resnet18_dashboard(num_classes=3)
model.load_state_dict(torch.load("model_best.pth", map_location=torch.device("cpu")))
model.eval()

# 创建一个假的输入张量，大小必须匹配模型预期输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出ONNX文件
torch.onnx.export(model,
                  dummy_input,
                  "resnet18_dashboard.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                  opset_version=11)

print("✅ 成功导出 resnet18_dashboard.onnx")

