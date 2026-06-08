# 四足机器狗仪表盘识别系统

2026 年中国高校智能机器人创意大赛（四足大型组）预选赛项目。

## 架构

```
摄像头 → 本地推理（PyTorch 或 TRT）→ 本地可视化
```

## 目录结构

```
yuxuansai/
├── start_dog.py          # 本地推理入口（PyTorch / TRT）
├── start_jetson.py       # 兼容入口（同 start_dog.py）
├── detect_dashboard_trt.py  # TensorRT 推理脚本（最终版）
├── Dashboard_detec2t.py     # PyTorch 推理脚本（最终版）
└── checkpoints/          # 模型权重 / TensorRT 引擎
```

## 快速启动（本地推理）

**PyTorch（默认本地运行路径）：**
```bash
python3 start_dog.py
```

或显式指定：
```bash
python3 start_dog.py --mode torch --model-path ./checkpoints/model_best.pth
```

**TensorRT：**
```bash
python3 start_dog.py --mode trt --engine-path ./checkpoints/resnet18_dashboard.trt
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | torch | 推理后端（torch/trt） |
| `--camera-index` | torch: 3 / trt: 2 | 摄像头索引 |
| `--camera-device` | torch: 空 / trt: `/dev/video2` | 摄像头设备路径 |
| `--width` | 640 | 摄像头宽度 |
| `--height` | 480 | 摄像头高度 |

## 说明

- **标签映射顺序固定为**：`down` → `normal` → `over`。训练集文件夹建议使用 `down/`、`normal/`、`over/`。
- TensorRT 推理按最终示例版本使用 224 输入尺寸。
