# 四足机器狗仪表盘识别系统

2026 年中国高校智能机器人创意大赛（四足大型组）预选赛项目。

## 架构

```
摄像头 → 本地推理（TRT 或 PyTorch）→ 本地可视化
```

## 目录结构

```
yuxuansai/
├── start_dog.py          # 本地推理入口（TRT / PyTorch）
├── start_jetson.py       # 兼容入口（同 start_dog.py）
├── start_dog.sh          # 狗端启动脚本
├── start_jetson.sh       # Jetson端启动脚本
├── detect_dashboard_trt.py  # TensorRT 推理脚本（最终版）
├── Dashboard_detec2t.py     # PyTorch 推理脚本（最终版）
├── perception/              # 保留旧版感知包（已停用）
├── scripts/              # 工具脚本
│   ├── build_trt.py      # 构建 TensorRT 引擎
│   ├── export_onnx.py    # 导出 ONNX 模型
│   └── video_extractor.py # 从视频抽帧
├── config/               # 配置文件
├── checkpoints/          # 模型权重 / TensorRT 引擎
├── data/                 # 训练/测试数据
└── requirements.txt
```

## 快速启动（本地推理）

**TensorRT：**
```bash
python3 start_dog.py --mode trt --engine-path ./checkpoints/resnet18_dashboard.trt
```

**PyTorch：**
```bash
python3 start_dog.py --mode torch --model-path ./checkpoints/model_best.pth
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--confidence-threshold` | 0.5 | 单帧置信度阈值 |
| `--hough-interval` | 3 | 圆检测间隔帧数 |
| `--cls-confirm-window` | 2 | 分类确认窗口 |
| `--switch-confirm-frames` | 2 | 输出切换确认帧数 |
| `--input-size` | 160 | 推理输入尺寸 |
| `--mode` | trt | 推理后端（trt/torch） |

## 说明

- **标签映射顺序固定为**：`down` → `normal` → `over`。训练集文件夹建议使用 `down/`、`normal/`、`over/`。
- TensorRT 推理会**自动读取引擎输入尺寸**进行预处理，不再硬编码 224 或 160。
