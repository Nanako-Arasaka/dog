# 四足机器狗仪表盘识别系统

2026 年中国高校智能机器人创意大赛（四足大型组）预选赛项目。

## 架构

```
摄像头(狗端 start_dog.py) → UDP JPEG帧 → Jetson推理(start_jetson.py) → UDP JSON结果 → 狗端可视化
```

## 目录结构

```
yuxuansai/
├── start_dog.py          # 狗端入口：采集 + 推流 + 显示结果
├── start_jetson.py       # Jetson端入口：接收帧 → 推理 → 回传结果
├── start_dog.sh          # 狗端启动脚本
├── start_jetson.sh       # Jetson端启动脚本
├── perception/           # 感知核心包
│   ├── cropper.py        # HoughCircleCropper 圆检测裁剪
│   ├── detector.py       # DashboardCameraDetector 推理分类
│   ├── model.py          # ResNet18/34 分类模型
│   ├── inference.py      # TensorRT 推理封装
│   └── visualize.py      # 可视化 / 中文渲染 / 防抖
├── scripts/              # 工具脚本
│   ├── build_trt.py      # 构建 TensorRT 引擎
│   ├── export_onnx.py    # 导出 ONNX 模型
│   └── video_extractor.py # 从视频抽帧
├── config/               # 配置文件
├── checkpoints/          # 模型权重 / TensorRT 引擎
├── data/                 # 训练/测试数据
└── requirements.txt
```

## 快速启动

**Jetson 端：**
```bash
python3 start_jetson.py --target-ip <狗端IP> --target-port 5005 --engine-path ./checkpoints/model_fp16_160.engine
```

**狗端：**
```bash
python3 start_dog.py --jetson-ip <Jetson IP> --jetson-frame-port 6006 --listen-port 5005
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--confidence-threshold` | 0.5 | 单帧置信度阈值 |
| `--hough-interval` | 3 | 圆检测间隔帧数 |
| `--cls-confirm-window` | 2 | 分类确认窗口 |
| `--switch-confirm-frames` | 2 | 输出切换确认帧数 |
| `--input-size` | 160 | 推理输入尺寸 |
| `--send-hz` | 15 | UDP 发送频率 |
