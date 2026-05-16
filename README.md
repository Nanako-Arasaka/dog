# 四足机器狗 · 中国高校智能机器人创意大赛

2026 年中国高校智能机器人创意大赛（四足大型组）参赛项目，分两阶段：预选赛（仪表盘识别）和国赛（三阶段任务）。

---

## 项目概览

| 赛段 | 目录 | 任务 | 状态 |
|------|------|------|:----:|
| 预选赛 | `yuxuansai_new/` | 仪表盘指示灯识别（High/Normal/Low） | 完成 |
| 国赛 | `国赛/` | 避障 → 巡检识别 + 语音 → 抓取投放 | 框架 |

---

## 一、预选赛 — 仪表盘识别系统

### 架构

```
摄像头(狗端) → UDP JPEG帧 → Jetson推理(TensorRT) → UDP JSON结果 → 狗端可视化
```

- **狗端** (`start_dog.py`)：采集摄像头画面 → UDP 发送 JPEG 帧 → 接收推理结果 → OpenCV 渲染
- **Jetson 端** (`start_jetson.py`)：接收视频帧 → HoughCircleCropper 裁剪仪表盘 → TensorRT 推理 → 回传 JSON
- 支持独立本地模式：直接连摄像头推理，无需网络（见 `start_dog.py` 内 `standalone` 模式）

### 目录结构

```
yuxuansai_new/
├── start_dog.py              # 狗端入口：采集 + 推流 + 结果可视化
├── start_jetson.py           # Jetson 端入口：收帧 → 推理 → 回传
├── start_dog.sh / start_jetson.sh   # Linux 一键启动脚本
├── perception/               # 感知核心
│   ├── cropper.py            # HoughCircleCropper 霍夫圆检测 + 裁剪
│   ├── detector.py           # DashboardCameraDetector 推理 + 分类
│   ├── model.py              # ResNet18 / ResNet34 分类模型
│   ├── inference.py          # TensorRT 推理封装
│   └── visualize.py          # PIL 中文渲染 / SwitchConfirm 防抖
├── scripts/
│   ├── build_trt.py          # 构建 TensorRT 引擎
│   ├── export_onnx.py        # 导出 ONNX 模型
│   └── video_extractor.py    # 视频抽帧工具
├── checkpoints/              # 模型权重 / TensorRT 引擎（需自行放置）
├── data/                     # 训练/测试数据（需自行放置）
└── requirements.txt
```

### 快速启动

**Jetson 端：**

```bash
python3 start_jetson.py \
  --target-ip <狗端IP> --target-port 5005 \
  --engine-path ./checkpoints/model_fp16_160.engine \
  --input-size 160
```

**狗端：**

```bash
python3 start_dog.py \
  --jetson-ip <Jetson IP> --jetson-frame-port 6006 \
  --listen-port 5005
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--confidence-threshold` | 0.5 | 单帧置信度阈值 |
| `--hough-interval` | 3 | 圆检测间隔帧数 |
| `--cls-confirm-window` | 2 | 分类确认窗口（连续 N 帧一致才切换） |
| `--input-size` | 160 | 推理输入尺寸（需与训练/ONNX 一致） |
| `--send-hz` | 15 | UDP 发送频率 |

### 结果显示

| 类别 | 颜色 | 含义 |
|:----:|:----:|------|
| High | 红色 | 仪表读数偏高 |
| Normal | 绿色 | 正常 |
| Low | 黄色 | 仪表读数偏低 |
| Unknown | 灰色 | 未检测到 / 置信度不足 |

### 注意事项

- **训练 `--input-size` / 导出 ONNX / Jetson 启动 `--input-size` 三者必须一致**
- **训练 `class_order` 与推理 `--class-names` 必须一致（high, normal, low）**
- 先 FP32 保精度，再切 FP16 做性能优化
- 调参顺序：先 `--hough-canny-high` 确保圆检测正确 → 再调 `--confidence-threshold` 控制误分类 → 最后调 `--stabilizer-window` 控制输出抖动

---

## 二、国赛 — 三阶段任务框架

> 国赛部分为代码骨架，现场联调时需替换感知接口。

### 三阶段流程

1. **避障通过** — 持续前进直到感知侧报告"已通过障碍区"
2. **巡检识别 + 语音播报** — 读取仪表状态，播报偏高/偏低/正常
3. **抓取投放** — 根据巡检异常区域执行红色长条投放，掉落累计 3 次判失败

详见 [国赛/README.md](国赛/README.md)

---

## 仓库结构

```
.
├── README.md                  # 本文件
├── yuxuansai_new/             # 预选赛：仪表盘识别系统
├── 国赛/                       # 国赛：三阶段任务框架
└── .gitignore
```
