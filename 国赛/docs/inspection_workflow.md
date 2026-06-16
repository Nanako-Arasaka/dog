# 巡检识别真实数据闭环

本文档只覆盖巡检识别：区域字母 A/B/C/D、仪表盘 bbox、仪表状态 low/normal/high。不要在这条流程里接入机械臂、抓取、搬运、避障、导航或语音播报。

## 1. 放置原始照片

拍完照片后，把原图放到：

```text
data/raw_photos/
```

建议保留原图，不要直接改名覆盖。可以按日期或场地分子目录：

```text
data/raw_photos/2026-06-10_site1/
data/raw_photos/2026-06-10_site2/
```

照片应尽量覆盖不同光照、倾斜、距离、反光、A/B/C/D 四个区域和 low/normal/high 三类仪表状态。

## 2. 整理 YOLO 数据集

在 `Thedog/国赛` 目录运行：

```powershell
python .\tools\dataset_builder.py --raw-dir .\data\raw_photos --out-dir .\data\inspection_yolo --workers 8
```

输出结构：

```text
data/inspection_yolo/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  dataset.yaml
  stats.csv
  stats.json
```

`dataset.yaml` 默认类别：

```text
0 zone_A
1 zone_B
2 zone_C
3 zone_D
4 gauge
```

## 3. 手工标注 letter/gauge

用 LabelImg、Roboflow、CVAT、Label Studio 等任一支持 YOLO 格式的标注工具打开：

```text
data/inspection_yolo/images/
```

标注规则：

- `zone_A`、`zone_B`、`zone_C`、`zone_D`：只框区域字母本体，框尽量贴近字母外接矩形。
- `gauge`：框完整仪表盘圆形区域，包含刻度和指针。
- 不要把整个区域牌、背景板、机械结构框进 letter。
- 每张图如果看不清字母或仪表，宁可不标或移到待定目录，不要制造错误标签。

保存标签到对应路径：

```text
data/inspection_yolo/labels/train/*.txt
data/inspection_yolo/labels/val/*.txt
data/inspection_yolo/labels/test/*.txt
```

## 4. 检查标签

逐个 split 检查：

```powershell
python .\tools\label_check.py --dataset-root .\data\inspection_yolo --split train --debug-dir .\output\debug_labels\train --workers 8
python .\tools\label_check.py --dataset-root .\data\inspection_yolo --split val --debug-dir .\output\debug_labels\val --workers 8
python .\tools\label_check.py --dataset-root .\data\inspection_yolo --split test --debug-dir .\output\debug_labels\test --workers 8
```

重点看：

- `ok: true` 表示没有格式错误或 bbox 越界错误。
- `issues` 里如果出现 `missing label file`、`invalid YOLO line`、`bbox extends outside image`，先回标注软件修。
- `output/debug_labels/` 里的图片会把框画回原图，用来快速检查类别和框的位置是否正确。

## 5. 训练 YOLO

安装并训练：

```powershell
pip install ultralytics
yolo detect train model=yolov8n.pt data=.\data\inspection_yolo\dataset.yaml imgsz=640 epochs=80 batch=16 workers=4
```

训练完成后，常用权重位置是：

```text
runs/detect/train/weights/best.pt
```

如果 Jetson Xavier NX 显存吃紧，先把 `batch` 调小，例如 `batch=4` 或 `batch=2`。

## 6. 运行 YOLO 定位

单张图：

```powershell
python .\tools\yolo_locator.py --model .\runs\detect\train\weights\best.pt --source .\data\inspection_yolo\images\test\sample.jpg --debug-dir .\output\debug_yolo --output-json .\output\yolo_single.json
```

文件夹：

```powershell
python .\tools\yolo_locator.py --model .\runs\detect\train\weights\best.pt --source .\data\inspection_yolo\images\test --debug-dir .\output\debug_yolo --output-json .\output\yolo_batch.json --workers 1
```

输出中：

- `object_type: zone_letter` 表示 A/B/C/D 字母定位。
- `object_type: gauge` 表示仪表盘定位。
- `bbox` 是像素坐标 `{x1,y1,x2,y2}`。

## 7. 读取仪表盘状态

从 YOLO 输出里复制某个 gauge 的 bbox，运行：

```powershell
python .\tools\gauge_reader.py --image .\data\inspection_yolo\images\test\sample.jpg --bbox 100,80,260,240 --zone A --debug-dir .\output\debug_gauge_roi
```

输出字段：

- `gauge_status`: `low` / `normal` / `high`
- `abnormal`: `true` 表示 low/high，`false` 表示 normal
- `speak_key`: 例如 `A_low`
- `text`: 中文结果，例如 `A区域仪表盘显示偏低，状态异常`
- `angle`: 检测到的指针角度
- `bbox`: 读取使用的仪表盘 ROI

## 8. 一键 demo 闭环

有 `best.pt` 后，可以直接跑组合 demo：

```powershell
python .\tools\inspection_pipeline_demo.py --source .\data\inspection_yolo\images\test\sample.jpg --model .\runs\detect\train\weights\best.pt --output-json .\output\inspection_single.json --debug-dir .\output\debug_inspection_pipeline
python .\tools\inspection_pipeline_demo.py --source .\data\inspection_yolo\images\test --model .\runs\detect\train\weights\best.pt --output-json .\output\inspection_batch.json --debug-dir .\output\debug_inspection_pipeline
```

demo 会做：

```text
image -> yolo_locator -> gauge bbox -> gauge_reader -> JSON/text/debug image
```

输出中看：

- `ok: true`：本次闭环执行完成。
- `items[].detections`：YOLO 原始定位结果。
- `items[].readings`：每个仪表盘的状态读取结果。
- `items[].texts`：可以直接展示或交给上层播报模块的中文文本。
- `items[].debug_image`：融合标注图路径。

如果还没有训练模型，会返回：

```json
{
  "ok": false,
  "error": "model_not_found"
}
```

这表示需要先完成标注和 YOLO 训练，不是程序崩溃。
