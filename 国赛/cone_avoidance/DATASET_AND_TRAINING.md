# 锥形桶数据集拍照与训练说明

## 拍照目标

第一版目标是训练一个稳定识别 PVC 圆锥的 YOLO 单类别模型，类别名固定为 `cone`。模型只负责检测锥桶 bbox，距离和避障判断交给 RGB-D 感知模块使用 RealSense D435i 的 aligned depth 完成。

建议第一轮拍摄 600-1000 张图。如果时间紧，最低也建议 300-500 张，但要保证场景变化足够多。

## 拍摄设备与视角

尽量使用比赛时实际安装在机器狗上的相机拍摄。相机高度、俯仰角、焦距、分辨率要尽量接近正式运行状态。

不要只用手机俯拍锥桶。手机照片可以作为补充，但主数据集应该来自机器人视角。

## 拍摄场景分配

推荐按下面比例拍：

```text
单锥桶: 35%
双锥桶: 45%
无锥桶负样本: 10%
干扰样本: 10%
```

无锥桶负样本很重要，用来减少误检。可以拍地面、场地边界、纸箱、检测区、抓取区、放置区、机器狗脚部入镜等。

干扰样本可以包含纸箱、黄色/橙色物体、标志贴纸、人的鞋子、场地边线等，但不要让干扰物完全盖住锥桶。

## 距离覆盖

锥桶避障最关心 0.5-2.5 m 范围，建议重点覆盖：

```text
0.4-0.7 m: 15%
0.7-1.2 m: 30%
1.2-1.8 m: 30%
1.8-2.5 m: 20%
2.5 m 以上: 5%
```

近距离图用于急停和绕行，远距离图用于提前规划。太远的小目标不必占太多。

## 画面位置覆盖

每个距离段都要覆盖锥桶在画面中的不同位置：

- 左侧边缘。
- 左前方。
- 正前方。
- 右前方。
- 右侧边缘。
- 只露出部分底座或部分侧边。

不要让所有锥桶都在画面中央，否则实机绕行时边缘检测会变差。

## 双锥桶摆法

双锥桶是比赛真实重点，建议多拍：

- 两个锥桶左右分开，中间可通行。
- 两个锥桶间距较小，中间不可通行。
- 一个近一个远。
- 一个在左前方，一个在右远方。
- 两个都偏左或都偏右。
- 一个锥桶部分遮挡另一个锥桶。

标注时两个锥桶都要分别框出来。

## 光照与运动

至少覆盖这些情况：

- 正常室内光。
- 偏暗。
- 强光或局部反光。
- 背景有阴影。
- 机器人低速移动时截图，保留少量运动模糊。

运动模糊不要占太多，建议 10%-15%。太糊的照片可以不放进训练集。

## 拍摄命名

建议按批次命名，方便回溯：

```text
cone_solo_near_left_0001.jpg
cone_pair_mid_gap_0001.jpg
cone_pair_blocked_0001.jpg
negative_field_0001.jpg
```

拍完原图先放到：

```text
cone_avoidance/datasets/cone_raw/images/
```

标注后的 YOLO 标签放到：

```text
cone_avoidance/datasets/cone_raw/labels/
```

每张图片必须对应一个同名 `.txt` 标签文件，例如：

```text
images/cone_pair_mid_gap_0001.jpg
labels/cone_pair_mid_gap_0001.txt
```

无锥桶负样本也要有 `.txt` 文件，但内容留空。

## 标注规则

使用 YOLO 检测框格式，类别只有一个：

```text
0 cone
```

框选原则：

- 框住完整可见锥桶，包括底座。
- 不要把阴影、地面、旁边无关物体框进去。
- 被遮挡时，只框可见部分。
- 画面里有两个锥桶，就标两个框。
- 很模糊但肉眼仍能确认是锥桶的，可以标；完全看不清的不要放进数据集。

## 数据集划分

标注完成后，可以只运行训练脚本并加 `--prepare`，它会自动划分数据集：

```bash
python3 scripts/train_cone_yolo.py --prepare --model yolov8n.pt --epochs 120 --imgsz 640 --batch 16
```

也可以先单独划分。在 `cone_avoidance` 目录运行：

```bash
python3 scripts/split_yolo_dataset.py --copy --clean
```

它会从：

```text
datasets/cone_raw/images
datasets/cone_raw/labels
```

复制并划分到：

```text
datasets/cone_yolo/images/train
datasets/cone_yolo/images/val
datasets/cone_yolo/images/test
datasets/cone_yolo/labels/train
datasets/cone_yolo/labels/val
datasets/cone_yolo/labels/test
```

默认比例是：

```text
train: 80%
val: 15%
test: 5%
```

## 训练

安装依赖：

```bash
pip install ultralytics
```

在 `cone_avoidance` 目录运行：

```bash
python3 scripts/train_cone_yolo.py \
  --prepare \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 640 \
  --batch 16
```

训练完成后，最佳模型会在：

```text
runs/detect/cone_yolo/weights/best.pt
```

训练脚本默认会把最佳模型复制到：

```text
models/cone_yolo_best.pt
```

RGB-D 感知负责人后续的 `cone_detector_yolo.py` 或等价节点应优先加载这个固定路径。

## Jetson 部署建议

第一版先用 `best.pt` 跑通。如果 Jetson Xavier NX 上速度不够，再导出 TensorRT：

```bash
python3 scripts/train_cone_yolo.py \
  --prepare \
  --model yolov8n.pt \
  --epochs 120 \
  --imgsz 640 \
  --batch 16 \
  --export engine
```

也可以训练完成后单独导出，后续再加专门的导出脚本。

## 第一轮验收指标

第一版不用追求特别高指标，先看实机稳定性：

- 锥桶在 0.5-2.5 m 内基本不漏检。
- 双锥桶能分开检测。
- 场地纸箱、边线、鞋子等不明显误检为锥桶。
- 推理速度能满足低速避障。

如果第一版漏检集中在某种情况，就补拍那种情况，而不是盲目增加同质照片。
