# Project: 四足机器狗仪表盘识别系统

## 项目概述

2026年中国高校智能机器人创意大赛（四足大型组）预选赛项目。机器狗通过摄像头采集仪表盘图像，通过UDP发送到Jetson设备进行TensorRT推理，识别仪表盘指示灯状态（High/Normal/Low），再将结果回传机器狗端可视化。

## 架构

```
摄像头(狗端 control_receiver.py) → UDP JPEG帧 → Jetson推理(jetson_inference_server.py) → UDP JSON结果 → 狗端可视化
```

核心文件：
- `yuxuansai/perception_sender.py` — 核心：HoughCircleCropper（圆检测）、DashboardCameraDetector（推理+分类）、PIL中文渲染
- `yuxuansai/jetson_inference_server.py` — Jetson端推理服务
- `yuxuansai/control_receiver.py` — 机器狗端采集+显示
- `yuxuansai/inference_trt.py` — TensorRT推理封装
- `yuxuansai/dashboard_model.py` — ResNet18/34 分类模型定义
- `yuxuansai/dashboard_train (1).py` — 训练脚本

## 2026-05-11 修改记录：霍夫圆检测 & 分类稳定性优化

### 问题诊断

用户反馈：实际跑动时圆检测有偏差、过于灵敏（无圆时也误检出圆）、分类结果乱跳。

### HoughCircleCropper 修改（perception_sender.py:66-195）

| 参数 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `canny_high` (param1) | 80 + 自动降到40 | **100**，不自动降 | 低阈值导致无圆时也检出伪圆 |
| `param2` 搜索范围 | 85,75,65,55,45,35,28,22 | **80,70,60,55,50** | 22极松，任何弧线都被当圆 |
| `dp` | 1.0 → 1.2 回落 | **仅 1.0** | dp=1.2 累加器精度下降，圆心偏移 |
| `detect_interval` | 2 | **5** | 检测太频繁导致位置跳动 |
| `miss_hold_frames` | 3 | **15** | 短暂遮挡不应立即丢弃 |
| `crop_expand_ratio` | 1.18 | **1.25** | 半径检测偏小时留更多余量 |
| `min_radius` | max(4, 短边*1.5%) | **max(10, 短边*4%)** | 过滤小噪声圆 |
| `min_dist` | max(12, 短边*12%) | **max(20, 短边*15%)** | 避免同一圆重复检出 |

### 圆形确认机制（新增）

`HoughCircleCropper` 新增 `_recent_hits` 滑窗（默认3次检测中至少2次命中才接受圆），防止单帧噪声导致的偶然误检。

### 分类结果时间平滑（新增 ResultStabilizer 类）

`perception_sender.py:196-230` — 新增 `ResultStabilizer` 类：
- 维护最近 N 帧（默认10帧）的分类历史
- 输出前检查窗口内是否有类别获得 ≥60% 票数
- 不达标的帧输出 `unknown`
- 没检测到圆时清空历史，重新确认

### CLI 参数变更

| 参数 | 新默认值 | 说明 |
|------|----------|------|
| `--hough-interval` | 5 | 霍夫圆检测间隔帧数 |
| `--hough-canny-high` | 100 | Canny上阈值，越高越严格 |
| `--confidence-threshold` | 0.65 | 单帧置信度阈值 |
| `--stabilizer-window` | 10 | 分类平滑窗口大小 |
| `--hough-debug-viz` | (flag) | 新增：绘制检测圆和裁剪框 |

两个脚本均需同步更新：
- `jetson_inference_server.py` — 推理端所有新参数
- `perception_sender.py` — 本地直连模式所有新参数

### 一致性注意事项（重要）

- 训练 `--input-size`、导出 ONNX、Jetson 启动 `--input-size` 三者必须一致
- 训练 `class_order` 与推理 `--class-names` 必须一致
- 先 FP32 保精度，再切 FP16 做性能优化
- 调节顺序：先调 `--hough-canny-high` 确保圆检测正确 → 再调 `--stabilizer-window` 和 `--confidence-threshold` 确保分类稳定

## 2026-05-12 修改记录：Codex 版本评估 & 参数修复

### Codex 改动回顾（用户要求的实用化调整）

用户认为 05-11 版本过于保守，要求 Codex 调成更实用的配置。Codex 在 `perception_sender.py` 和 `jetson_inference_server.py` 中做了以下改动：

| 参数 | 05-11 版本 | Codex 版本 | 评价 |
|------|-----------|-----------|------|
| `confidence_threshold` | 0.65 | 0.45 | 合理，实际场景置信度不高 |
| `hough_interval` | 5 | 1 | 合理，跑动需每帧检测 |
| `hough_canny_high` | 100 | 80 | 合理，100 过于严格 |
| `stabilizer_window` | 10 | 4 | 偏短但可接受 |
| `min_radius` | max(10, 短边*4%) | max(4, 短边*1.5%) | 合理，支持远处小圆 |
| 取消"没圆清空历史" | 有 | 移除 | 合理，跑动时短暂遮挡不应重置 |
| `confirm_window` | 3 | **2** | 太弱，基本等于没确认 |
| `confirm_min_hits` | 2 | **1** | 1/2 任意单次误检就通过 |
| `param2` 下限 | 50 | **32** | 过于宽松，容易误检 |
| `stabilizer_min_ratio` | 0.6 (CLI不可调) | 0.5 (CLI不可调) | 需暴露到 CLI |

### 本次修复（2026-05-12）

修复了 Codex 版本的三个问题，所有修改在副本 `yuxuansai/jetson_deploy/` 中（原文件不动）：

1. **HoughCircleCropper 确认机制恢复**
   - `confirm_window`: 2 → 3（需 3 次检测确认）
   - `confirm_min_hits`: 1 → 2（3 次至少 2 次命中才接受圆）
   - `param2` 搜索范围: `(70, 60, 50, 40, 32)` → `(70, 60, 50, 40)`（去掉 32，下限提到 40）

2. **新增 CLI 参数（两个脚本同步）**
   - `--stabilizer-ratio`（默认 0.5）：分类平滑一致性比例，窗口内多数类占比阈值
   - `--hough-confirm-window`（默认 3）：霍夫圆确认滑窗大小
   - `--hough-confirm-min-hits`（默认 2）：霍夫圆确认最少命中次数

3. **DashboardCameraDetector 透传 `stabilizer_min_ratio`**
   - 之前硬编码 0.5，现在从 CLI `--stabilizer-ratio` 传入

### 部署说明

- Jetson 部署文件在 `yuxuansai/jetson_deploy/` 目录（已于 2026-05-14 弃用删除，主线回退到精简版 `perception_sender.py`）
- 原点文件 `yuxuansai/perception_sender.py` 和 `yuxuansai/jetson_inference_server.py` 保持不变

## 2026-05-12 修改记录（2）：针对“效果更差”的再调优

### 问题复盘

在 `jetson_deploy/` 副本中，虽然恢复了 `confirm_window=3 / confirm_min_hits=2` 与 `param2` 下限 40，但默认参数组合仍偏“松”，导致：

- 误检圆进入裁剪流程（尤其纹理复杂背景）
- 低置信度类别过早进入稳定器并被输出
- 结果呈现“看起来能检出，但总体更飘”

### 本次调优（已修改）

涉及文件：
- `yuxuansai/jetson_deploy/perception_sender.py`
- `yuxuansai/jetson_deploy/jetson_inference_server.py`

| 参数 | 调整前 | 调整后 | 目的 |
|------|--------|--------|------|
| `confidence_threshold` | 0.45 | **0.50** | 过滤边界低质帧，减少错误标签进入稳定器 |
| `hough_canny_high` | 80 | **90** | 提高边缘判定质量，抑制噪声圆 |
| `stabilizer_window` | 4 | **5** | 增强时序抗抖，避免 1-2 帧误判主导输出 |
| `stabilizer_ratio` | 0.50 | **0.55** | 降低“过半即通过”导致的错误稳定输出 |
| `min_radius` | max(4, 短边*1.5%) | **max(6, 短边*2%)** | 过滤过小噪声圆，同时保留中远距离目标 |
| `param2` 搜索范围 | 70,60,50,40 | **70,60,50,45,40** | 在 50→40 之间加过渡点，提升稳定性 |

### 当前推荐默认组合

- `--confidence-threshold 0.5`
- `--hough-canny-high 90`
- `--hough-interval 1`
- `--hough-confirm-window 3`
- `--hough-confirm-min-hits 2`
- `--stabilizer-window 5`
- `--stabilizer-ratio 0.55`

### 调参优先级（现场）

1. 先调 `--hough-canny-high`（80~100）确保裁剪圆稳定  
2. 再调 `--confidence-threshold`（0.45~0.55）控制误分类  
3. 最后调 `--stabilizer-window` / `--stabilizer-ratio` 控制输出抖动  

## 2026-05-13 修改记录：远处圆检测优化

### 问题

近处圆识别准确率已达标，但远处圆检测不到。

### 修改（perception_sender.py HoughCircleCropper）

| 参数 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| `min_radius` | `max(4, 短边*1.5%)` | `max(4, 短边*1%)` | 远处圆半径小（480p约5px），1.5%截掉了 |
| `param2` 搜索范围 | `70,60,50,40,32` | `70,60,50,40,35,28` | 远处边缘弱，32太严；加35过渡不过激 |
| `confirm_window` 默认值 | 2 | **3** | 检测变敏感后需更强确认门槛 |
| `confirm_min_hits` 默认值 | 1 | **2** | 3次至少2次命中才接受，过滤单帧噪声 |

### 设计思路

用更严的确认门槛对冲更敏感的检测参数，让远处弱圆能进来，但单帧噪声进不了推理管线。

### 后续调整（同日）

- `param2` 下限降到 28 后识别乱跳，回退到 32——28 区间历史验证误检重灾区
- `GaussianBlur(5,5)` → `(3,3)`：5x5 模糊对远处小圆太重，直接抹平边缘导致检不出；3x3 保留更多边缘信息，让 `param2=32` 也能检出远处弱圆

## 2026-05-13/14 修改记录：代码大幅简化 & 中文输出

### 背景

经过多轮调参后发现：复杂的自适应参数、确认机制、时序平滑在跑动场景下稳定性收益有限，反而增加了调试难度。决定回退到精简版本，聚焦核心链路正确性，后期按需加回。

### perception_sender.py 简化（2026-05-13）

**HoughCircleCropper** 大幅精简：

| 移除项 | 原因 |
|--------|------|
| CLAHE 预处理 | 增强效果有限，且增加每帧开销 |
| crop_expand_ratio | 精确按圆半径裁剪即可 |
| confirm_window / confirm_min_hits / _recent_hits | 确认机制移到分类层统一处理 |
| 自适应 min_radius / max_radius / min_dist | 固定值更可预测，远处圆通过降低 param2 下限处理 |
| param1 (canny_high) | 不设即用 OpenCV 默认 100 |

**当前 HoughCircleCropper 参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `dp` | 0.8 | 累加器分辨率（<1.0 可检测更模糊的圆） |
| `minDist` | 50 | 圆心最小间距（固定值，简单可控） |
| `minRadius` | 5 | 最小半径（覆盖远处小圆） |
| `maxRadius` | 300 | 最大半径（覆盖近处大圆） |
| `param2` | 100, 85, 70 | 累加器阈值分级搜索（从严格到宽松） |
| `GaussianBlur` | (5,5) | 高斯降噪 |
| `detect_interval` | 3 | 每 3 帧检测一次圆 |
| `miss_sleep_ms` | 1.0 | 未命中时休眠降低 CPU |

**DashboardCameraDetector：**
- 移除 ResultStabilizer 时序平滑——直接输出单帧分类结果
- 移除 hough_canny_high、stabilizer_window、stabilizer_min_ratio 等参数
- 分类逻辑：置信度 >= confidence_threshold(0.5) 输出类别，否则 unknown
- CLI 参数精简为：`--hough-interval`、`--hough-sleep-ms`、`--confidence-threshold`

### 中文输出改造（2026-05-14）

**问题：** Linux 下 OpenCV 不支持中文字体，`cv2.putText` 显示中文为问号。

**方案：** 用 PIL/Pillow 的 ImageDraw 替代 cv2.putText 渲染所有文字。

**perception_sender.py 改动：**

| 位置 | 内容 |
|------|------|
| 新增 `CLASS_LABEL_ZH` | `{"high":"偏高","normal":"正常","low":"偏低","unknown":"未知"}` |
| 新增 `FONT_CANDIDATES` | 覆盖 Linux（文泉驿/Noto/AR PL）+ Windows（微软雅黑/黑体）路径 |
| 新增 `get_font()` | 字体缓存 + 候选自动查找 + 环境变量 `DASHBOARD_FONT_PATH` + PIL 默认兜底 |
| 新增 `draw_text_lines_pil()` | BGR→RGB→PIL→画字→RGB→BGR 全帧转换 |
| 重写 `draw_result()` | 所有文字渲染走 PIL，不再调用 cv2.putText |
| 新增 `--font-path` CLI | 手动指定中文字体路径 |
| UDP payload | `class` 字段改为中文，同时保留 `class_en` 英文 |

**control_receiver.py 改动：**

| 位置 | 内容 |
|------|------|
| 新增 `CLASS_ZH_TO_EN` / `CLASS_EN_TO_ZH` | 中英文双向映射 |
| 新增 `normalize_class()` | 归一化函数：无论收到中文还是英文都转为英文 key |
| `draw_overlay` | 优先读 `class_en`，fallback 到 `class`；显示用中文 |
| 日志输出 | `RESULT=` 显示中文类别名 |

### 技术报告生成（2026-05-14）

- 基于模板 `.docx` 直接在副本上编辑正文内容，完整保留封面、字体、颜色、页码格式
- 修复模板本身的空白第二页问题（删除多余空段落）
- 修复封面页码问题（Section 0 添加 titlePg）
- 报告内容描述独立本地运行架构（不提及 Jetson/TensorRT/UDP 外部通信，符合比赛规则）
- 输出：`2025_DOG_Dashboard_Technical_Report_FINAL.docx/.pdf`

### 自定义 Skills 安装（2026-05-14）

GitHub 不可达，手动创建 3 个 SKILL.md：
- `python-dev`：OpenCV、PyTorch、CLI 设计知识
- `doc-automation`：python-docx/docx2pdf 操作技巧
- `code-review`：CV/DL 项目代码审查清单

### 待处理

- [x] 分类确认机制：已通过 `SwitchConfirm` 类实现（`jetson_inference_server.py`），独立模式（`standalone_detector.py`）无确认

## 2026-05-14/15 修改记录：文件拆分 & 插件安装 & 确认机制收尾

### 文件架构最终形态

经过多轮迭代后，最终形成三文件架构：

| 文件 | 模式 | 用途 | 确认机制 |
|------|------|------|:---:|
| `standalone_detector.py` | **独立** | 摄像头直连 → 推理 → 直接可视化（无 UDP） | 无 |
| `perception_sender.py` | **分布式发送** | 摄像头 → 推理 → UDP 发送结果给狗端 | 无（接收端处理） |
| `jetson_inference_server.py` | **分布式接收** | 接收 UDP 帧 → 推理 → `SwitchConfirm` → JSON 回传 | SwitchConfirm |
| `control_receiver.py` | **狗端** | 采集 → UDP 发帧 → 接收 JSON → 可视化 | 无 |

**设计原则：**
- `SwitchConfirm` 作为唯一确认层，仅存在于 `jetson_inference_server.py`（分布式接收端）
- `DashboardCameraDetector.predict()` 保持原始单帧输出，不内置确认
- 独立模式不做确认，行为与旧近处版一致

### SwitchConfirm 机制（jetson_inference_server.py 第 42-85 行）

Codex 实现的轻量确认类，工作方式：
1. 维护 `stable_class`（当前稳定输出）和 `candidate_class`（候选类别）
2. 新类别出现时设为候选，计数 +1
3. 候选累计 `confirm_frames` 次后晋升为 stable
4. 稳定类别匹配到时重置候选计数器
5. 非匹配非候选时重置候选为新类别

CLI 参数：`--cls-confirm-window`（默认 2，即 2 帧确认）

⚠️ **注意：不要在此文件内做第二次确认。`predict()` 已经返回原始值，`SwitchConfirm` 是唯一确认层。**

### 确认机制踩坑记录

**问题 1：最初在 `DashboardCameraDetector.predict()` 内加 `_cls_history`，但改了 `detected` 的判断逻辑。**
- 后果：`detected` 从 `confidence >= threshold` 变成了 `self._stable_class != "unknown"`，前端显示异常
- 教训：**绝对不要更改 `detected` 的判断逻辑**，确认只控 `class_name`

**问题 2：`predict()` 内部确认 + `SwitchConfirm` 外部确认形成两层防抖。**
- 后果：两层叠加等于 N×2 帧延迟
- 解决：撤销 `predict()` 内的确认，保留 `SwitchConfirm` 作为唯一确认层

### 启动脚本（Linux）

三个一键启动脚本，直接复制粘贴到终端：

| 脚本 | 用途 |
|------|------|
| `start_dog.sh` | 狗端启动（control_receiver.py） |
| `start_jetson.sh` | Jetson 端启动（jetson_inference_server.py），需修改 `192.168.31.XXX` 为狗端 IP |
| `start_standalone.sh` | 独立本地模式（standalone_detector.py），无网络依赖 |

使用方法：`chmod +x start_*.sh && ./start_dog.sh`

### 插件/Marketplace 安装记录

**问题：** GitHub SSH 认证失败，`claude plugin install` 走 SSH 协议。

**解决：** 强制 Git 使用 HTTPS：
```bash
git config --global url."https://github.com/".insteadOf git@github.com:
```

**netresearch 插件已安装（4 个）：**
- `git-workflow` — Git 工作流
- `peer-qa-review` — 代码 QA 审查
- `markdown-to-pdf` — MD 转 PDF
- `data-tools` — 数据处理
- `docker-development` — Docker 开发

**agents-skills-plugins 插件已安装（4 个）：**
- `python-development` — Python 脚手架 + Agent（FastAPI/Django/Python Pro）
- `anthropic-code-review` — 代码审查
- `code-documentation` — 代码文档（Agent: code-reviewer, docs-architect, tutorial-engineer）
- `documentation-generation` — 文档生成（Agent: api-documenter, mermaid-expert, tutorial-engineer）

**对本项目有用的命令：**
- `/anthropic-code-review:code-review` — 审查代码
- `/code-documentation:code-explain` — 解释逻辑
- `python-pro` Agent — 写/改 Python CV 代码

**注意：** 之前手动创建的 `python-dev`、`doc-automation`、`code-review` 三个自定义 skill（`~/.claude/skills/`）已被 marketplace 插件覆盖，仅保留 `code-simplify`、`documentation-writer`、`find-skills`。

### 当前近处版参数（不可随意改动）

HoughCircleCropper：
- `dp=0.8`, `minDist=50`, `minRadius=5`, `maxRadius=300`
- `param2=(100, 85, 70)`, `GaussianBlur=(5,5)`, `detect_interval=3`

DashboardCameraDetector：
- `confidence_threshold=0.5`, `input_size=160`
- `preprocess_mode=resize_center_crop`
- `class_names=["high", "normal", "low"]`

这些参数针对近处识别优化，远处识别需要另外一套参数组合（见 05-13 记录）。
