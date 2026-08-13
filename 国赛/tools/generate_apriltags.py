#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 AprilTag tag36h11 打印版图片。

用 OpenCV 内置的 DICT_APRILTAG_36h11 字典生成（编码与官方 apriltag 库 tag36h11
完全一致），自动加检测所需的静区（quiet zone），并拼成 A4 PDF 方便打印。

用法：
  python3 tools/generate_apriltags.py                 # 生成全部 10 个 tag
  python3 tools/generate_apriltags.py --ids 1,2,3     # 只生成部分
  python3 tools/generate_apriltags.py --size-cm 18    # 指定 tag 黑框边长（cm）

输出（output/apriltags/）：
  tag36_11_00001.png ... tag36_11_00010.png   每 tag 独立高清 PNG（含静区 + 标签）
  tag36h11_all_A4.pdf                         10 个 tag 拼一页 A4 的核对版
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output" / "apriltags"

# tag 名称（对应 config/tags.yaml）
TAG_NAMES = {
    1: "点一 start_exit",
    2: "点二 start_mid",
    3: "点三 start_far",
    4: "点四 仪表箱1侧2",
    5: "点五 仪表箱1侧1",
    6: "点六 障碍入口",
    7: "点七 障碍出口",
    8: "点八 仪表箱2侧",
    9: "点九 抓取接近",
    10: "点十 放置接近",
}

DICT = cv2.aruco.DICT_APRILTAG_36h11
TAG_FAMILY = "tag36h11"


def load_cjk_font(size: int):
    """尝试加载系统中文字体，失败返回默认字体。"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def generate_marker(tag_id: int, side_pixels: int) -> np.ndarray:
    """生成单个 tag 的灰度图（0/255 uint8），标准 AprilTag 极性（白底黑框、黑格=数据1）。

    ⚠️ 关键：OpenCV `generateImageMarker` 对 DICT_APRILTAG_36h11 输出的是
    "黑底白框、内部白格=数据1"（OpenCV 极性）。官方 apriltag 库（现场主后端）
    期望的是标准 AprilTag："白底黑框、内部黑格=数据1"。直接打印 OpenCV 输出
    会导致官方 apriltag 库检测不到。

    修复：只反色内部 6x6 数据格，保留外圈 1 cell 黑边，得到标准极性图。
    OpenCV 自带 detectMarkers 对标准极性返回 None（极性反），反过来证明是标准布局。
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side_pixels, borderBits=1)
    cell = side_pixels // 8
    inner = slice(cell, -cell)
    marker[inner, inner] = 255 - marker[inner, inner]
    return marker


def self_verify(marker: np.ndarray, tag_id: int) -> bool:
    """布局自检：标准 AprilTag 应为「白底黑框、内部有黑有白」。

    OpenCV detectMarkers 对标准极性返回 None（极性反），不能用 detect 验证；
    改为检查 4 角黑 + 内部有黑有白。
    """
    h = marker.shape[0]
    cell = h // 8
    if not (marker[0, 0] == 0 and marker[0, -1] == 0 and marker[-1, 0] == 0 and marker[-1, -1] == 0):
        return False
    inner = marker[cell:-cell, cell:-cell]
    return bool((inner == 0).any() and (inner == 255).any())


def add_quiet_zone(marker: np.ndarray, quiet_cells: int) -> np.ndarray:
    """给 tag 加白色静区（AprilTag 检测必需，建议 >=1 个 cell 宽）。"""
    h, w = marker.shape
    cell = h / 8  # tag 是 8x8 格（含黑框）
    pad = int(round(cell * quiet_cells))
    padded = np.full((h + 2 * pad, w + 2 * pad), 255, dtype=np.uint8)
    padded[pad:pad + h, pad:pad + w] = marker
    return padded


def render_tag_png(tag_id: int, size_cm: float, dpi: int = 300) -> Path:
    """生成单个 tag 的打印版 PNG，返回路径。"""
    name = TAG_NAMES.get(tag_id, f"tag {tag_id}")
    # tag 图案物理边长 size_cm，8x8 格
    tag_px = int(round(size_cm / 2.54 * dpi))
    marker = generate_marker(tag_id, tag_px)
    ok = self_verify(marker, tag_id)
    if not ok:
        print(f"[警告] tag {tag_id} 布局自检失败，请检查 OpenCV 版本/字典")

    img = add_quiet_zone(marker, quiet_cells=2)

    # 底部标签区（白底 + 文字）
    label_h = int(round(tag_px * 0.28))
    canvas = np.full((img.shape[0] + label_h, img.shape[1]), 255, dtype=np.uint8)
    canvas[: img.shape[0], :] = img

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font_id = load_cjk_font(max(24, int(tag_px * 0.06)))
    font_sub = load_cjk_font(max(18, int(tag_px * 0.04)))
    cx = canvas.shape[1] / 2
    draw.text((cx, img.shape[0] + label_h * 0.22), f"tag36h11 #{tag_id}", font=font_id, fill=0, anchor="mm")
    draw.text((cx, img.shape[0] + label_h * 0.62), f"{name}  ({size_cm:.0f}cm)", font=font_sub, fill=80, anchor="mm")

    out = OUT_DIR / f"tag36_11_{tag_id:05d}.png"
    pil.save(out, dpi=(dpi, dpi))
    return out


def render_a4_pdf(tag_ids: list[int]) -> Path:
    """10 个 tag 拼一页 A4 的核对版 PDF。"""
    a4 = (2480, 3508)  # 300 dpi A4
    cols, rows = 5, 2
    cell_w, cell_h = a4[0] // cols, a4[1] // rows
    page = Image.new("L", a4, 255)
    draw = ImageDraw.Draw(page)
    font = load_cjk_font(48)

    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    for idx, tag_id in enumerate(tag_ids):
        r, c = divmod(idx, cols)
        marker = generate_marker(tag_id, 520)
        marker = add_quiet_zone(marker, 2)
        m = Image.fromarray(marker)
        # 缩放到 cell 内居中
        scale = min((cell_w - 160) / marker.shape[1], (cell_h - 260) / marker.shape[0])
        mw, mh = int(marker.shape[1] * scale), int(marker.shape[0] * scale)
        m = m.resize((mw, mh), Image.LANCZOS)
        x0 = c * cell_w + (cell_w - mw) // 2
        y0 = r * cell_h + 120
        page.paste(m, (x0, y0))
        label = f"#{tag_id}  {TAG_NAMES.get(tag_id, '')}"
        draw.text((c * cell_w + cell_w // 2, y0 + mh + 60), label, font=font, fill=0, anchor="mm")

    out = OUT_DIR / "tag36h11_all_A4.pdf"
    page.save(out, "PDF", resolution=300.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 AprilTag tag36h11 打印版")
    parser.add_argument("--ids", default="1,2,3,4,5,6,7,8,9,10", help="逗号分隔的 tag id")
    parser.add_argument("--size-cm", type=float, default=20.0, help="tag 黑框边长（cm），默认 20")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag_ids = [int(x) for x in args.ids.split(",") if x.strip()]

    print(f"字典 DICT_APRILTAG_36h11，tag 边长 {args.size_cm:.0f}cm，输出 {OUT_DIR}")
    paths = []
    for tid in tag_ids:
        p = render_tag_png(tid, args.size_cm)
        paths.append(p)
        print(f"  ✓ {p.name}  ({TAG_NAMES.get(tid, '')})")

    pdf = render_a4_pdf(tag_ids)
    print(f"  ✓ {pdf.name}  (A4 核对版)")

    print("\n打印提示：")
    print(f"  1. 独立 PNG 按 100% 比例打印，tag 黑框边长 = {args.size_cm:.0f}cm")
    print("  2. 打印后裁剪，tag 四周务必保留白色静区（已内置 2 格）")
    print("  3. 贴墙面/立柱时 tag 正面正对狗来向，高度≈相机高度，避开强光直射")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
