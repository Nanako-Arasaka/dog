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
    """生成单个 tag 的灰度图（0/255 uint8），标准 AprilTag 极性。

    ⚠️ v2 修复（先前 ds 的版本是错的）：
    旧版曾对内部 6x6 做反色，理由是误以为 OpenCV `generateImageMarker` 对
    DICT_APRILTAG_36h11 输出"OpenCV 反极性"（黑底白框、白格=数据1），需再反一次
    才能给官方 apriltag 库用。**这是错的**——OpenCV 4.6+/5.x 的输出本身就是
    标准 AprilTag 极性：1 cell 黑边（外圈）、内部 6x6 数据位 black=data1 / white=data0。
    反色后 OpenCV detectMarkers 立刻从 [tag_id] → []（亲测），官方 apriltag 库同理。

    所以：直接 `generateImageMarker(..., borderBits=1)`，不要再做任何反色。
    现场主后端（官方 apriltag 库）和降级后端（OpenCV ArUco）都直接可读。

    ⚠️ side_pixels 必须整除 8（10 cell 总宽下也行——`h // 8` 仍等于 cell 像素数，
    因为 10 和 8 的最小公倍数整除 h 时值相同），否则 cell 边界与 OpenCV 位渲染错位。
    """
    side_pixels = (int(side_pixels) // 8) * 8
    if side_pixels <= 0:
        side_pixels = 8
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side_pixels, borderBits=1)
    return marker


def self_verify(marker: np.ndarray, tag_id: int) -> bool:
    """布局自检：直接用 OpenCV detectMarkers 读 marker，看能否返回正确 ID。

    detectMarkers 需要 marker 周围有白色静区才稳定识别，先 pad 2 cell 再检测。
    这是唯一权威的"位图正确"判据（仅检查"4 角黑+内部有黑有白"分不清标准极性 vs
    反色——两种都满足，但只有标准极性检测器才读得到）。
    """
    h = marker.shape[0]
    cell = h // 8
    pad = cell * 2
    padded = cv2.copyMakeBorder(marker, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=255)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    det = detector.detectMarkers(padded)
    if det[1] is None:
        return False
    return int(tag_id) in det[1].flatten().tolist()


def add_quiet_zone(marker: np.ndarray, quiet_cells: int) -> np.ndarray:
    """给 tag 加白色静区（AprilTag 检测必需，建议 >=1 个 cell 宽）。"""
    h, w = marker.shape
    cell = h / 8  # tag 是 8x8 格（含黑框）
    pad = int(round(cell * quiet_cells))
    padded = np.full((h + 2 * pad, w + 2 * pad), 255, dtype=np.uint8)
    padded[pad:pad + h, pad:pad + w] = marker
    return padded


def render_tag_png(tag_id: int, size_cm: float, dpi: int = 300) -> Path:
    """生成单个 tag 的 A4 满版打印 PNG。

    ⚠️ 打印方式：必须按「实际尺寸 / 100%」打印（不勾选"适应页面"缩放），
    这样 tag 黑框物理边长 = size_cm，与 tags.yaml 的 size_m 一致，PnP 尺度才正确。
    """
    name = TAG_NAMES.get(tag_id, f"tag {tag_id}")
    a4_w, a4_h = int(21.0 / 2.54 * dpi), int(29.7 / 2.54 * dpi)  # A4 @ dpi
    # tag_px 必须整除 8（generate_marker 内部对齐），避免 cell 浮点边界
    tag_px = max(8, (int(round(size_cm / 2.54 * dpi)) // 8) * 8)
    marker = generate_marker(tag_id, tag_px)
    ok = self_verify(marker, tag_id)
    if not ok:
        print(f"[警告] tag {tag_id} 布局自检失败，请检查 OpenCV 版本/字典")

    # 画布：A4 白底；tag 居中偏上，四周留白做静区；底部标签
    canvas = np.full((a4_h, a4_w), 255, dtype=np.uint8)
    margin_x = (a4_w - tag_px) // 2
    top = int(a4_h * 0.10)
    canvas[top:top + tag_px, margin_x:margin_x + tag_px] = marker

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font_id = load_cjk_font(int(a4_w * 0.028))
    font_sub = load_cjk_font(int(a4_w * 0.020))
    label_y = top + tag_px + int(a4_h * 0.045)
    cx = a4_w / 2
    draw.text((cx, label_y), f"tag36h11 #{tag_id}", font=font_id, fill=0, anchor="mm")
    draw.text((cx, label_y + int(a4_h * 0.035)), f"{name}  ({size_cm:.0f}cm)", font=font_sub, fill=80, anchor="mm")

    out = OUT_DIR / f"tag36_11_{tag_id:05d}.png"
    pil.save(out, dpi=(dpi, dpi))
    return out


def render_a4_pdf(tag_ids: list[int], size_cm: float = 4.0) -> Path:
    """10 个 tag 拼成 2 页 A4 的紧凑贴附版（每页 5 个，tag 约 size_cm 边长）。

    适合现场 tag 离狗近（<2m）的位点；远距离点位请用独立 A4 PNG（18cm）。

    ⚠️ size_cm 默认 4cm：A4 宽 21cm，5 tag 横向需各 ≤ 4cm + 留白才能放下，
    旧默认 8cm 会让 tag 互相重叠贴边、检测失败。
    """
    a4 = (2480, 3508)  # 300 dpi A4
    cols, rows = 5, 2
    cell_w, cell_h = a4[0] // cols, a4[1] // rows
    pages = []
    font = load_cjk_font(56)
    font_sub = load_cjk_font(40)

    for page_idx in range(2):
        page = Image.new("L", a4, 255)
        draw = ImageDraw.Draw(page)
        for slot in range(5):
            idx = page_idx * 5 + slot
            if idx >= len(tag_ids):
                break
            tag_id = tag_ids[idx]
            marker = generate_marker(tag_id, int(size_cm / 2.54 * 300))
            m = Image.fromarray(marker)
            x0 = slot * cell_w + (cell_w - m.width) // 2
            y0 = 200
            page.paste(m, (x0, y0))
            label = f"#{tag_id}  {TAG_NAMES.get(tag_id, '')}"
            draw.text((slot * cell_w + cell_w // 2, y0 + m.height + 80), label, font=font, fill=0, anchor="mm")
            draw.text((slot * cell_w + cell_w // 2, y0 + m.height + 190),
                      f"{size_cm:.0f}cm tag", font=font_sub, fill=120, anchor="mm")
        pages.append(page)

    out = OUT_DIR / "tag36h11_A4_2pages.pdf"
    pages[0].save(out, "PDF", resolution=300.0, save_all=True, append_images=pages[1:])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 AprilTag tag36h11 打印版")
    parser.add_argument("--ids", default="1,2,3,4,5,6,7,8,9,10", help="逗号分隔的 tag id")
    parser.add_argument("--size-cm", type=float, default=18.0, help="tag 黑框边长（cm），默认 18（A4 内最大）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag_ids = [int(x) for x in args.ids.split(",") if x.strip()]

    print(f"字典 DICT_APRILTAG_36h11，tag 边长 {args.size_cm:.0f}cm，输出 {OUT_DIR}")
    paths = []
    for tid in tag_ids:
        p = render_tag_png(tid, args.size_cm)
        paths.append(p)
        print(f"  ✓ {p.name}  ({TAG_NAMES.get(tid, '')})")

    pdf = render_a4_pdf(tag_ids, size_cm=4.0)
    print(f"  ✓ {pdf.name}  (2 页 A4，紧凑贴附版，tag 4cm)")

    print("\n打印提示：")
    print(f"  1. 独立 A4 PNG：打印机选「实际尺寸 / 100%」（切勿勾选适应页面缩放），")
    print(f"     打印出来 tag 黑框边长 = {args.size_cm:.0f}cm，与 config/tags.yaml 的 size_m 一致")
    print(f"     （缩放会导致 PnP 位姿尺度错误，务必按实际尺寸打）")
    print("  2. 裁剪后 tag 四周保留白色静区（A4 版已内置，四周留白即静区）")
    print("  3. 贴墙面/立柱时 tag 正面正对狗来向，高度≈相机高度，避开强光直射")
    print(f"  4. 若改了 --size-cm，必须同步改 config/tags.yaml 的 size_m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
