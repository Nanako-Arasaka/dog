import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    "high": (0, 0, 255),
    "low": (0, 255, 255),
    "normal": (0, 255, 0),
    "unknown": (128, 128, 128),
}

CLASS_LABEL_ZH = {
    "high": "偏高",
    "normal": "正常",
    "low": "偏低",
    "unknown": "未知",
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
]
_FONT_CACHE = {}


class SwitchConfirm:
    def __init__(self, confirm_frames=2):
        self.confirm_frames = max(1, int(confirm_frames))
        self.stable_class = None
        self.stable_detected = False
        self.stable_confidence = 0.0
        self.candidate_class = None
        self.candidate_detected = False
        self.candidate_count = 0

    def update(self, class_name, detected, confidence):
        class_name = str(class_name)
        detected = bool(detected)
        confidence = float(confidence)

        if self.stable_class is None:
            self.stable_class = class_name
            self.stable_detected = detected
            self.stable_confidence = confidence
            return self.stable_class, self.stable_detected, self.stable_confidence

        if class_name == self.stable_class and detected == self.stable_detected:
            self.stable_confidence = confidence
            self.candidate_class = None
            self.candidate_detected = False
            self.candidate_count = 0
            return self.stable_class, self.stable_detected, self.stable_confidence

        if class_name == self.candidate_class and detected == self.candidate_detected:
            self.candidate_count += 1
        else:
            self.candidate_class = class_name
            self.candidate_detected = detected
            self.candidate_count = 1

        if self.candidate_count >= self.confirm_frames:
            self.stable_class = self.candidate_class
            self.stable_detected = self.candidate_detected
            self.stable_confidence = confidence
            self.candidate_class = None
            self.candidate_detected = False
            self.candidate_count = 0

        return self.stable_class, self.stable_detected, self.stable_confidence


def get_font(font_size, font_path=""):
    cache_key = (int(font_size), str(font_path or "").strip())
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    candidates = []
    if font_path:
        candidates.append(str(font_path))
    env_font = os.environ.get("DASHBOARD_FONT_PATH", "").strip()
    if env_font:
        candidates.append(env_font)
    candidates.extend(FONT_CANDIDATES)

    for path in candidates:
        if path and Path(path).exists():
            try:
                font = ImageFont.truetype(path, int(font_size))
                _FONT_CACHE[cache_key] = font
                return font
            except OSError:
                continue

    font = ImageFont.load_default()
    _FONT_CACHE[cache_key] = font
    return font


def draw_text_lines_pil(frame, items, font_size=24, font_path=""):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size=font_size, font_path=font_path)

    for text, position, bgr_color in items:
        r, g, b = int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])
        draw.text(position, text, font=font, fill=(r, g, b))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_result(frame, class_name, confidence, probabilities, class_names, fps=None, detected=True, font_path=""):
    result = frame.copy()
    _, w = result.shape[:2]
    color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])

    label = CLASS_LABEL_ZH.get(class_name, class_name)
    title = f"{label} ({confidence:.2%})" if detected else "未知 (未检测到仪表盘)"

    if detected:
        bar_max = max(1, w - 40)
        bar_width = int(bar_max * max(0.0, min(1.0, confidence)))
        cv2.rectangle(result, (20, 55), (20 + bar_width, 75), color, -1)
        cv2.rectangle(result, (20, 55), (20 + bar_max, 75), (100, 100, 100), 1)

    text_items = [(title, (20, 12), color)]
    y = 84
    for idx, name in enumerate(class_names):
        prob = float(probabilities[idx]) if idx < len(probabilities) else 0.0
        p_color = CLASS_COLORS.get(name, (255, 255, 255))
        label = CLASS_LABEL_ZH.get(name, name)
        text_items.append((f"{label}: {prob:.2%}", (20, y + idx * 26), p_color))

    if fps is not None:
        text_items.append((f"FPS: {fps:.1f}", (w - 140, 10), (255, 255, 0)))

    return draw_text_lines_pil(result, text_items, font_size=24, font_path=font_path)


def make_unknown_crop(size=224):
    blank = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.putText(blank, "UNKNOWN", (32, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    return blank
