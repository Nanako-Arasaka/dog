import time

import cv2
import numpy as np


class HoughCircleCropper:
    def __init__(self, detect_interval=3, miss_sleep_ms=1.0, crop_expand_ratio=1.15):
        self.detect_interval = max(1, int(detect_interval))
        self.miss_sleep_s = max(0.0, float(miss_sleep_ms) / 1000.0)
        self.crop_expand_ratio = max(1.0, float(crop_expand_ratio))
        self.frame_idx = 0
        self.last_circle = None

    def _detect_circle(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = None
        for param2 in (100, 85, 70):
            cand = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=0.8,
                minDist=50,
                minRadius=5,
                param2=param2,
                maxRadius=300,
            )
            if cand is not None and len(cand[0]) >= 1:
                circles = cand
                break
        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype("int")
        threshold = 10
        merged_circles = []
        for (x1, y1, r1) in circles:
            merged = False
            for idx, (x2, y2, r2) in enumerate(merged_circles):
                distance = np.hypot(x1 - x2, y1 - y2)
                if distance < threshold:
                    merged_circles[idx] = (x1, y1, r1) if r1 >= r2 else (x2, y2, r2)
                    merged = True
                    break
            if not merged:
                merged_circles.append((x1, y1, r1))

        if not merged_circles:
            return None
        return max(merged_circles, key=lambda item: item[2])

    def detect(self, frame):
        self.frame_idx += 1
        should_update = self.last_circle is None or (self.frame_idx % self.detect_interval == 0)
        if should_update:
            self.last_circle = self._detect_circle(frame)
            if self.last_circle is None and self.miss_sleep_s > 0:
                time.sleep(self.miss_sleep_s)

        if self.last_circle is None:
            return None, None, None

        cx, cy, radius = self.last_circle
        radius = int(radius * self.crop_expand_ratio)
        x1 = max(cx - radius, 0)
        y1 = max(cy - radius, 0)
        x2 = min(cx + radius, frame.shape[1] - 1)
        y2 = min(cy + radius, frame.shape[0] - 1)

        if x2 <= x1 or y2 <= y1:
            return None, None, None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, None

        return crop, int(cx), int(cy)
