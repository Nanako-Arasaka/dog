#!/usr/bin/env python3
"""从视频中提取仪表盘圆形区域帧，保存为训练数据。"""
import argparse

import cv2
import numpy as np


def crop_with_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=0.8,
        minDist=50, minRadius=10, param2=100, maxRadius=300,
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype("int")
    threshold = 10
    merged = []
    for (x1, y1, r1) in circles:
        merged_flag = False
        for idx, (x2, y2, r2) in enumerate(merged):
            if np.hypot(x1 - x2, y1 - y2) < threshold:
                merged[idx] = (x1, y1, r1) if r1 >= r2 else (x2, y2, r2)
                merged_flag = True
                break
        if not merged_flag:
            merged.append((x1, y1, r1))

    if not merged:
        return None
    cx, cy, radius = max(merged, key=lambda item: item[2])
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, image.shape[1] - 1)
    y2 = min(cy + radius, image.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="从视频抽帧")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--output", required=True, help="输出图片目录")
    parser.add_argument("--start-idx", type=int, default=0, help="起始帧编号")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    frame_count = args.start_idx

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped = crop_with_hough(frame)
        if cropped is not None:
            cv2.imwrite(f"{args.output}/frame{frame_count:04d}.jpg", cropped)
            frame_count += 1

    cap.release()
    print(f"完成，共保存 {frame_count - args.start_idx} 帧到 {args.output}")


if __name__ == "__main__":
    main()
