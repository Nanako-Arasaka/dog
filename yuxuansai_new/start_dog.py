#!/usr/bin/env python3
"""机器狗端：采集摄像头帧 → UDP 发送到 Jetson → 接收推理结果 → 本地可视化。"""
import argparse
import json
import os
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- 常量 ----

CLASS_COLORS = {
    "high": (0, 0, 255),
    "normal": (0, 255, 0),
    "low": (0, 255, 255),
    "unknown": (128, 128, 128),
}

CLASS_ZH_TO_EN = {"偏高": "high", "正常": "normal", "偏低": "low"}
CLASS_EN_TO_ZH = {"high": "偏高", "normal": "正常", "low": "偏低", "unknown": "未知"}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
]
_FONT_CACHE = {}

# ---- 工具函数 ----


def normalize_class(value):
    text = str(value).strip()
    low = text.lower()
    if low in ("high", "normal", "low", "unknown"):
        return low
    return CLASS_ZH_TO_EN.get(text, "unknown")


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


def open_camera(camera_device="", camera_index=0):
    if camera_device:
        cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(camera_index, cv2.CAP_ANY)


def encode_frame_packet(frame_id, frame, jpeg_quality=70):
    quality = int(max(30, min(95, jpeg_quality)))
    for _ in range(6):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            payload = encoded.tobytes()
            if len(payload) <= 65000:
                header = struct.pack("!BI", 1, frame_id)
                return header + payload
        quality = max(30, quality - 10)
    return None


def draw_overlay(frame, payload, frame_center_x, deadzone, send_fps, recv_fps,
                 infer_width, infer_height, font_path=""):
    view = frame.copy()
    h, w = view.shape[:2]
    scale_x = (w / float(infer_width)) if infer_width > 0 else 1.0
    scale_y = (h / float(infer_height)) if infer_height > 0 else 1.0
    center = max(0, min(int(frame_center_x * scale_x), w - 1))
    left_bound = max(0, center - int(deadzone * scale_x))
    right_bound = min(w - 1, center + int(deadzone * scale_x))

    cv2.line(view, (center, 0), (center, h - 1), (255, 255, 0), 1)
    cv2.line(view, (left_bound, 0), (left_bound, h - 1), (80, 80, 80), 1)
    cv2.line(view, (right_bound, 0), (right_bound, h - 1), (80, 80, 80), 1)

    cls = normalize_class(payload.get("class_en", payload.get("class", "unknown")))
    conf = float(payload.get("confidence", 0.0))
    detected = bool(payload.get("detected", False))
    cx = int(payload.get("cx", -1)) if isinstance(payload.get("cx", -1), (int, float)) else -1
    cy = int(payload.get("cy", -1)) if isinstance(payload.get("cy", -1), (int, float)) else -1

    if cx >= 0 and cy >= 0:
        draw_x = max(0, min(int(cx * scale_x), w - 1))
        draw_y = max(0, min(int(cy * scale_y), h - 1))
        cv2.circle(view, (draw_x, draw_y), 8, (255, 0, 255), 2)

    if not detected or cls not in ("high", "normal", "low"):
        show_name = "未知"
        show_color = CLASS_COLORS["unknown"]
    else:
        show_name = CLASS_EN_TO_ZH[cls]
        show_color = CLASS_COLORS[cls]

    text_items = [
        (f"结果: {show_name}", (20, 12), show_color),
        (f"class={cls} conf={conf:.2f} detected={detected}", (20, 44), (220, 220, 220)),
        (f"cx={cx} cy={cy}", (20, 72), (220, 220, 220)),
        (f"send_fps={send_fps:.1f} recv_fps={recv_fps:.1f}", (20, 100), (220, 220, 220)),
    ]
    view = draw_text_lines_pil(view, text_items, font_size=24, font_path=font_path)
    return view


# ---- CLI & Main ----


def parse_args():
    parser = argparse.ArgumentParser(description="机器狗主控：本地采集/可视化 + 发送到 Jetson 推理")
    parser.add_argument("--jetson-ip", required=True, help="Jetson IP")
    parser.add_argument("--jetson-frame-port", type=int, default=6006, help="Jetson 接收视频帧端口")
    parser.add_argument("--listen-ip", default="0.0.0.0", help="本机监听推理结果 IP")
    parser.add_argument("--listen-port", type=int, default=5005, help="本机监听推理结果端口")
    parser.add_argument("--frame-center-x", type=int, default=-1, help="推理坐标系中心 x（<0 自动取 stream-width/2）")
    parser.add_argument("--deadzone", type=int, default=40, help="左右偏移死区像素")
    parser.add_argument("--buffer-size", type=int, default=65535, help="UDP 缓冲区大小")
    parser.add_argument("--camera-device", default="/dev/video2", help="本地摄像头设备路径")
    parser.add_argument("--camera-index", type=int, default=2, help="本地摄像头索引")
    parser.add_argument("--camera-width", type=int, default=640, help="可视化窗口摄像头宽度")
    parser.add_argument("--camera-height", type=int, default=480, help="可视化窗口摄像头高度")
    parser.add_argument("--stream-width", type=int, default=320, help="发送给 Jetson 的流宽度")
    parser.add_argument("--stream-height", type=int, default=240, help="发送给 Jetson 的流高度")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG 压缩质量")
    parser.add_argument("--send-hz", type=float, default=15.0, help="发帧频率")
    parser.add_argument("--socket-timeout", type=float, default=0.005, help="UDP 接收超时（秒）")
    parser.add_argument("--font-path", default="", help="中文字体路径（PIL），留空自动查找")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.send_hz <= 0:
        raise ValueError("--send-hz 必须大于 0")
    if args.stream_width <= 0 or args.stream_height <= 0:
        raise ValueError("--stream-width 和 --stream-height 必须大于 0")

    infer_center_x = args.frame_center_x if args.frame_center_x >= 0 else (args.stream_width // 2)

    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_target = (args.jetson_ip, args.jetson_frame_port)

    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind((args.listen_ip, args.listen_port))
    rx_sock.settimeout(args.socket_timeout)

    latest_payload = {"class": "unknown", "confidence": 0.0, "detected": False, "cx": -1, "cy": -1}
    cap = open_camera(args.camera_device, args.camera_index)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"无法打开本地摄像头: {args.camera_device} / index={args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    send_interval = 1.0 / args.send_hz
    next_send = time.perf_counter()
    frame_id = 0

    send_counter = 0
    recv_counter = 0
    fps_tick = time.perf_counter()
    send_fps = 0.0
    recv_fps = 0.0

    print(f"推流目标(Jetson): {tx_target[0]}:{tx_target[1]}")
    print(f"结果监听(本机): {args.listen_ip}:{args.listen_port}")
    print("开始本地采集、发送与可视化...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.perf_counter()
            if now >= next_send:
                send_frame = cv2.resize(frame, (args.stream_width, args.stream_height),
                                        interpolation=cv2.INTER_LINEAR)
                packet = encode_frame_packet(frame_id=frame_id, frame=send_frame,
                                             jpeg_quality=args.jpeg_quality)
                if packet is not None:
                    tx_sock.sendto(packet, tx_target)
                    frame_id += 1
                    send_counter += 1
                while next_send <= now:
                    next_send += send_interval

            try:
                data, addr = rx_sock.recvfrom(args.buffer_size)
                try:
                    payload = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = None

                if isinstance(payload, dict):
                    latest_payload = payload
                    recv_counter += 1
                    cls = normalize_class(payload.get("class_en", payload.get("class", "unknown")))
                    if bool(payload.get("detected", False)) and cls in ("high", "normal", "low"):
                        print(
                            f"FROM={addr[0]}:{addr[1]} RESULT={CLASS_EN_TO_ZH[cls]} "
                            f"CONF={float(payload.get('confidence', 0.0)):.3f}"
                        )
            except socket.timeout:
                pass

            if now - fps_tick >= 1.0:
                span = max(1e-6, now - fps_tick)
                send_fps = send_counter / span
                recv_fps = recv_counter / span
                send_counter = 0
                recv_counter = 0
                fps_tick = now

            view = draw_overlay(
                frame, latest_payload, infer_center_x, args.deadzone,
                send_fps=send_fps, recv_fps=recv_fps,
                infer_width=args.stream_width, infer_height=args.stream_height,
                font_path=args.font_path,
            )
            cv2.imshow("Robot Control View", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("收到退出按键，正在退出...")
                break
    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")
    finally:
        tx_sock.close()
        rx_sock.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
