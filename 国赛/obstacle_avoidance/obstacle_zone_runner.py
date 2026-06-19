#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run YOLO cone detection and send obstacle-zone velocity commands."""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Iterable

from .cone_detector_yolo import ConeYoloDetector
from .cone_strategy import AvoidanceConfig, ConeDetection, plan_cone_avoidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cone obstacle-zone runner")
    parser.add_argument("--model", default="/home/jetson/yolo_deploy/cone_best.pt")
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--send-hz", type=float, default=8.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for obstacle_zone_runner") from exc

    detector = ConeYoloDetector(args.model, conf=args.conf)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera: {args.camera}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.udp_host, int(args.udp_port))
    config = AvoidanceConfig(min_confidence=args.conf)
    send_interval = 1.0 / max(args.send_hz, 0.5)
    last_send_time = 0.0

    print(f"[obstacle] model={args.model}")
    print(f"[obstacle] camera={args.camera}")
    print(f"[obstacle] UDP target={target[0]}:{target[1]} dry_run={args.dry_run}")
    print("[obstacle] press q to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[obstacle] failed to read frame")
                time.sleep(0.05)
                continue

            detections = detector.detect(frame)
            decision = plan_cone_avoidance(detections, frame.shape, config)
            now = time.time()
            if now - last_send_time >= send_interval:
                payload = decision.to_payload()
                payload["source"] = "cone_obstacle"
                send_payload(sock, target, payload, args.dry_run)
                last_send_time = now

            if not args.no_display:
                draw_debug(frame, detections, decision)
                cv2.imshow("Cone Obstacle Avoidance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        send_payload(sock, target, {"vx": 0.0, "vy": 0.0, "wz": 0.0, "source": "cone_obstacle", "state": "stop"}, args.dry_run)
        cap.release()
        sock.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


def send_payload(sock: socket.socket, target, payload: dict, dry_run: bool) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if dry_run:
        print(f"[dry-run] {payload}")
        return
    sock.sendto(data, target)


def draw_debug(frame, detections: Iterable[ConeDetection], decision) -> None:
    import cv2

    for det in detections:
        x1, y1, x2, y2 = [int(value) for value in det.xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        label = f"cone {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

    lines = [
        f"state: {decision.state}",
        f"vx: {decision.vx:.2f}",
        f"wz: {decision.wz:.2f}",
        decision.reason,
    ]
    for index, text in enumerate(lines):
        y = 32 + index * 30
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


if __name__ == "__main__":
    raise SystemExit(main())

