#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO cone detector wrapper.

The expected training class list is intentionally small:

    0 cone
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .cone_strategy import ConeDetection


class ConeYoloDetector:
    def __init__(self, model_path: str, conf: float = 0.35) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"cone YOLO model not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for ConeYoloDetector") from exc
        self.model = YOLO(str(path))
        self.conf = float(conf)

    def detect(self, frame) -> List[ConeDetection]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections: List[ConeDetection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item()) if box.cls is not None else 0
                class_name = str(names.get(class_id, class_id))
                if class_name != "cone" and class_id != 0:
                    continue
                confidence = float(box.conf[0].item()) if box.conf is not None else 1.0
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    ConeDetection(
                        xyxy=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_name="cone",
                    )
                )
        return detections

