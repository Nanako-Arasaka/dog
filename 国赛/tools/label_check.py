"""Validate YOLO labels and optionally render label boxes onto images."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LabelIssue:
    level: str
    image: str
    label: str
    line: int
    message: str


@dataclass(frozen=True)
class CheckConfig:
    images_dir: Path
    labels_dir: Path
    class_count: int
    debug_dir: Path | None = None
    allow_empty: bool = False
    workers: int = 4


def try_cv2() -> Any | None:
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None


def image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def matching_label(image: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = image.relative_to(images_dir)
    return (labels_dir / rel).with_suffix(".txt")


def parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, width, height = (float(v) for v in parts[1:])
    except ValueError:
        return None
    return cls, cx, cy, width, height


def validate_one(image: Path, cfg: CheckConfig) -> tuple[list[LabelIssue], int]:
    label = matching_label(image, cfg.images_dir, cfg.labels_dir)
    issues: list[LabelIssue] = []
    valid_boxes = 0
    if not label.exists():
        return [LabelIssue("error", str(image), str(label), 0, "missing label file")], 0

    lines = label.read_text(encoding="utf-8").splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty and not cfg.allow_empty:
        issues.append(LabelIssue("warning", str(image), str(label), 0, "empty label file"))

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            issues.append(LabelIssue("error", str(image), str(label), idx, "invalid YOLO line; expected cls cx cy w h"))
            continue
        cls, cx, cy, width, height = parsed
        if cls < 0 or cls >= cfg.class_count:
            issues.append(LabelIssue("error", str(image), str(label), idx, f"class id {cls} outside [0,{cfg.class_count - 1}]"))
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            issues.append(LabelIssue("error", str(image), str(label), idx, "normalized bbox values out of range"))
            continue
        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
            issues.append(LabelIssue("error", str(image), str(label), idx, "bbox extends outside image"))
            continue
        valid_boxes += 1
    return issues, valid_boxes


def draw_debug_for_image(image: Path, cfg: CheckConfig, class_names: list[str]) -> None:
    if cfg.debug_dir is None:
        return
    cv2 = try_cv2()
    if cv2 is None:
        return
    img = cv2.imread(str(image))
    if img is None:
        return
    h, w = img.shape[:2]
    label = matching_label(image, cfg.images_dir, cfg.labels_dir)
    for line in label.read_text(encoding="utf-8").splitlines() if label.exists() else []:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        cls, cx, cy, bw, bh = parsed
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        color = (0, 255, 255) if 0 <= cls < len(class_names) and class_names[cls] == "gauge" else (255, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        cv2.putText(img, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    out_path = cfg.debug_dir / image.relative_to(cfg.images_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def check_labels(cfg: CheckConfig, class_names: list[str] | None = None) -> dict[str, object]:
    class_names = class_names or [str(i) for i in range(cfg.class_count)]
    images = image_files(cfg.images_dir)
    all_issues: list[LabelIssue] = []
    valid_boxes = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        futures = [pool.submit(validate_one, image, cfg) for image in images]
        for future in as_completed(futures):
            issues, count = future.result()
            all_issues.extend(issues)
            valid_boxes += count
    if cfg.debug_dir is not None:
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
            list(pool.map(lambda path: draw_debug_for_image(path, cfg, class_names), images))

    result = {
        "images": len(images),
        "valid_boxes": valid_boxes,
        "issues": [issue.__dict__ for issue in sorted(all_issues, key=lambda x: (x.image, x.line, x.message))],
        "ok": not any(issue.level == "error" for issue in all_issues),
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO label files and draw debug boxes.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="YOLO dataset root containing images/ and labels/.")
    parser.add_argument("--split", default="train", help="Split under dataset root.")
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--labels-dir", type=Path, default=None)
    parser.add_argument("--classes", default="zone_A,zone_B,zone_C,zone_D,gauge")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    if args.dataset_root:
        images_dir = args.dataset_root / "images" / args.split
        labels_dir = args.dataset_root / "labels" / args.split
    else:
        if args.images_dir is None or args.labels_dir is None:
            raise SystemExit("Either --dataset-root or both --images-dir and --labels-dir are required")
        images_dir = args.images_dir
        labels_dir = args.labels_dir
    cfg = CheckConfig(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_count=len(class_names),
        debug_dir=args.debug_dir,
        allow_empty=args.allow_empty,
        workers=args.workers,
    )
    result = check_labels(cfg, class_names)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
