#!/usr/bin/env python3
"""Split labeled YOLO images into train/val/test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "datasets" / "cone_raw"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "cone_yolo"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw YOLO labels into cone_yolo dataset.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Raw dataset with images/ and labels/.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output YOLO dataset root.")
    parser.add_argument("--train", type=float, default=0.80, help="Train ratio.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio.")
    parser.add_argument("--test", type=float, default=0.05, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    parser.add_argument("--clean", action="store_true", help="Remove existing output images/labels before splitting.")
    return parser.parse_args()


def list_labeled_images(raw_root: Path) -> list[Path]:
    image_dir = raw_root / "images"
    label_dir = raw_root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}")

    images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS]
    labeled = []
    missing_labels = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            labeled.append(image)
        else:
            missing_labels.append(image.name)

    if missing_labels:
        preview = "\n".join(missing_labels[:20])
        raise RuntimeError(
            "Every image must have a matching YOLO txt label. "
            "For negative samples, create an empty txt file.\n"
            f"Missing labels ({len(missing_labels)}):\n{preview}"
        )
    if not labeled:
        raise RuntimeError("No labeled images found.")
    return labeled


def prepare_output(out_root: Path, clean: bool) -> None:
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test must equal 1.0")

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    images = list_labeled_images(raw_root)
    random.Random(args.seed).shuffle(images)
    prepare_output(out_root, args.clean)

    operation = shutil.copy2 if args.copy else shutil.move
    counts = {"train": 0, "val": 0, "test": 0}

    for index, image in enumerate(images):
        split = split_name(index, len(images), args.train, args.val)
        label = raw_root / "labels" / f"{image.stem}.txt"
        operation(image, out_root / "images" / split / image.name)
        operation(label, out_root / "labels" / split / label.name)
        counts[split] += 1

    print("Split complete:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts[split]}")


if __name__ == "__main__":
    main()
