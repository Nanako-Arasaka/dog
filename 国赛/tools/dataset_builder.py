"""Build a YOLO-style inspection dataset from raw images and labels.

The output layout is:

    dataset/
      images/{train,val,test}/...
      labels/{train,val,test}/...
      dataset.yaml
      stats.csv
      stats.json

Labels are copied when a same-stem `.txt` file exists next to the source image
or under an optional labels root. Missing labels are allowed so raw capture can
be organized before annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ("zone_A", "zone_B", "zone_C", "zone_D", "gauge")


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path
    output_dir: Path
    labels_dir: Path | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 2026
    classes: tuple[str, ...] = DEFAULT_CLASSES
    move: bool = False
    workers: int = 4


@dataclass(frozen=True)
class BuildItem:
    image: Path
    label: Path | None
    split: str
    output_image: Path
    output_label: Path
    has_label: bool


def discover_images(raw_dir: Path) -> list[Path]:
    return sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def find_label(image_path: Path, labels_dir: Path | None) -> Path | None:
    candidates = [image_path.with_suffix(".txt")]
    if labels_dir is not None:
        candidates.append(labels_dir / f"{image_path.stem}.txt")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def split_images(images: list[Path], cfg: BuildConfig) -> dict[str, list[Path]]:
    ratios = (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("split ratios must be non-negative and sum to > 0")

    normalized = [r / sum(ratios) for r in ratios]
    shuffled = list(images)
    random.Random(cfg.seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(n_total * normalized[0]))
    n_val = int(round(n_total * normalized[1]))
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def unique_name(path: Path, used: set[str]) -> str:
    name = path.name
    if name not in used:
        used.add(name)
        return name
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    stem = f"{path.stem}_{digest}{path.suffix.lower()}"
    used.add(stem)
    return stem


def plan_build(cfg: BuildConfig) -> list[BuildItem]:
    images = discover_images(cfg.raw_dir)
    splits = split_images(images, cfg)
    used_names: set[str] = set()
    items: list[BuildItem] = []
    for split, split_images_ in splits.items():
        for image in split_images_:
            filename = unique_name(image, used_names)
            label = find_label(image, cfg.labels_dir)
            output_image = cfg.output_dir / "images" / split / filename
            output_label = cfg.output_dir / "labels" / split / f"{Path(filename).stem}.txt"
            items.append(BuildItem(
                image=image,
                label=label,
                split=split,
                output_image=output_image,
                output_label=output_label,
                has_label=label is not None,
            ))
    return items


def copy_or_move_item(item: BuildItem, move: bool = False) -> BuildItem:
    item.output_image.parent.mkdir(parents=True, exist_ok=True)
    item.output_label.parent.mkdir(parents=True, exist_ok=True)
    action = shutil.move if move else shutil.copy2
    action(str(item.image), str(item.output_image))
    if item.label is not None:
        shutil.copy2(item.label, item.output_label)
    else:
        item.output_label.write_text("", encoding="utf-8")
    return item


def write_dataset_yaml(cfg: BuildConfig) -> None:
    lines = [
        f"path: {cfg.output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(cfg.classes)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(cfg.classes))
    (cfg.output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_counts(labels: Iterable[Path], class_count: int) -> list[int]:
    counts = [0 for _ in range(class_count)]
    for label in labels:
        if not label.exists():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= cls < class_count:
                counts[cls] += 1
    return counts


def write_stats(cfg: BuildConfig, items: list[BuildItem]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"total_images": len(items), "splits": {}, "classes": list(cfg.classes)}
    for split in ("train", "val", "test"):
        split_items = [item for item in items if item.split == split]
        labels = [item.output_label for item in split_items]
        counts = class_counts(labels, len(cfg.classes))
        empty_labels = sum(1 for label in labels if not label.exists() or not label.read_text(encoding="utf-8").strip())
        split_summary = {
            "images": len(split_items),
            "labels": sum(1 for item in split_items if item.has_label),
            "empty_labels": empty_labels,
            "class_counts": dict(zip(cfg.classes, counts)),
        }
        summary["splits"][split] = split_summary
        rows.append({"split": split, **split_summary, "class_counts": json.dumps(split_summary["class_counts"], ensure_ascii=False)})

    with (cfg.output_dir / "stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "images", "labels", "empty_labels", "class_counts"])
        writer.writeheader()
        writer.writerows(rows)
    (cfg.output_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_dataset(cfg: BuildConfig) -> dict[str, object]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    items = plan_build(cfg)
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        futures = [pool.submit(copy_or_move_item, item, cfg.move) for item in items]
        items = [future.result() for future in as_completed(futures)]
    write_dataset_yaml(cfg)
    return write_stats(cfg, items)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset for zone-letter and gauge localization.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Raw image directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Optional directory containing YOLO .txt labels.")
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test ratios.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES), help="Comma separated class names.")
    parser.add_argument("--move", action="store_true", help="Move raw images instead of copying them.")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ratios = tuple(float(x) for x in args.split.split(","))
    if len(ratios) != 3:
        raise SystemExit("--split must be train,val,test, for example 0.8,0.1,0.1")
    classes = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    cfg = BuildConfig(
        raw_dir=args.raw_dir,
        output_dir=args.out_dir,
        labels_dir=args.labels_dir,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        seed=args.seed,
        classes=classes,
        move=args.move,
        workers=args.workers,
    )
    summary = build_dataset(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
