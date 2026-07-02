#!/usr/bin/env python3
"""Prepare data and train a YOLO detector for the PVC cone obstacle."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "config" / "cone_dataset.yaml"
DEFAULT_RAW = PROJECT_ROOT / "datasets" / "cone_raw"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "cone_yolo"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "detect"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
RUNTIME_DATA = PROJECT_ROOT / "runs" / "cone_dataset_runtime.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO for cone detection.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO dataset yaml path.")
    parser.add_argument("--prepare", action="store_true", help="Split datasets/cone_raw into datasets/cone_yolo before training.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Raw labeled dataset with images/ and labels/.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Prepared YOLO dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split ratio used with --prepare.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio used with --prepare.")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio used with --prepare.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --prepare.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model, such as yolov8n.pt or yolov8s.pt.")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Use -1 for Ultralytics auto batch.")
    parser.add_argument("--device", default="", help="Device string: '', 'cpu', '0', or '0,1'.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--name", default="cone_yolo", help="Run name under runs/detect.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting the run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Ultralytics run.")
    parser.add_argument("--export", choices=["onnx", "engine"], help="Optional export format after training.")
    parser.add_argument("--copy-best", action="store_true", default=True, help="Copy best.pt to models/cone_yolo_best.pt.")
    parser.add_argument("--no-copy-best", dest="copy_best", action="store_false", help="Do not copy best.pt to models/.")
    return parser.parse_args()


def image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def prepare_dataset(raw_root: Path, out_root: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> None:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")

    image_dir = raw_root / "images"
    label_dir = raw_root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected raw dataset folders: {image_dir} and {label_dir}")

    images = image_files(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    labeled_images: list[Path] = []
    missing_labels: list[str] = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if label.exists():
            labeled_images.append(image)
        else:
            missing_labels.append(image.name)

    if missing_labels:
        preview = "\n".join(missing_labels[:20])
        raise RuntimeError(
            "Every training image needs a matching YOLO txt label. "
            "For negative samples, create an empty txt file.\n"
            f"Missing labels ({len(missing_labels)}):\n{preview}"
        )

    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.Random(seed).shuffle(labeled_images)
    train_end = int(len(labeled_images) * train_ratio)
    val_end = train_end + int(len(labeled_images) * val_ratio)
    counts = {"train": 0, "val": 0, "test": 0}

    for index, image in enumerate(labeled_images):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "val"
        else:
            split = "test"

        label = label_dir / f"{image.stem}.txt"
        shutil.copy2(image, out_root / "images" / split / image.name)
        shutil.copy2(label, out_root / "labels" / split / label.name)
        counts[split] += 1

    print("Prepared YOLO dataset:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {counts[split]}")


def validate_label_file(label_file: Path) -> list[str]:
    errors: list[str] = []
    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return errors

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_file}:{line_no}: expected 5 columns, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_file}:{line_no}: non-numeric YOLO value")
            continue
        if class_id != 0:
            errors.append(f"{label_file}:{line_no}: class id must be 0 for cone, got {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{label_file}:{line_no}: bbox values must be normalized to [0, 1]")
    return errors


def ensure_dataset_layout(data_yaml: Path, dataset_root: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset directories:\n" + "\n".join(str(path) for path in missing))

    train_images = image_files(dataset_root / "images" / "train")
    val_images = image_files(dataset_root / "images" / "val")
    if not train_images or not val_images:
        raise RuntimeError(
            "Train/val images are empty. Put labeled YOLO data under datasets/cone_yolo "
            "or run scripts/split_yolo_dataset.py after labeling."
        )

    errors: list[str] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image in image_files(image_dir):
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"Missing label for {image}")
            else:
                errors.extend(validate_label_file(label))

    if errors:
        preview = "\n".join(errors[:30])
        raise RuntimeError(f"Dataset validation failed with {len(errors)} issue(s):\n{preview}")


def write_runtime_data_yaml(dataset_root: Path) -> Path:
    RUNTIME_DATA.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_DATA.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: cone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return RUNTIME_DATA


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    dataset_root = args.dataset.resolve()

    if args.prepare:
        prepare_dataset(
            raw_root=args.raw.resolve(),
            out_root=dataset_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    ensure_dataset_layout(data_yaml, dataset_root)
    train_data_yaml = write_runtime_data_yaml(dataset_root)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it on the training machine with:\n"
            "  pip install ultralytics\n"
            "or use the environment from the previous dashboard YOLO training."
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(train_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        pretrained=True,
        cache=False,
        single_cls=True,
        plots=True,
        close_mosaic=10,
        degrees=5.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.40,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training finished. Best weights: {best_pt}")

    if args.copy_best:
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        target = DEFAULT_MODEL_DIR / "cone_yolo_best.pt"
        shutil.copy2(best_pt, target)
        print(f"Copied best weights to: {target}")

    if args.export:
        exported = YOLO(str(best_pt)).export(format=args.export, imgsz=args.imgsz, device=args.device)
        print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()
