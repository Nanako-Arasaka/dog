#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair guosai_final.yaml paths from files found on the Jetson."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile

import yaml


FILENAMES = {
    "map_path": "guosai_rgbd_map_FINAL.osa",
    "settings_yaml": "guosai_realsense_rgbd_FINAL.yaml",
    "waypoints_yaml": "waypoints_FINAL.yaml",
    "vocabulary_path": "ORBvoc.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find final map/config files and patch guosai_final.yaml.")
    parser.add_argument("--config", default="config/guosai_final.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Extra directory to search. Can be passed more than once.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def default_roots(project_root: Path, extra_roots: list[str]) -> list[Path]:
    roots = [
        project_root,
        project_root.parent,
        project_root.parent.parent,
        Path("/home/jetson/Desktop/guosai"),
        Path("/home/jetson"),
    ]
    roots.extend(Path(item).expanduser() for item in extra_roots)
    out = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def find_file(search_roots: list[Path], filename: str) -> Path | None:
    for root in search_roots:
        direct = root / filename
        if direct.exists():
            return direct
    for root in search_roots:
        try:
            for path in root.rglob(filename):
                if path.is_file():
                    return path
        except (OSError, PermissionError):
            continue
    return None


def ensure_orbvoc(search_roots: list[Path]) -> Path | None:
    existing = find_file(search_roots, "ORBvoc.txt")
    if existing:
        return existing
    archive = find_file(search_roots, "ORBvoc.txt.tar.gz")
    if not archive:
        return None
    target_dir = archive.parent
    print(f"[INFO] extracting {archive} -> {target_dir}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(target_dir)
    return find_file([target_dir], "ORBvoc.txt")


def shorten(path: Path, project_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return "${GUOSAI_ROOT}/" + rel.as_posix()
    except ValueError:
        return str(path.resolve())


def rebuild_orbslam_command(vocab: str, settings: str, atlas: str) -> str:
    return (
        "ros2 run orbslam3 rgbd "
        f"{vocab} {settings} {atlas} "
        "--ros-args "
        "-r /camera/color/image_raw:=/camera/camera/color/image_raw "
        "-r /camera/aligned_depth_to_color/image_raw:=/camera/camera/aligned_depth_to_color/image_raw"
    )


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = Path(args.root).resolve() if args.root else config_path.parents[1]
    os.environ["GUOSAI_ROOT"] = str(project_root)
    if not config_path.exists():
        print(f"[ERROR] config not found: {config_path}")
        return 2

    cfg = load_yaml(config_path)
    cfg.setdefault("slam", {})
    cfg.setdefault("orbslam3", {})
    roots = default_roots(project_root, args.search_root)
    print("[INFO] search roots:")
    for root in roots:
        print(f"  - {root}")

    found: dict[str, str] = {}
    for key, filename in FILENAMES.items():
        path = ensure_orbvoc(roots) if key == "vocabulary_path" else find_file(roots, filename)
        if path:
            value = shorten(path, project_root)
            cfg["slam"][key] = value
            found[key] = value
            print(f"[OK] {key}: {value}")
        else:
            print(f"[MISS] {key}: {filename}")

    if all(key in found for key in ("vocabulary_path", "settings_yaml", "map_path")):
        cfg["orbslam3"]["command"] = rebuild_orbslam_command(
            found["vocabulary_path"],
            found["settings_yaml"],
            found["map_path"],
        )
        print("[OK] orbslam3.command paths rebuilt")

    if args.dry_run:
        print("[DRY-RUN] not writing config")
        return 0
    write_yaml(config_path, cfg)
    print(f"[DONE] wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
