#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态分析 config/tags.yaml 里已标定的 tag —— 不需要 ROS。

输出:
  1. 已标定 / 占位 tag 概览
  2. 已标定 tag 之间的 3D 距离矩阵
  3. 与占位值 (z=0.45) 的偏离提示(注意:ORB 世界系轴向 arbitrary,这是参考而非判据)
  4. 给每个 tag 打 OK / WARN / FAIL 标记

用法:
  python3 tools/analyze_tags.py
  python3 tools/analyze_tags.py --tags-yaml config/tags.yaml
  python3 tools/analyze_tags.py --json   # 输出 JSON 供程序消费
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# 现场实测默认参数(对齐 calibrate_tags.py --verify)
MIN_INTER_TAG_DISTANCE_M = 0.5    # 两 tag 最小 3D 距离(参照 waypoints_FINAL.yaml 的 ≥0.5m 判据)
MAX_INTER_TAG_DISTANCE_M = 6.0    # 两 tag 最大 3D 距离(超出怀疑飘到别处)
PLACEHOLDER_Z_M = 0.45            # tags.yaml 默认占位 z(贴墙高度)
# 任何"已标定"tag 距离占位值超过这个数,基本可确认是真值而非忘了删占位
PLACEHOLDER_DRIFT_THRESHOLD_M = 0.15


@dataclass
class TagEntry:
    id: int
    name: str
    size_m: float
    note: str
    world: dict  # {x, y, z, yaw_deg, pitch_deg, roll_deg}
    is_calibrated: bool


def load_tags(path: Path) -> list[TagEntry]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[TagEntry] = []
    placeholder = {
        "x": 0.0, "y": 0.0, "z": PLACEHOLDER_Z_M,
        "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
    }
    for e in data.get("tags", []):
        world = e.get("world") or {}
        # 占位 tag = 6 个字段完全等于 (0, 0, 0.45, 0, 0, 0)
        # 任何字段偏离 > 阈值都算已标定
        calibrated = any(
            abs(float(world.get(k, placeholder[k])) - placeholder[k]) > 1e-3
            for k in placeholder
        )
        out.append(TagEntry(
            id=int(e["id"]),
            name=str(e.get("name", f"tag_{e['id']}")),
            size_m=float(e.get("size_m", data.get("default_size_m", 0.20))),
            note=str(e.get("note", "")),
            world={
                "x": float(world.get("x", 0.0)),
                "y": float(world.get("y", 0.0)),
                "z": float(world.get("z", 0.45)),
                "yaw_deg": float(world.get("yaw_deg", 0.0)),
                "pitch_deg": float(world.get("pitch_deg", 0.0)),
                "roll_deg": float(world.get("roll_deg", 0.0)),
            },
            is_calibrated=calibrated,
        ))
    return out


def dist(a: TagEntry, b: TagEntry) -> float:
    dx = a.world["x"] - b.world["x"]
    dy = a.world["y"] - b.world["y"]
    dz = a.world["z"] - b.world["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def fmt_pose(tag: TagEntry) -> str:
    w = tag.world
    return (
        f"({w['x']:+.3f}, {w['y']:+.3f}, {w['z']:+.3f}) "
        f"yaw={w['yaw_deg']:+.1f}° pitch={w['pitch_deg']:+.1f}° roll={w['roll_deg']:+.1f}°"
    )


def assess_tag(tag: TagEntry) -> tuple[str, list[str]]:
    """返回 (状态, 提示列表)。状态: OK / WARN / FAIL / PLACEHOLDER。"""
    notes: list[str] = []
    if not tag.is_calibrated:
        return "PLACEHOLDER", ["待现场标定"]

    w = tag.world
    # 占位漂移(忘了改回 0,0,0.45)
    placeholder_dist = math.sqrt(w["x"] ** 2 + w["y"] ** 2 + (w["z"] - PLACEHOLDER_Z_M) ** 2)
    if placeholder_dist < PLACEHOLDER_DRIFT_THRESHOLD_M:
        return "FAIL", [f"坐标几乎就是占位值 (drift={placeholder_dist*100:.1f}cm),可能被覆盖丢了"]

    # roll/pitch 异常大提示
    if abs(w["roll_deg"]) > 30:
        notes.append(f"roll={w['roll_deg']:.1f}° 较大,确认 tag 确实斜贴或 ZYX 分解边界")
    if abs(w["pitch_deg"]) > 60:
        notes.append(f"pitch={w['pitch_deg']:.1f}° 较大,贴墙 tag 一般 ±90°")
    if abs(w["yaw_deg"]) > 360:
        notes.append("yaw 超出 [-360, 360] 范围")

    return ("WARN" if notes else "OK"), notes


def report_text(entries: list[TagEntry]) -> str:
    calibrated = [t for t in entries if t.is_calibrated]
    placeholders = [t for t in entries if not t.is_calibrated]

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  AprilTag 标定状态报告 —— {len(calibrated)}/{len(entries)} 已标定")
    lines.append("=" * 78)

    # —— 表 1:逐 tag 概览
    lines.append("")
    lines.append("【1】逐 tag 状态")
    lines.append("-" * 78)
    for t in entries:
        status, notes = assess_tag(t)
        marker = {"OK": "✓", "WARN": "△", "FAIL": "✗", "PLACEHOLDER": "·"}.get(status, "?")
        if status == "PLACEHOLDER":
            lines.append(f"  {marker} tag {t.id:>2} {t.name:<6}  PLACEHOLDER   {t.note}")
        else:
            lines.append(f"  {marker} tag {t.id:>2} {t.name:<6}  {status:<11} {fmt_pose(t)}")
            lines.append(f"        note: {t.note}")
            for n in notes:
                lines.append(f"        - {n}")

    # —— 表 2:已标定 tag 之间的距离矩阵
    if len(calibrated) >= 2:
        lines.append("")
        lines.append("【2】已标定 tag 之间的 3D 距离 (米)")
        lines.append("-" * 78)
        ids = [t.id for t in calibrated]
        header = "       " + "".join(f"  t{tid:>2}" for tid in ids)
        lines.append(header)
        for a in calibrated:
            row = f"  t{a.id:>2}  "
            for b in calibrated:
                if a.id == b.id:
                    row += "    - "
                else:
                    d = dist(a, b)
                    marker = ""
                    if d < MIN_INTER_TAG_DISTANCE_M:
                        marker = " ⚠"
                    elif d > MAX_INTER_TAG_DISTANCE_M:
                        marker = " ⚠远"
                    row += f"{d:5.2f}{marker}"
            lines.append(row)
        lines.append(f"  (⚠ = < {MIN_INTER_TAG_DISTANCE_M}m 或 > {MAX_INTER_TAG_DISTANCE_M}m,可能重叠或失联)")
    else:
        lines.append("")
        lines.append("【2】已标定 tag 不足 2 个,跳过距离矩阵")

    # —— 表 3:与占位 z=0.45 的偏离
    lines.append("")
    lines.append("【3】已标定 tag 与占位 (z=0.45) 的偏离 (米,仅参考 —— ORB 世界系轴向 arbitrary)")
    lines.append("-" * 78)
    for t in calibrated:
        w = t.world
        d = math.sqrt(w["x"] ** 2 + w["y"] ** 2 + (w["z"] - PLACEHOLDER_Z_M) ** 2)
        lines.append(f"  tag {t.id:>2}  Δxyz = ({w['x']:+.3f}, {w['y']:+.3f}, {w['z']-PLACEHOLDER_Z_M:+.3f})  |Δ|={d:.3f}m")

    # —— 表 4:整体结论
    lines.append("")
    lines.append("【4】整体结论")
    lines.append("-" * 78)
    ok_count = sum(1 for t in entries if assess_tag(t)[0] == "OK")
    warn_count = sum(1 for t in entries if assess_tag(t)[0] == "WARN")
    fail_count = sum(1 for t in entries if assess_tag(t)[0] == "FAIL")
    ph_count = len(placeholders)

    if calibrated:
        pairs = [(a, b) for i, a in enumerate(calibrated) for b in calibrated[i+1:]]
        bad_pairs = [(a, b, dist(a, b)) for a, b in pairs
                     if dist(a, b) < MIN_INTER_TAG_DISTANCE_M or dist(a, b) > MAX_INTER_TAG_DISTANCE_M]
    else:
        bad_pairs = []

    lines.append(f"  OK={ok_count}  WARN={warn_count}  FAIL={fail_count}  PLACEHOLDER={ph_count}")
    if bad_pairs:
        lines.append(f"  ⚠ {len(bad_pairs)} 对 tag 距离异常:")
        for a, b, d in bad_pairs:
            lines.append(f"    - tag {a.id} ↔ tag {b.id}: {d:.3f}m")
    else:
        lines.append("  ✓ 已标定 tag 之间距离全部在合理区间")

    lines.append("")
    lines.append("【5】建议下一步")
    lines.append("-" * 78)
    if not calibrated:
        lines.append("  · 当前没有任何已标定 tag,先跑:")
        lines.append("      python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml --yes --ids 1")
    elif ph_count > 0:
        # 还没标定的 tag
        lines.append(f"  · 还有 {ph_count} 个 tag 待采,优先级(SOP):")
        order = [7, 6, 5, 4, 1, 9, 8, 2, 3, 10]   # memory 中的优先级
        ph_ids = [t.id for t in placeholders]
        sorted_ph = [i for i in order if i in ph_ids] + [i for i in ph_ids if i not in order]
        for tid in sorted_ph[:7]:  # 最多打印 7 行
            tname = next((t.name for t in placeholders if t.id == tid), f"tag_{tid}")
            note = next((t.note for t in placeholders if t.id == tid), "")
            lines.append(f"      tag {tid:>2} ({tname})   {note}")
        lines.append("  · 每采一个就跑一次 analyze_tags.py 立即看,有问题 r 重采")
    else:
        lines.append("  · 全部 tag 已标定,跑 --verify 验证:")
        lines.append("      python3 tools/calibrate_tags.py --tags-yaml config/tags.yaml --verify --ids all")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def report_json(entries: list[TagEntry]) -> dict:
    calibrated = [t for t in entries if t.is_calibrated]
    pairs = []
    for i, a in enumerate(calibrated):
        for b in calibrated[i+1:]:
            d = dist(a, b)
            pairs.append({
                "a": a.id, "b": b.id,
                "distance_m": round(d, 4),
                "ok": MIN_INTER_TAG_DISTANCE_M <= d <= MAX_INTER_TAG_DISTANCE_M,
            })
    return {
        "total": len(entries),
        "calibrated": len(calibrated),
        "placeholders": len(entries) - len(calibrated),
        "tags": [
            {
                "id": t.id,
                "name": t.name,
                "note": t.note,
                "is_calibrated": t.is_calibrated,
                "status": assess_tag(t)[0],
                "world": {k: round(v, 4) for k, v in t.world.items()},
            }
            for t in entries
        ],
        "pairs": pairs,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="静态分析 tags.yaml")
    p.add_argument("--tags-yaml", default=str(REPO_ROOT / "config" / "tags.yaml"))
    p.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = p.parse_args()

    path = Path(args.tags_yaml).expanduser().resolve()
    if not path.exists():
        print(f"[错误] 找不到 {path}", file=sys.stderr)
        return 2

    entries = load_tags(path)
    if args.json:
        print(json.dumps(report_json(entries), ensure_ascii=False, indent=2))
    else:
        print(report_text(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
