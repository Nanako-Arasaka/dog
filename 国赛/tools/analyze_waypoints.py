#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态分析 waypoints_FINAL.yaml —— 不需要 ROS。

输出:
  1. 13 个航点的全局散布 (x-span / y-span / yaw 一致性)
  2. FSM 关键路径相邻航点距离矩阵 (start → obstacle → inspection → pick → place → finish)
  3. 距离 < 0.5m 的「可疑重合」对 —— 这些几乎肯定 SLAM 失跟污染,需重采
  4. inspection 两侧 (side_1 / side_2) 的 yaw 差 —— 是否真正「在两侧」
  5. 整体结论:能用 / 哪几个必须重采 / 重采优先级

用法:
  python3 tools/analyze_waypoints.py
  python3 tools/analyze_waypoints.py --waypoints-yaml /home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml
  python3 tools/analyze_waypoints.py --json   # 输出 JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = Path("/home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml")
GIT_PATH = REPO_ROOT / "jetson_payload/slam_maps/waypoints_FINAL.yaml"

# 现场阈值(对齐 memory waypoints_FINAL.yaml 判据)
MIN_INTER_WAYPOINT_M = 0.5    # 两航点最小 3D 距离
MAX_INTER_WAYPOINT_M = 8.0    # 两航点最大 3D 距离(超出怀疑飞到另一簇)
SIDE_YAW_DIFF_DEG = 90.0      # inspection 两侧 yaw 差(若物理上真是两侧)
PLACE_MIN_DISTANCE_M = 0.5    # 放置点之间的最小距离(独立箱子)
ORIGIN_CLUSTER_RADIUS_M = 0.15 # 距原点 < 此值 = 「聚在原点」(SLAM 失跟典型征兆)

# FSM 关键路径(从 config/guosai_final.yaml 的 fsm.* 字段读,这里硬编码做兜底)
DEFAULT_FSM_PATH = [
    "start_exit",
    "obstacle_entry",
    "obstacle_exit",
    "inspection_box_1_side_1",
    "inspection_box_1_side_2",
    "inspection_box_2_side_1",
    "inspection_box_2_side_2",
    "pick_area",
    "place_A",
    "place_B",
    "place_C",
    "place_D",
    "finish",
]


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    yaw: float


@dataclass
class Analysis:
    waypoints: list[Waypoint]
    path_warnings: list[str] = field(default_factory=list)
    suspect_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    side_yaw_issues: list[tuple[str, str, float]] = field(default_factory=list)
    origin_cluster: list[str] = field(default_factory=list)


def load_waypoints(path: Path) -> list[Waypoint]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for e in data.get("waypoints", []):
        out.append(Waypoint(
            name=str(e["name"]),
            x=float(e.get("x", 0.0)),
            y=float(e.get("y", 0.0)),
            yaw=float(e.get("yaw", 0.0)),
        ))
    return out


def dist2d(a: Waypoint, b: Waypoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def yaw_diff_deg(a: Waypoint, b: Waypoint) -> float:
    """考虑 ±π 环绕的 yaw 差(度)。"""
    d = (a.yaw - b.yaw) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return abs(math.degrees(d))


def ascii_plot(wps: list[Waypoint], a: Analysis) -> list[str]:
    """ASCII 俯视图:每个航点用首字母标注,聚在原点的航点用大写红 ⚠。"""
    if not wps:
        return ["  (空)"]
    width, height = 70, 22
    xs = [w.x for w in wps]
    ys = [w.y for w in wps]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # 加点 padding
    pad_x = max(0.2, (xmax - xmin) * 0.05)
    pad_y = max(0.2, (ymax - ymin) * 0.05)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    # 注意:y 行向上,所以 row = (ymax - y) / (ymax-ymin) * (height-1)
    grid = [[" "] * width for _ in range(height)]
    origin_cluster = set(a.origin_cluster)
    plotted: list[tuple[int, int, str, str]] = []
    for w in wps:
        cx = int((w.x - xmin) / (xmax - xmin) * (width - 1)) if xmax > xmin else width // 2
        # y 翻转:屏幕行 0 = 顶部 = ymax
        cy = int((ymax - w.y) / (ymax - ymin) * (height - 1)) if ymax > ymin else height // 2
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        label = w.name[0].upper()
        plotted.append((cy, cx, label, w.name))

    # 把每个点画到 grid,短名标在旁边(避免重叠)
    for cy, cx, label, full_name in plotted:
        is_bad = full_name in origin_cluster
        ch = "✗" if is_bad else label
        grid[cy][cx] = ch

    out: list[str] = []
    out.append(f"  x: [{xmin:+.2f}, {xmax:+.2f}]    y: [{ymin:+.2f}, {ymax:+.2f}]")
    out.append("  " + "+" + "-" * width + "+")
    for row in grid:
        out.append("  |" + "".join(row) + "|")
    out.append("  " + "+" + "-" * width + "+")

    # 图例
    out.append("")
    out.append("  图例:")
    out.append("    首字母 = 航点 (S=start, O=obstacle, I=inspection, P=pick/place, F=finish)")
    out.append("    ✗ (红) = 距原点 < 0.15m,SLAM 失跟污染")
    legend = []
    by_initial: dict[str, list[str]] = {}
    for w in wps:
        ch = w.name[0].upper()
        by_initial.setdefault(ch, []).append(w.name)
    for ch in sorted(by_initial):
        names = by_initial[ch]
        if len(names) == 1:
            legend.append(f"    {ch} = {names[0]}")
        else:
            legend.append(f"    {ch} = {names[0]} 等 ({len(names)} 个)")
    out.extend(legend)
    return out


def analyze(wps: list[Waypoint], fsm_path: list[str]) -> Analysis:
    a = Analysis(waypoints=wps)
    by_name = {w.name: w for w in wps}

    # 1) 整体散布
    if not wps:
        return a

    # 2) 与原点距离(失跟典型征兆)
    for w in wps:
        if math.hypot(w.x, w.y) < ORIGIN_CLUSTER_RADIUS_M:
            a.origin_cluster.append(w.name)

    # 3) FSM 关键路径相邻距离
    missing = [n for n in fsm_path if n not in by_name]
    if missing:
        a.path_warnings.append(f"FSM 路径上有 {len(missing)} 个航点缺失:{missing}")

    prev = None
    for name in fsm_path:
        if name not in by_name:
            prev = None
            continue
        cur = by_name[name]
        if prev is not None:
            d = dist2d(prev, cur)
            if d < MIN_INTER_WAYPOINT_M:
                a.suspect_pairs.append((prev.name, cur.name, d))
        prev = cur

    # 4) place_A↔B↔C↔D 互相距离(独立箱子 ≥ 0.5m)
    places = [by_name[n] for n in ("place_A", "place_B", "place_C", "place_D") if n in by_name]
    for i in range(len(places)):
        for j in range(i + 1, len(places)):
            d = dist2d(places[i], places[j])
            if d < PLACE_MIN_DISTANCE_M:
                a.suspect_pairs.append((places[i].name, places[j].name, d))

    # 5) inspection 两侧 yaw 差
    for box in (1, 2):
        s1 = by_name.get(f"inspection_box_{box}_side_1")
        s2 = by_name.get(f"inspection_box_{box}_side_2")
        if s1 and s2:
            diff = yaw_diff_deg(s1, s2)
            if diff < SIDE_YAW_DIFF_DEG:
                a.side_yaw_issues.append((s1.name, s2.name, diff))

    return a


def report_text(a: Analysis, source_label: str) -> str:
    wps = a.waypoints
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append(f"  航点分析报告 —— {len(wps)} 个航点 (来源: {source_label})")
    lines.append("=" * 78)

    if not wps:
        lines.append("  (空)")
        return "\n".join(lines)

    # —— 顶部:ASCII 俯视图 (x=列, y=行, y 向上)
    lines.append("")
    lines.append("【0】俯视图 (ORB 世界系 xy 平面 —— 轴向 arbitrary,看分布即可)")
    lines.append("-" * 78)
    lines.extend(ascii_plot(wps, a))
    lines.append("")

    # 整体散布
    xs = [w.x for w in wps]
    ys = [w.y for w in wps]
    lines.append("")
    lines.append("【1】整体散布 (ORB 世界系,轴向 arbitrary —— 看距离,不要看绝对值)")
    lines.append("-" * 78)
    lines.append(f"  x 范围: [{min(xs):+.3f}, {max(xs):+.3f}]   span={max(xs)-min(xs):.3f}m")
    lines.append(f"  y 范围: [{min(ys):+.3f}, {max(ys):+.3f}]   span={max(ys)-min(ys):.3f}m")

    # 原点聚类
    if a.origin_cluster:
        lines.append("")
        lines.append(f"  ⚠ {len(a.origin_cluster)} 个航点距原点 < {ORIGIN_CLUSTER_RADIUS_M}m —— SLAM 失跟典型征兆:")
        for n in a.origin_cluster:
            lines.append(f"    - {n}")
    else:
        lines.append(f"  ✓ 无航点聚在原点附近 (阈值 {ORIGIN_CLUSTER_RADIUS_M}m)")

    # 关键路径相邻距离
    lines.append("")
    lines.append("【2】FSM 关键路径相邻航点距离")
    lines.append("-" * 78)
    by_name = {w.name: w for w in wps}
    prev = None
    for name in DEFAULT_FSM_PATH:
        cur = by_name.get(name)
        if cur is None:
            lines.append(f"  --- {name} (缺失)")
            prev = None
            continue
        if prev is not None:
            d = dist2d(prev, cur)
            marker = " ⚠" if d < MIN_INTER_WAYPOINT_M else (" ⚠远" if d > MAX_INTER_WAYPOINT_M else " ✓")
            lines.append(
                f"  {prev.name:>25} → {cur.name:<25}  "
                f"Δ={d:5.2f}m  Δyaw={yaw_diff_deg(prev, cur):5.1f}°{marker}"
            )
        else:
            lines.append(f"  --- {cur.name} (起点)")
        prev = cur

    # place 互相距离
    lines.append("")
    lines.append("【3】place_A/B/C/D 互相距离 (独立箱子 ≥ 0.5m)")
    lines.append("-" * 78)
    places = [(n, by_name[n]) for n in ("place_A", "place_B", "place_C", "place_D") if n in by_name]
    if len(places) < 2:
        lines.append("  (不足 2 个 place 航点)")
    else:
        for i in range(len(places)):
            for j in range(i + 1, len(places)):
                na, wa = places[i]
                nb, wb = places[j]
                d = dist2d(wa, wb)
                marker = " ⚠" if d < PLACE_MIN_DISTANCE_M else " ✓"
                lines.append(f"  {na:>8} ↔ {nb:<8}  Δ={d:.2f}m{marker}")

    # inspection 两侧 yaw
    lines.append("")
    lines.append(f"【4】inspection 两侧 yaw 差 (期望 ≥ {SIDE_YAW_DIFF_DEG:.0f}°)")
    lines.append("-" * 78)
    if a.side_yaw_issues:
        for n1, n2, d in a.side_yaw_issues:
            lines.append(f"  ✗ {n1} ↔ {n2}: yaw 差仅 {d:.1f}° —— 不是真正在两侧")
    else:
        lines.append("  ✓ 所有 inspection 箱子两侧 yaw 差都 ≥ 90°")

    # 全部两两距离中的可疑对
    lines.append("")
    lines.append("【5】全部可疑对 (Δ < 0.5m,可能采到同一位置)")
    lines.append("-" * 78)
    if a.suspect_pairs:
        # 按距离升序排
        for n1, n2, d in sorted(a.suspect_pairs, key=lambda x: x[2]):
            lines.append(f"  ⚠ {n1:>25} ↔ {n2:<25}  Δ={d:.3f}m")
    else:
        lines.append("  ✓ 无")

    # 整体结论
    lines.append("")
    lines.append("【6】整体结论 & 建议")
    lines.append("-" * 78)
    critical = len(a.suspect_pairs) + len(a.origin_cluster) + len(a.side_yaw_issues)
    if critical == 0:
        lines.append("  ✓ 全部 13 个航点可用,FSM 可以跑通")
    else:
        lines.append(f"  ⚠ 发现 {critical} 个问题,详细看上面表格")

        # 哪些必须重采
        must_recollect = set()
        if a.origin_cluster:
            for n in a.origin_cluster:
                must_recollect.add(n)
                # 原点聚类的航点通常成对出错,把它的邻居也加进去(启发式)
        for n1, n2, d in a.suspect_pairs:
            must_recollect.add(n1)
            must_recollect.add(n2)
        for n1, n2, _ in a.side_yaw_issues:
            must_recollect.add(n1)
            must_recollect.add(n2)

        if must_recollect:
            lines.append("")
            lines.append(f"  🔴 必须重采的航点 ({len(must_recollect)} 个):")
            for n in sorted(must_recollect, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 99):
                lines.append(f"      - {n}")
            lines.append("")
            lines.append("  重采命令(单点模式,降低 ORB 失跟风险):")
            lines.append("      python3 scripts/waypoint_capture_tool.py \\")
            lines.append("          --output /home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml \\")
            lines.append("          --pose-topic /camera_pose --pose-type pose_stamped \\")
            lines.append("          --timeout-sec 60 \\")
            lines.append("          --single <NAME>")
            lines.append("      # 重采完跑本分析器确认通过,再 cp 到 git 路径并 commit")
            lines.append("      cp /home/jetson/Desktop/guosai/slam_maps/waypoints_FINAL.yaml \\")
            lines.append("         /home/jetson/Desktop/guosai/dog_repo/国赛/jetson_payload/slam_maps/waypoints_FINAL.yaml")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def report_json(a: Analysis) -> dict:
    return {
        "total": len(a.waypoints),
        "origin_cluster": a.origin_cluster,
        "suspect_pairs": [{"a": n1, "b": n2, "distance_m": round(d, 4)} for n1, n2, d in a.suspect_pairs],
        "side_yaw_issues": [{"a": n1, "b": n2, "yaw_diff_deg": round(d, 2)} for n1, n2, d in a.side_yaw_issues],
        "waypoints": [
            {"name": w.name, "x": round(w.x, 4), "y": round(w.y, 4), "yaw_deg": round(math.degrees(w.yaw), 2)}
            for w in a.waypoints
        ],
    }


def _obstacle_rect(repo_root: Path):
    """读取 cone_avoidance/competition_map.yaml 的 obstacle_zone_rect。

    返回 ((xmin,ymin),(xmax,ymax)) 或 None(文件不存在/无该键)。
    """
    path = repo_root / "cone_avoidance" / "competition_map.yaml"
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    rect = data.get("obstacle_zone_rect") or {}
    try:
        xmin = float(rect["xmin"])
        xmax = float(rect["xmax"])
        ymin = float(rect["ymin"])
        ymax = float(rect["ymax"])
    except (KeyError, TypeError, ValueError):
        return None
    return ((xmin, ymin), (xmax, ymax))


def render_svg(a: Analysis, width: int = 900, height: int = 600) -> str:
    """生成 SVG 导航路径图,浏览器打开看(无需 ROS,纯静态)。

    包含:FSM 顺序路径(带方向箭头 + 序号 + 距离)、障碍区边界框、
    航点颜色分区 + yaw 朝向、可疑对红线。
    """
    wps = a.waypoints
    if not wps:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><text>empty</text></svg>"
    xs = [w.x for w in wps]
    ys = [w.y for w in wps]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 1.0
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    # 等比例缩放
    sx = (width - 120) / (xmax - xmin)
    sy = (height - 120) / (ymax - ymin)
    s = min(sx, sy)

    def to_svg(x: float, y: float) -> tuple[float, float]:
        # 翻转 y 向上,加 padding
        cx = 60 + (x - xmin) * s
        cy = (height - 60) - (y - ymin) * s
        return cx, cy

    bad_set = set(a.origin_cluster)
    suspect_set = {n for n1, n2, _ in a.suspect_pairs for n in (n1, n2)}

    # 按类别上色
    def color_of(name: str) -> str:
        if name in bad_set:
            return "#dc2626"  # 红 - 原点聚类(SLAM 失跟)
        if name.startswith("start"):
            return "#16a34a"   # 绿 - 起点
        if name.startswith("obstacle"):
            return "#ea580c"   # 橙 - 障碍
        if name.startswith("inspection"):
            return "#7c3aed"   # 紫 - 巡检
        if name.startswith("pick") or name.startswith("place"):
            return "#0891b2"   # 青 - 抓放
        if name == "finish":
            return "#db2777"   # 粉 - 终点
        return "#6b7280"

    parts: list[str] = []
    parts.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        "style='font-family:ui-monospace,Menlo,monospace;background:#fafafa'>"
    )
    parts.append(f"<text x='20' y='30' font-size='18' font-weight='bold'>导航路径俯视图 (ORB 世界系 · FSM 顺序)</text>")
    parts.append(f"<text x='20' y='52' font-size='12' fill='#555'>{len(wps)} 个航点 · 灰箭头=导航方向 · 数字=FSM 顺序 · 蓝虚线=障碍区边界</text>")

    # 网格
    parts.append(f"<g stroke='#e5e7eb' stroke-width='1'>")
    step = 1.0 if (xmax - xmin) > 6 else 0.5
    gx = math.floor(xmin / step) * step
    while gx <= xmax:
        cx, _ = to_svg(gx, ymin)
        parts.append(f"<line x1='{cx:.1f}' y1='0' x2='{cx:.1f}' y2='{height}'/>")
        parts.append(f"<text x='{cx+2:.1f}' y='{height-5}' font-size='10' fill='#888'>{gx:+.1f}</text>")
        gx += step
    gy = math.floor(ymin / step) * step
    while gy <= ymax:
        _, cy = to_svg(xmin, gy)
        parts.append(f"<line x1='0' y1='{cy:.1f}' x2='{width}' y2='{cy:.1f}'/>")
        parts.append(f"<text x='5' y='{cy-2:.1f}' font-size='10' fill='#888'>{gy:+.1f}</text>")
        gy += step
    parts.append("</g>")

    # 可疑对连线(淡红)
    for n1, n2, d in a.suspect_pairs:
        w1 = next((w for w in wps if w.name == n1), None)
        w2 = next((w for w in wps if w.name == n2), None)
        if w1 and w2:
            x1, y1 = to_svg(w1.x, w1.y)
            x2, y2 = to_svg(w2.x, w2.y)
            parts.append(
                f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                f"stroke='#dc2626' stroke-width='2' stroke-dasharray='4,3' opacity='0.7'/>"
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            parts.append(f"<text x='{mx:.1f}' y='{my-5:.1f}' font-size='10' fill='#dc2626' text-anchor='middle'>Δ={d:.2f}m</text>")

    # 障碍区边界(可选,读 cone_avoidance/competition_map.yaml)
    rect = _obstacle_rect(REPO_ROOT)
    if rect is not None:
        (rx1, ry1), (rx2, ry2) = rect
        x1, y1 = to_svg(rx1, ry1)
        x2, y2 = to_svg(rx2, ry2)
        parts.append(
            f"<rect x='{min(x1,x2):.1f}' y='{min(y1,y2):.1f}' width='{abs(x2-x1):.1f}' "
            f"height='{abs(y2-y1):.1f}' fill='#3b82f6' fill-opacity='0.05' "
            f"stroke='#3b82f6' stroke-width='1.5' stroke-dasharray='6,4'/>"
        )
        parts.append(f"<text x='{(x1+x2)/2:.1f}' y='{min(y1,y2)-6:.1f}' font-size='11' fill='#2563eb' text-anchor='middle'>障碍区 (cone avoidance)</text>")

    # FSM 导航路径(带方向箭头 + 序号 + 距离)
    by_name = {w.name: w for w in wps}
    path_pts = [by_name[n] for n in DEFAULT_FSM_PATH if n in by_name]
    for i in range(len(path_pts) - 1):
        wa, wb = path_pts[i], path_pts[i + 1]
        x1, y1 = to_svg(wa.x, wa.y)
        x2, y2 = to_svg(wb.x, wb.y)
        # 画主线
        parts.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='#64748b' stroke-width='1.6' opacity='0.55'/>"
        )
        # 方向箭头(线段 55% 处,指向终点)
        ax = x1 + (x2 - x1) * 0.55
        ay = y1 + (y2 - y1) * 0.55
        ang = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 9
        for da in (0.6, -0.6):
            px = ax - arrow_len * math.cos(ang + da)
            py = ay - arrow_len * math.sin(ang + da)
            parts.append(f"<line x1='{ax:.1f}' y1='{ay:.1f}' x2='{px:.1f}' y2='{py:.1f}' stroke='#475569' stroke-width='1.6'/>")
        # 距离标注(线段中点偏上)
        d = math.hypot(wa.x - wb.x, wa.y - wb.y)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 6
        parts.append(f"<text x='{mx:.1f}' y='{my:.1f}' font-size='9.5' fill='#64748b' text-anchor='middle'>{d:.2f}m</text>")

    # 航点(带 FSM 顺序编号)
    for idx, w in enumerate(wps):
        cx, cy = to_svg(w.x, w.y)
        col = color_of(w.name)
        # yaw 箭头
        arrow_len = 30
        ax = cx + math.cos(w.yaw) * arrow_len
        ay = cy - math.sin(w.yaw) * arrow_len
        parts.append(
            f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{ax:.1f}' y2='{ay:.1f}' "
            f"stroke='{col}' stroke-width='1.5' opacity='0.6'/>"
        )
        r = 9 if w.name in bad_set else 6
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{r}' fill='{col}' stroke='white' stroke-width='2'>"
            f"<title>{w.name}  x={w.x:+.3f}  y={w.y:+.3f}  yaw={math.degrees(w.yaw):+.1f}°</title></circle>"
        )
        # FSM 顺序号(画在点内,起点 1 终点 13)
        order = DEFAULT_FSM_PATH.index(w.name) + 1 if w.name in DEFAULT_FSM_PATH else None
        if order is not None:
            parts.append(
                f"<text x='{cx:.1f}' y='{cy+3.5:.1f}' font-size='9' fill='white' text-anchor='middle' font-weight='bold'>{order}</text>"
            )
        # 名称标签
        parts.append(
            f"<text x='{cx+11:.1f}' y='{cy+4:.1f}' font-size='11' fill='#1f2937'>{w.name}</text>"
        )

    # 图例
    legend_x = width - 200
    parts.append(f"<g transform='translate({legend_x},80)'>")
    parts.append("<rect x='0' y='0' width='180' height='200' fill='white' stroke='#d1d5db' rx='4'/>")
    parts.append("<text x='10' y='20' font-size='12' font-weight='bold'>图例</text>")
    items = [
        ("● start_exit", "#16a34a"),
        ("● obstacle_*", "#ea580c"),
        ("● inspection_*", "#7c3aed"),
        ("● pick/place_*", "#0891b2"),
        ("● finish", "#db2777"),
        ("✗ 距原点 < 0.15m", "#dc2626"),
        ("── Δ < 0.5m 可疑对", "#dc2626"),
    ]
    for i, (txt, col) in enumerate(items):
        parts.append(f"<text x='10' y='{45 + i * 22}' font-size='11' fill='{col}'>{txt}</text>")
    parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description="静态分析 waypoints_FINAL.yaml")
    p.add_argument("--waypoints-yaml", default=str(RUNTIME_PATH))
    p.add_argument("--json", action="store_true", help="输出 JSON 报告")
    p.add_argument("--html", type=str, default=None,
                   help="生成浏览器可看的 SVG 文件路径(如 waypoints.html)")
    args = p.parse_args()

    path = Path(args.waypoints_yaml).expanduser().resolve()
    if not path.exists():
        print(f"[错误] 找不到 {path}", file=sys.stderr)
        return 2

    wps = load_waypoints(path)
    analysis = analyze(wps, DEFAULT_FSM_PATH)

    if args.html:
        svg = render_svg(analysis)
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Waypoints 分析</title>"
            "<style>body{font-family:ui-sans-serif,system-ui;margin:24px;max-width:1000px}"
            ".wrap{display:flex;gap:24px;align-items:flex-start}"
            ".panel{background:white;padding:16px;border:1px solid #e5e7eb;border-radius:8px}"
            "pre{font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.4}"
            "</style></head><body>"
            "<h2>Waypoints 分析报告</h2>"
            "<div class='wrap'>"
            "<div class='panel'>" + svg + "</div>"
            "<div class='panel'><h3>问题列表</h3><pre>"
        )
        # 简洁问题列表(去掉 ASCII plot 那块)
        for s in a_suspect(analysis):
            html += s + "\n"
        html += "</pre></div></div></body></html>"
        out_path = Path(args.html).expanduser().resolve()
        out_path.write_text(html, encoding="utf-8")
        print(f"已生成 {out_path}")
    elif args.json:
        print(json.dumps(report_json(analysis), ensure_ascii=False, indent=2))
    else:
        print(report_text(analysis, source_label=str(path)))
    return 0


def a_suspect(a: Analysis) -> list[str]:
    """HTML 模式下嵌入的简化问题列表。"""
    out: list[str] = []
    out.append(f"共 {len(a.waypoints)} 个航点")
    if a.origin_cluster:
        out.append("")
        out.append(f"🔴 {len(a.origin_cluster)} 个航点聚在原点 (SLAM 失跟):")
        for n in a.origin_cluster:
            out.append(f"   - {n}")
    if a.suspect_pairs:
        out.append("")
        out.append(f"🔴 {len(a.suspect_pairs)} 对可疑距离 (< 0.5m):")
        for n1, n2, d in sorted(a.suspect_pairs, key=lambda x: x[2]):
            out.append(f"   - {n1} ↔ {n2}: {d:.3f}m")
    if a.side_yaw_issues:
        out.append("")
        out.append("⚠ inspection 两侧 yaw 差:")
        for n1, n2, d in a.side_yaw_issues:
            out.append(f"   - {n1} ↔ {n2}: {d:.1f}° (期望 ≥ 90°)")
    if not (a.origin_cluster or a.suspect_pairs or a.side_yaw_issues):
        out.append("✓ 全部通过")
    return out


if __name__ == "__main__":
    sys.exit(main())
