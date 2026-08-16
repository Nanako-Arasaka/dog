from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ObstacleZoneRect:
    """障碍区长方形, planner 2D 平面坐标 (= ORB 世界系 x-z 地面平面)。

    xmin/xmax → 世界 x (横向); ymin/ymax → 世界 z (前进方向);
    zmin/zmax → 世界 y (竖直, 仅供 contains(x,y,z) 三参调用时生效)。
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float = float("-inf")
    zmax: float = float("inf")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObstacleZoneRect":
        return cls(
            xmin=float(data["xmin"]),
            xmax=float(data["xmax"]),
            ymin=float(data["ymin"]),
            ymax=float(data["ymax"]),
            zmin=float(data.get("zmin", float("-inf"))),
            zmax=float(data.get("zmax", float("inf"))),
        )

    def contains(self, x: float, y: float, z: float | None = None, margin: float = 0.0) -> bool:
        if not (
            self.xmin + margin <= x <= self.xmax - margin
            and self.ymin + margin <= y <= self.ymax - margin
        ):
            return False
        if z is not None:
            return self.zmin <= z <= self.zmax
        return True


def load_map_config(path: str | Path) -> tuple[list[tuple[float, float]], ObstacleZoneRect]:
    """Load the tiny JSON-compatible YAML map used by the local planner."""
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _parse_global_path(data.get("global_path", [])), ObstacleZoneRect.from_mapping(data["obstacle_zone_rect"])


def _parse_global_path(raw_path: Sequence[Any]) -> list[tuple[float, float]]:
    path: list[tuple[float, float]] = []
    for item in raw_path:
        if isinstance(item, Mapping):
            path.append((float(item["x"]), float(item["y"])))
        else:
            x, y = item[:2]
            path.append((float(x), float(y)))
    return path
