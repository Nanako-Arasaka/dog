#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSONL event logger for integration bridge events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .schemas import now_ts


class EventLogger:
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Mapping[str, Any]) -> None:
        if not self.log_path:
            return
        data = dict(event)
        data.setdefault("logged_at", now_ts())
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")
