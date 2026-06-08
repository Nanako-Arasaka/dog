"""计时与频率控制工具。"""

from __future__ import annotations

import time


class RateLimiter:
    """固定频率循环控制器。

    用法:
        rl = RateLimiter(20.0)
        while running:
            rl.sleep()  # 保证循环频率 ≈ 20 Hz
    """

    def __init__(self, hz: float) -> None:
        self._interval = 1.0 / max(hz, 0.1)
        self._last_tick = time.perf_counter()

    def sleep(self) -> None:
        """阻塞至下一个 tick 时刻。"""
        now = time.perf_counter()
        elapsed = now - self._last_tick
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_tick = time.perf_counter()

    def reset(self) -> None:
        self._last_tick = time.perf_counter()


class Timer:
    """一次性或重复计时器。"""

    def __init__(self, duration_sec: float) -> None:
        self._duration = duration_sec
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def expired(self) -> bool:
        return self.elapsed >= self._duration

    @property
    def remaining(self) -> float:
        return max(0.0, self._duration - self.elapsed)

    def reset(self) -> None:
        self._start = time.perf_counter()
