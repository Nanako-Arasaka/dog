"""国赛任务入口。

加载配置 → 组装 DI 容器 → 启停生命周期。
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path

from app.config import load_app_config
from app.container import AppContainer


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "robot_config.json"
    cfg = load_app_config(str(cfg_path))

    container = AppContainer(cfg)

    running = True

    def _stop_handler(*_: object) -> None:
        nonlocal running
        running = False
        logging.info("收到停止信号，准备退出...")

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    container.dog.start_background_loops()
    container.mission.start()

    try:
        while running and not container.mission.is_finished:
            container.mission.tick()
            container.dog.sleep_for_main_tick()
    finally:
        container.mission.stop()
        container.dog.stop_background_loops()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
