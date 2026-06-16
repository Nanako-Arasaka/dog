"""巡检识别闭环本地入口。

完整 TCP 输出由项目根目录的 vision_server.py 提供；此入口只验证配置和
RemotePerceptionGateway / AudioFileSpeaker 可被装配。
"""

from __future__ import annotations

import logging
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

    _container = AppContainer(cfg)
    logging.info("巡检识别闭环组件装配完成。TCP 输出请启动 vision_server.py。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
