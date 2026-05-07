"""结构化日志，中文友好。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """
    初始化日志系统。

    返回根 logger，所有模块通过 logging.getLogger(__name__) 使用。
    """
    logger = logging.getLogger("my-agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(_ChineseFormatter())
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(_ChineseFormatter(include_time=True))
        logger.addHandler(file_handler)

    return logger


class _ChineseFormatter(logging.Formatter):
    """中文友好日志格式。"""

    def __init__(self, include_time: bool = False):
        if include_time:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        else:
            fmt = "[%(levelname)s] %(name)s: %(message)s"
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def formatException(self, ei) -> str:
        """异常信息保持原始英文，技术信息不翻译。"""
        return super().formatException(ei)
