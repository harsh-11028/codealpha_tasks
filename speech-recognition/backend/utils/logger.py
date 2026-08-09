"""
utils/logger.py — Centralized structured logging using Loguru.
"""

import sys
import os
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru logger with console + rotating file handlers."""
    # Remove default handler
    logger.remove()

    # Console handler (coloured, human-readable)
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (JSON-structured, rotating)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        serialize=True,              # JSON lines format
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logger initialized | level={log_level} | file={log_file}")


def get_logger(name: str):
    """Return a bound logger with module context."""
    return logger.bind(module=name)


# Initialize with defaults when module is first imported
_log_level = os.getenv("LOG_LEVEL", "INFO")
_log_file = os.getenv("LOG_FILE", "logs/app.log")
setup_logger(log_level=_log_level, log_file=_log_file)
