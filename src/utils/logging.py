"""
Structured logging configuration using Loguru.

Provides rich console output and rotating file logging with
structured JSON format for production environments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def configure_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    json_logs: bool = False,
) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Minimum logging level.
        log_file: Optional file path for persistent logs.
        json_logs: Emit JSON-structured logs (for production).
    """
    logger.remove()  # Remove default handler

    # Console handler — rich format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler — rotating with compression
    if log_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=file_format if not json_logs else "{message}",
            level=log_level,
            rotation="50 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
            serialize=json_logs,
        )

    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def get_logger(name: str) -> Any:
    """Return a named logger context."""
    return logger.bind(module=name)


__all__ = ["configure_logging", "get_logger", "logger"]
