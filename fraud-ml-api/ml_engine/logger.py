"""
Centralized logging utility for ML Engine.

- Safe for Django startup
- Safe for training & inference
- Supports console + file logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGER_CACHE = {}

# Default global log level (can be overridden)
DEFAULT_LOG_LEVEL = logging.INFO

# Optional log file (created lazily)
LOG_FILE_PATH = Path("fraudapp/ml_engine/logs/ml_engine.log")


def get_logger(
    name: str,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Returns a configured logger instance.
    Prevents duplicate handlers.

    Args:
        name (str): Logger name (usually __name__)
        level (int, optional): Logging level override

    Returns:
        logging.Logger
    """

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)

    log_level = level if level is not None else DEFAULT_LOG_LEVEL
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional but recommended)
        try:
            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE_PATH)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fail silently if file logging is not possible
            pass

        logger.propagate = False

    _LOGGER_CACHE[name] = logger
    return logger
