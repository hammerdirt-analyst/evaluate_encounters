"""
logging_config.py
Author: Roger Erismann

Purpose:
---------
Provides a centralized utility to create and configure Python loggers for consistent logging across modules.

Key Features:
-------------
- Unified logger setup for both console and file output
- Ensures no duplicate handlers are attached to the same logger
- Automatically creates log directories if they do not exist

Function:
---------
- get_logger(name, level=INFO, to_console=True, to_file=None, ...):
    → Returns a configured logger instance for immediate use.

Usage Example:
--------------
from logging_config import get_logger

logger = get_logger(__name__, to_file='logs/app.log')
logger.info("Logger initialized.")
"""

import logging
from typing import Optional
import os

def get_logger(
    name: str,
    level: int = logging.INFO,
    to_console: bool = True,
    to_file: Optional[str] = None,
    file_mode: str = 'a',
    fmt: str = '[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s',
    datefmt: str = '%Y-%m-%d %H:%M:%S'
) -> logging.Logger:
    """
    Returns a configured logger.

    Args:
        name: Unique logger name (usually __name__ from the module).
        level: Logging level (INFO, DEBUG, ERROR, etc.).
        to_console: If True, logs to stdout.
        to_file: If provided, logs to this file path.
        file_mode: 'a' = append, 'w' = overwrite file.
        fmt: Format string for logs.
        datefmt: Date formatting for timestamp.

    Returns:
        A ready-to-use logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(fmt, datefmt)

        if to_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if to_file:
            log_dir = os.path.dirname(to_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(to_file, mode=file_mode)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
