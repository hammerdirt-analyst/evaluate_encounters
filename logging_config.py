# polygon_search/logging_config.py

import logging
from typing import Optional

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
            file_handler = logging.FileHandler(to_file, mode=file_mode)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
