"""
Logging Configuration
==================

.. module:: pipeline.utils.logging
   :synopsis: Centralized logging configuration

.. moduleauthor:: aai540-group3
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

class ColoredFormatter(logging.Formatter):
    """Custom formatter providing colored output for different log levels.

    :param fmt: Log message format
    :type fmt: str
    :param datefmt: Date format
    :type datefmt: str
    """

    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[91m',
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with appropriate colors.

        :param record: Log record to format
        :type record: logging.LogRecord
        :return: Formatted log message
        :rtype: str
        """
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            record.levelname = (
                f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}"
                f"{record.levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)

def setup_logging(
    cfg: Optional[DictConfig] = None,
    name: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Set up logging configuration with console and file handlers.

    :param cfg: Configuration object containing logging settings
    :type cfg: Optional[DictConfig]
    :param name: Logger name
    :type name: Optional[str]
    :param log_file: Path to log file
    :type log_file: Optional[Path]
    :return: Configured logger
    :rtype: logging.Logger
    :raises OSError: If log file cannot be created
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = logging.DEBUG if cfg and cfg.get('debug', False) else logging.INFO
    logger.setLevel(log_level)

    # Console handler setup
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler setup
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(file_handler)
        except OSError as e:
            raise OSError(f"Failed to create log file: {e}")

    logger.propagate = False
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger with the specified name.

    :param name: Logger name
    :type name: Optional[str]
    :return: Logger instance
    :rtype: logging.Logger
    """
    return setup_logging(name=name)
