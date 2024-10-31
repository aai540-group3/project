"""
Logging Configuration
=====================

.. module:: pipeline.utils.logging
   :synopsis: Centralized logging configuration

.. moduleauthor:: aai540-group3
"""

import sys
from pathlib import Path
from typing import Optional

import loguru
from omegaconf import DictConfig
from rich.logging import RichHandler


def setup_logging(cfg: Optional[DictConfig] = None) -> loguru._Logger:
    """
    Set up logging configuration using Loguru and Rich based on the provided configuration.

    :param cfg: Configuration object containing logging settings
    :type cfg: Optional[DictConfig]
    :return: Configured Loguru logger
    :rtype: Logger
    :raises OSError: If log file cannot be created
    """
    if not cfg or not cfg.get("enabled", True):
        loguru.logger.disable("pipeline")
        return loguru.logger

    # Remove the default Loguru handler to prevent duplicate logs
    loguru.logger.remove()

    # Set global log levels with colors
    loguru.logger.level("DEBUG", color="<cyan>{level}</cyan>")
    loguru.logger.level("INFO", color="<green>{level}</green>")
    loguru.logger.level("WARNING", color="<yellow>{level}</yellow>")
    loguru.logger.level("ERROR", color="<red>{level}</red>")
    loguru.logger.level("CRITICAL", color="<magenta>{level}</magenta>")

    # Console handler setup with Rich
    console_cfg = cfg.get("console", {})
    if console_cfg.get("enabled", True):
        if console_cfg.get("use_rich", False):
            # Integrate Rich with Loguru
            rich_handler = RichHandler(
                level=console_cfg.get("level", "DEBUG"),
                format=console_cfg.get(
                    "format",
                    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
                ),
                rich_tracebacks=True,  # Enables Rich's enhanced tracebacks
                markup=console_cfg.get("colored", True),
            )
            loguru.logger.add(rich_handler, enqueue=True)
        else:
            # Traditional Loguru console handler
            loguru.logger.add(
                sys.stdout,
                level=console_cfg.get("level", "DEBUG"),
                format=console_cfg.get(
                    "format",
                    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
                ),
                colorize=console_cfg.get("colored", True),
                enqueue=True,
            )

    # File handler setup
    file_cfg = cfg.get("file", {})
    if file_cfg.get("enabled", False):
        log_file = Path(file_cfg.get("path", "pipeline.log"))
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            loguru.logger.add(
                str(log_file),
                level=file_cfg.get("level", "INFO"),
                format=file_cfg.get(
                    "format",
                    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
                ),
                rotation=file_cfg.get("rotation", "1 day"),
                retention=file_cfg.get("retention", "7 days"),
                compression=file_cfg.get("compression", "zip"),
                encoding="utf-8",
                enqueue=True,
            )
        except OSError as e:
            raise OSError(f"Failed to create log file: {e}")

    # Set up specific loggers
    loggers_cfg = cfg.get("loggers", {})
    for logger_name, logger_settings in loggers_cfg.items():
        # Loguru does not support named loggers like the standard logging module.
        # However, you can bind context to log messages to simulate named loggers.
        # Here, we'll add a sink with a filter based on the loguru.logger name.
        loguru.logger.add(
            sys.stdout,
            level=logger_settings.get("level", "INFO"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            filter=lambda record, name=logger_name: name in record["name"],
            colorize=logger_settings.get("colored", True),
            enqueue=True,
        )

    # Exception handling to log uncaught exceptions
    def handle_exception(exc_type: type, exc_value: BaseException, exc_traceback: Optional[object]) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        loguru.logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return loguru.logger


def get_logger(name: Optional[str] = None, cfg: Optional[DictConfig] = None) -> loguru._Logger:
    """
    Get or create a logger with the specified name based on the config.

    :param name: Logger name
    :type name: Optional[str]
    :param cfg: Configuration object containing logging settings
    :type cfg: Optional[DictConfig]
    :return: Logger instance
    :rtype: loguru.Logger
    """
    setup_logging(cfg)
    if name:
        return loguru.logger.bind(name=name)
    return loguru.logger
