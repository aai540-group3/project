"""
Callback System
===============

.. module:: pipeline.utils.callbacks.manager
   :synopsis: Event handling and callback system for pipeline stages

.. moduleauthor:: aai540-group3
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from loguru import logger
from omegaconf import DictConfig


class CallbackEvent:
    """Pipeline event representation."""

    def __init__(
        self, event_type: str, stage_name: str, timestamp: float = None, data: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline event.

        :param event_type: Type of event
        :type event_type: str
        :param stage_name: Name of the stage
        :type stage_name: str
        :param timestamp: Event timestamp
        :type timestamp: float
        :param data: Additional event data
        :type data: Optional[Dict[str, Any]]
        """
        self.event_type = event_type
        self.stage_name = stage_name
        self.timestamp = timestamp or time.time()
        self.data = data or {}


class BaseCallback(ABC):
    """Abstract base class for callbacks."""

    @abstractmethod
    def on_stage_start(self, event: CallbackEvent) -> None:
        """Handle stage start event.

        :param event: Event details
        :type event: CallbackEvent
        """
        pass

    @abstractmethod
    def on_stage_end(self, event: CallbackEvent) -> None:
        """Handle stage end event.

        :param event: Event details
        :type event: CallbackEvent
        """
        pass

    @abstractmethod
    def on_stage_error(self, event: CallbackEvent) -> None:
        """Handle stage error event.

        :param event: Event details
        :type event: CallbackEvent
        """
        pass


class LoggingCallback(BaseCallback):
    """Callback for logging events."""

    def on_stage_start(self, event: CallbackEvent) -> None:
        """Log stage start.

        :param event: Event details
        :type event: CallbackEvent
        """
        logger.info(
            f"Stage '{event.stage_name}' started at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}"
        )

    def on_stage_end(self, event: CallbackEvent) -> None:
        """Log stage end.

        :param event: Event details
        :type event: CallbackEvent
        """
        duration = time.time() - event.timestamp
        status = event.data.get("status", "completed")
        logger.info(f"Stage '{event.stage_name}' {status} in {duration:.2f} seconds")

    def on_stage_error(self, event: CallbackEvent) -> None:
        """Log stage error.

        :param event: Event details
        :type event: CallbackEvent
        """
        error = event.data.get("error", "Unknown error")
        logger.error(f"Stage '{event.stage_name}' failed: {error}")


class MetricsCallback(BaseCallback):
    """Callback for collecting stage metrics."""

    def __init__(self):
        """Initialize metrics callback."""
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}

    def on_stage_start(self, event: CallbackEvent) -> None:
        """Record stage start metrics.

        :param event: Event details
        :type event: CallbackEvent
        """
        if event.stage_name not in self.metrics:
            self.metrics[event.stage_name] = []

        self.metrics[event.stage_name].append({"event": "start", "timestamp": event.timestamp, "data": event.data})

    def on_stage_end(self, event: CallbackEvent) -> None:
        """Record stage end metrics.

        :param event: Event details
        :type event: CallbackEvent
        """
        duration = time.time() - event.timestamp
        self.metrics[event.stage_name].append(
            {"event": "end", "timestamp": event.timestamp, "duration": duration, "data": event.data}
        )

    def on_stage_error(self, event: CallbackEvent) -> None:
        """Record stage error metrics.

        :param event: Event details
        :type event: CallbackEvent
        """
        self.metrics[event.stage_name].append(
            {"event": "error", "timestamp": event.timestamp, "error": str(event.data.get("error")), "data": event.data}
        )


class SlackCallback(BaseCallback):
    """Callback for Slack notifications."""

    def __init__(self, webhook_url: str):
        """Initialize Slack callback.

        :param webhook_url: Slack webhook URL
        :type webhook_url: str
        """
        self.webhook_url = webhook_url

    def _send_slack_message(self, message: str, color: str = "good") -> None:
        """Send message to Slack.

        :param message: Message to send
        :type message: str
        :param color: Message color/severity
        :type color: str
        """
        try:
            import requests

            payload = {"attachments": [{"color": color, "text": message, "ts": time.time()}]}
            requests.post(self.webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def on_stage_start(self, event: CallbackEvent) -> None:
        """Send stage start notification.

        :param event: Event details
        :type event: CallbackEvent
        """
        message = f"Stage '{event.stage_name}' started"
        self._send_slack_message(message, "good")

    def on_stage_end(self, event: CallbackEvent) -> None:
        """Send stage end notification.

        :param event: Event details
        :type event: CallbackEvent
        """
        duration = time.time() - event.timestamp
        message = (
            f"Stage '{event.stage_name}' completed in {duration:.2f} seconds\n"
            f"Status: {event.data.get('status', 'completed')}"
        )
        self._send_slack_message(message, "good")

    def on_stage_error(self, event: CallbackEvent) -> None:
        """Send stage error notification.

        :param event: Event details
        :type event: CallbackEvent
        """
        error = event.data.get("error", "Unknown error")
        message = f"Stage '{event.stage_name}' failed: {error}"
        self._send_slack_message(message, "danger")


class CallbackManager:
    """Manager for handling multiple callbacks."""

    def __init__(self, cfg: Optional[DictConfig] = None):
        """Initialize callback manager.

        :param cfg: Optional callback configuration
        :type cfg: Optional[DictConfig]
        """
        self.callbacks: List[BaseCallback] = []
        self._registered_types: Set[str] = set()

        # Initialize default callbacks
        self.add_callback(LoggingCallback())
        self.add_callback(MetricsCallback())

        # Initialize configured callbacks
        if cfg and cfg.get("callbacks"):
            self._initialize_configured_callbacks(cfg.callbacks)

    def _initialize_configured_callbacks(self, callback_cfg: DictConfig) -> None:
        """Initialize callbacks from configuration.

        :param callback_cfg: Callback configuration
        :type callback_cfg: DictConfig
        """
        if callback_cfg.get("slack", {}).get("enabled", False):
            webhook_url = callback_cfg.slack.webhook_url
            if webhook_url:
                self.add_callback(SlackCallback(webhook_url))

    def add_callback(self, callback: BaseCallback) -> None:
        """Add callback to manager.

        :param callback: Callback to add
        :type callback: BaseCallback
        """
        callback_type = callback.__class__.__name__
        if callback_type not in self._registered_types:
            self.callbacks.append(callback)
            self._registered_types.add(callback_type)
            logger.debug(f"Registered callback: {callback_type}")

    def notify_stage_start(self, stage_name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Notify callbacks of stage start.

        :param stage_name: Name of the stage
        :type stage_name: str
        :param data: Additional event data
        :type data: Optional[Dict[str, Any]]
        """
        event = CallbackEvent("start", stage_name, data=data)
        for callback in self.callbacks:
            try:
                callback.on_stage_start(event)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_stage_start: {e}")

    def notify_stage_end(self, stage_name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Notify callbacks of stage end.

        :param stage_name: Name of the stage
        :type stage_name: str
        :param data: Additional event data
        :type data: Optional[Dict[str, Any]]
        """
        event = CallbackEvent("end", stage_name, data=data)
        for callback in self.callbacks:
            try:
                callback.on_stage_end(event)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_stage_end: {e}")

    def notify_stage_error(self, stage_name: str, error: Exception) -> None:
        """Notify callbacks of stage error.

        :param stage_name: Name of the stage
        :type stage_name: str
        :param error: The error that occurred
        :type error: Exception
        """
        event = CallbackEvent("error", stage_name, data={"error": str(error)})
        for callback in self.callbacks:
            try:
                callback.on_stage_error(event)
            except Exception as e:
                logger.error(f"Error in {callback.__class__.__name__}.on_stage_error: {e}")

    def get_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get collected metrics.

        :return: Collected metrics by stage
        :rtype: Dict[str, List[Dict[str, Any]]]
        """
        for callback in self.callbacks:
            if isinstance(callback, MetricsCallback):
                return callback.metrics
        return {}
