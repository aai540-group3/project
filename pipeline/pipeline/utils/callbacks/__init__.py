"""
Callback System
===============

.. module:: pipeline.utils.callbacks
   :synopsis: Event handling and callback system

.. moduleauthor:: aai540-group3
"""

from .manager import BaseCallback, CallbackEvent, CallbackManager, LoggingCallback, MetricsCallback, SlackCallback

__all__ = [
    "CallbackEvent",
    "BaseCallback",
    "LoggingCallback",
    "MetricsCallback",
    "SlackCallback",
    "CallbackManager",
]
