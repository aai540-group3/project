"""
Training Callbacks
==================

.. module:: pipeline.utils.callbacks
   :synopsis: Custom training callbacks

.. moduleauthor:: aai540-group3
"""

import time
from pathlib import Path
from typing import Dict, Optional

import GPUtil
import psutil
from omegaconf import DictConfig
from tensorflow.keras.callbacks import Callback

from .logging import get_logger

logger = get_logger(__name__)


class MetricsCallback(Callback):
    """Custom callback for metrics logging."""

    def __init__(self, cfg: DictConfig):
        """Initialize callback.

        :param cfg: Callback configuration
        :type cfg: DictConfig
        """
        super().__init__()
        self.cfg = cfg
        self.metrics_history = []
        self.start_time = None

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at epoch start.

        :param epoch: Current epoch
        :type epoch: int
        :param logs: Logs dict
        :type logs: Optional[Dict]
        """
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at epoch end.

        :param epoch: Current epoch
        :type epoch: int
        :param logs: Logs dict
        :type logs: Optional[Dict]
        """
        logs = logs or {}

        # Calculate epoch time
        if self.start_time:
            logs["epoch_time"] = time.time() - self.start_time

        # Log metrics
        self.metrics_history.append({"epoch": epoch, "metrics": logs})

        # Log to configured trackers
        if self.cfg.wandb.enabled:
            import wandb

            wandb.log(logs, step=epoch)

        if self.cfg.mlflow.enabled:
            import mlflow

            mlflow.log_metrics(logs, step=epoch)


class PerformanceMonitorCallback(Callback):
    """Monitor system performance during training."""

    def __init__(self, cfg: DictConfig):
        """Initialize callback.

        :param cfg: Callback configuration
        :type cfg: DictConfig
        """
        super().__init__()
        self.cfg = cfg
        self.performance_logs = []

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at batch end.

        :param batch: Current batch
        :type batch: int
        :param logs: Logs dict
        :type logs: Optional[Dict]
        """
        if batch % self.cfg.log_every_n_steps == 0:
            performance = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "batch": batch,
            }

            # GPU stats if available
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    performance.update(
                        {f"gpu_{i}_load": gpu.load, f"gpu_{i}_memory": gpu.memoryUtil}
                    )
            except Exception:
                pass

            self.performance_logs.append(performance)


class ModelCheckpointCallback(Callback):
    """Enhanced model checkpoint callback."""

    def __init__(
        self,
        cfg: DictConfig,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ):
        """Initialize callback.

        :param cfg: Callback configuration
        :type cfg: DictConfig
        :param filepath: Path to save checkpoints
        :type filepath: str
        :param monitor: Metric to monitor
        :type monitor: str
        :param save_best_only: Whether to save only best models
        :type save_best_only: bool
        :param mode: One of {'min', 'max'}
        :type mode: str
        """
        super().__init__()
        self.cfg = cfg
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at epoch end.

        :param epoch: Current epoch
        :type epoch: int
        :param logs: Logs dict
        :type logs: Optional[Dict]
        """
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.save_best_only:
            if (self.mode == "min" and current < self.best_value) or (
                self.mode == "max" and current > self.best_value
            ):
                self.best_value = current
                self._save_model(epoch, logs)
        else:
            self._save_model(epoch, logs)

    def _save_model(self, epoch: int, logs: Dict) -> None:
        """Save model checkpoint.

        :param epoch: Current epoch
        :type epoch: int
        :param logs: Logs dict
        :type logs: Dict
        """
        try:
            filepath = str(self.filepath).format(epoch=epoch, **logs)
            self.model.save(filepath)
            logger.info(f"Saved checkpoint: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
