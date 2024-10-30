import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import optuna
from dvclive import Live
from optuna.trial import Trial
from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)


class DVCCallback(Callback):
    """DVC tracking callback for Keras models."""

    def __init__(self, metrics_dir: str, model_name: str, live: Optional[Live] = None):
        super().__init__()
        self.live = live or Live(metrics_dir)
        self.model_name = model_name

    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """Log metrics at end of epoch."""
        if logs:
            for metric, value in logs.items():
                self.live.log_metric(f"{self.model_name}/{metric}", value, step=epoch)

    def on_train_end(self, logs: Dict = None) -> None:
        """Save artifacts at end of training."""
        self.live.end()


class OptunaCallback(Callback):
    """Optuna pruning callback for Keras models."""

    def __init__(self, trial: Trial, monitor: str = "val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """Report values for pruning."""
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        # Report value to Optuna for pruning
        self.trial.report(current_value, epoch)

        # Handle pruning based on Optuna's suggestion
        if self.trial.should_prune():
            raise optuna.TrialPruned()
