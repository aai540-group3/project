"""Base class for pipeline stages."""

import logging
from abc import abstractmethod
from typing import Dict

import mlflow
import wandb
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PipelineStage:
    """Base class for all pipeline stages.

    This class provides common functionality for all pipeline stages including
    configuration management, logging, and metric tracking.

    :param cfg: Configuration object containing stage parameters
    :type cfg: DictConfig

    :raises ConfigurationError: If required configuration parameters are missing
    """

    def __init__(self, cfg: DictConfig):
        """Initialize pipeline stage.

        :param cfg: Stage configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_tracking()

    def _setup_tracking(self) -> None:
        """Configure experiment tracking systems.

        Sets up MLflow and Weights & Biases tracking based on configuration.

        :raises TrackingError: If tracking system initialization fails
        """
        if self.cfg.wandb.enabled:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=self.cfg,
                tags=[self.__class__.__name__],
            )

        if self.cfg.mlflow.enabled:
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(self.cfg.experiment.name)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to configured tracking systems.

        :param metrics: Dictionary of metric names and values
        :type metrics: Dict[str, float]
        """
        if self.cfg.wandb.enabled:
            wandb.log(metrics)

        if self.cfg.mlflow.enabled:
            mlflow.log_metrics(metrics)

    @abstractmethod
    def run(self) -> None:
        """Execute pipeline stage.

        This method must be implemented by all pipeline stages.

        :raises NotImplementedError: If the stage doesn't implement this method
        """
        raise NotImplementedError
