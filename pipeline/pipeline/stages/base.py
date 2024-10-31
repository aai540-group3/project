"""
Base Pipeline Stage
===================

.. module:: pipeline.stages.base
   :synopsis: Base class for pipeline stages with robust tracking handling

.. moduleauthor:: aai540-group3
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List

import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf

from ..utils.logging import get_logger

logger = get_logger(__name__)


class StageConfigNotFoundError(Exception):
    """Exception raised when a stage-specific configuration is missing.

    Attributes:
        stage_name (str): Name of the missing stage.
        available_stages (List[str]): List of available stage names.
        message (str): Explanation of the error.
    """

    def __init__(self, stage_name: str, available_stages: List[str]):
        self.stage_name = stage_name
        self.available_stages = available_stages
        self.message = (
            f"Stage configuration for '{self.stage_name}' not found in 'pipeline.stages'. "
            f"Available stages: {self.available_stages}"
        )
        super().__init__(self.message)


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    def __init__(self, cfg: DictConfig):
        """Initialize pipeline stage.

        :param cfg: Stage configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        logger.debug("Initialized logger for PipelineStage.")

        self.stage_name = self.__class__.__name__.lower().replace("stage", "")
        logger.debug(f"Stage name set to: {self.stage_name}")

        self.tracking_initialized = False
        logger.debug("Tracking initialized set to False.")

        # Get stage-specific config
        self.stage_config = self._get_stage_config()

        # Log the entire configuration for debugging
        logger.debug(f"Full configuration (cfg):\n{OmegaConf.to_yaml(self.cfg)}")

        # Setup tracking if enabled
        if self._should_enable_tracking():
            logger.debug("Tracking is enabled. Proceeding to setup tracking.")
            self._setup_tracking()
        else:
            logger.debug("Tracking is disabled. Skipping setup.")

    def _get_stage_config(self) -> DictConfig:
        """Get stage-specific configuration.

        :return: Stage configuration as an OmegaConf object
        :rtype: DictConfig
        :raises StageConfigNotFoundError: If the stage-specific configuration is not found.
        """
        try:
            # Attempt to retrieve the stage-specific configuration
            stage_config_node = self.cfg.pipeline.stages[self.stage_name]
            logger.debug(f"Retrieved configuration for stage '{self.stage_name}':\n{OmegaConf.to_yaml(stage_config_node)}")
            return stage_config_node
        except KeyError as e:
            # Retrieve available stages for better error messaging
            available_stages = list(self.cfg.pipeline.stages.keys()) if hasattr(self.cfg.pipeline, 'stages') else []
            error_message = (
                f"Stage configuration for '{self.stage_name}' not found in 'pipeline.stages'. "
                f"Available stages: {available_stages}"
            )
            logger.error(error_message)
            raise StageConfigNotFoundError(self.stage_name, available_stages) from e

    def _should_enable_tracking(self) -> bool:
        """Check if tracking should be enabled.

        :return: Whether tracking should be enabled
        :rtype: bool
        """
        # Check stage-specific tracking config
        stage_tracking = self.stage_config.get("tracking", {})
        return stage_tracking.get("enabled", False)

    def _setup_tracking(self) -> None:
        """Setup tracking systems."""
        try:
            run_name = f"{self.stage_name}_{self.cfg.experiment.name}"
            stage_tracking = self.stage_config.get("tracking", {})

            if stage_tracking.get("wandb", {}).get("enabled", False):
                wandb.init(
                    project=self.cfg.wandb.project,
                    entity=self.cfg.wandb.entity,
                    name=run_name,
                    group=self.cfg.experiment.name,
                    job_type=self.stage_name,
                    config=self.stage_config,
                    mode=self.cfg.wandb.mode,
                    tags=self.cfg.wandb.tags + [self.stage_name],
                    reinit=True,
                )

            if stage_tracking.get("mlflow", {}).get("enabled", False):
                mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
                mlflow.set_experiment(self.cfg.mlflow.experiment_name)
                mlflow.start_run(run_name=run_name)

            self.tracking_initialized = True
            logger.debug(f"Tracking initialized for {self.stage_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize tracking: {e}")
            self.tracking_initialized = False

    def _cleanup_tracking(self) -> None:
        """Cleanup tracking systems."""
        if not self.tracking_initialized:
            return

        try:
            if wandb.run is not None:
                wandb.finish()

            if mlflow.active_run():
                mlflow.end_run()

        except Exception as e:
            logger.warning(f"Error during tracking cleanup: {e}")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics if tracking is initialized.

        :param metrics: Metrics to log
        :type metrics: Dict[str, Any]
        """
        if not self.tracking_initialized:
            return

        stage_tracking = self.stage_config.get("tracking", {})

        try:
            if stage_tracking.get("wandb", {}).get("enabled", False):
                wandb.log(metrics)

            if stage_tracking.get("mlflow", {}).get("enabled", False):
                mlflow.log_metrics(metrics)

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log artifact if tracking is initialized.

        :param local_path: Path to artifact
        :type local_path: str
        :param artifact_path: Optional path within artifact storage
        :type artifact_path: Optional[str]
        """
        if not self.tracking_initialized:
            return

        stage_tracking = self.stage_config.get("tracking", {})

        try:
            if stage_tracking.get("wandb", {}).get("enabled", False):
                artifact = wandb.Artifact(
                    name=Path(local_path).name,
                    type=self.stage_name,
                    description=f"Artifact from {self.stage_name}",
                )
                artifact.add_file(local_path)
                wandb.log_artifact(artifact)

            if stage_tracking.get("mlflow", {}).get("enabled", False):
                mlflow.log_artifact(local_path, artifact_path)

        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def get_path(self, key: str) -> Path:
        """Get path from configuration.

        :param key: Path key
        :type key: str
        :return: Resolved path
        :rtype: Path
        """
        path = Path(self.cfg.paths[key])
        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
        return path

    @abstractmethod
    def run(self) -> None:
        """Execute pipeline stage."""
        raise NotImplementedError("Pipeline stages must implement run()")
