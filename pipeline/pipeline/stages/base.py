"""
Base Pipeline Stage
===================

.. module:: pipeline.stages.base
   :synopsis: Base class for pipeline stages with robust tracking handling, including DVC Live.

.. moduleauthor:: aai540-group3
"""

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import wandb
from dvclive import Live
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from pipeline.conf import Config


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
            f"Available stages: {available_stages}"
        )
        super().__init__(self.message)


class PipelineStage(ABC):
    """Base class for all pipeline stages, handling configuration, tracking, error handling, and logging."""

    def __init__(self):
        """Initialize base stage with configuration and setup tracking."""
        # Start time tracking
        self.start_time = time.time()
        logger.debug("Starting PipelineStage initialization.")

        # Explicit log to check return
        loaded_config = self.load_config()
        logger.debug(f"Config.load() returned: {loaded_config}")
        self.cfg = loaded_config
        logger.debug(f"Assigned self.cfg: {self.cfg}")

        if self.cfg is None:
            logger.error("Configuration failed to load. 'self.cfg' is None.")
            raise ValueError("Configuration is missing or failed to load.")
        if self.cfg is None:
            logger.error("Configuration failed to load. 'self.cfg' is None.")
            raise ValueError("Configuration is missing or failed to load.")

        self.stage_name = self.__class__.__name__.lower().replace("stage", "")
        logger.info(f"Initializing '{self.stage_name}' stage with configuration.")

        # Check `self.cfg.pipeline` and `self.cfg.pipeline.stages`
        if not hasattr(self.cfg, "pipeline") or not hasattr(self.cfg.pipeline, "stages"):
            logger.error("Configuration 'pipeline' or 'pipeline.stages' is missing.")
            raise ValueError(
                "Configuration 'pipeline' or 'pipeline.stages' section is missing or improperly formatted."
            )

        # Retrieve stage-specific config and initialize tracking
        self.stage_config = self._get_stage_config()
        self.tracking_initialized = False  # Tracking setup flag
        self.live = None  # Placeholder for DVC Live tracking if enabled

    @staticmethod
    def load_config() -> Optional[DictConfig]:
        """Load configuration using Config class."""
        try:
            config = Config.load()
            logger.debug(f"Config.load() returned: {type(config)}")
            if config:
                logger.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None

    def _get_stage_config(self) -> DictConfig:
        """Retrieve stage-specific configuration."""
        logger.debug(f"Attempting to retrieve configuration for stage '{self.stage_name}' from pipeline.stages.")
        try:
            # Log available stages for clarity
            available_stages = self.cfg.pipeline.stages.keys()
            logger.debug(f"Available stages in configuration: {available_stages}")

            stage_config_node = self.cfg.pipeline.stages[self.stage_name]
            logger.debug(
                f"Stage configuration for '{self.stage_name}': {OmegaConf.to_container(stage_config_node, resolve=True)}"
            )
            return stage_config_node
        except KeyError as e:
            available_stages = list(self.cfg.pipeline.stages.keys())
            logger.error(f"Configuration for stage '{self.stage_name}' not found. Available stages: {available_stages}")
            raise StageConfigNotFoundError(self.stage_name, available_stages) from e

    def execute(self) -> None:
        """Run the pipeline stage with centralized error handling and status logging."""
        logger.debug(f"Starting execution of '{self.stage_name}'")
        success = False

        try:
            self.run()
            success = True
        except Exception as e:
            logger.error(f"{self.stage_name.capitalize()} failed: {str(e)}")
            self.log_metrics({f"{self.stage_name}_error": str(e)})  # Log error for tracking
            raise RuntimeError(f"{self.stage_name.capitalize()} setup failed: {str(e)}") from e
        finally:
            self._log_final_status(success)
            logger.debug(
                f"Completed execution of '{self.stage_name}' with status: {'success' if success else 'failure'}"
            )

    @abstractmethod
    def run(self) -> None:
        """Abstract method to execute the specific logic of each stage."""
        pass

    def _should_enable_tracking(self) -> bool:
        """Determine if tracking is enabled for this stage.

        :return: Whether tracking should be enabled
        :rtype: bool
        """
        return self.stage_config.get("tracking", {}).get("enabled", False)

    def _setup_tracking(self) -> None:
        """Initialize tracking systems (MLflow, WandB, DVC Live) if configured."""
        try:
            run_name = f"{self.stage_name}_{self.cfg.experiment.name}"
            stage_tracking = self.stage_config.get("tracking", {})

            # Initialize WandB if enabled
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

            # Initialize MLflow if enabled
            if stage_tracking.get("mlflow", {}).get("enabled", False):
                mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
                mlflow.set_experiment(self.cfg.mlflow.experiment_name)
                mlflow.start_run(run_name=run_name)

            # Initialize DVC Live if enabled
            if stage_tracking.get("dvclive", {}).get("enabled", False):
                metrics_dir = os.path.join(self.cfg.paths.artifacts, "metrics")
                self.live = Live(dir=metrics_dir, dvcyaml=False)
                logger.info("DVC Live initialized for real-time metric tracking.")

            self.tracking_initialized = True
            logger.debug(f"Tracking initialized for {self.stage_name}")

        except Exception as e:
            logger.warning(f"Failed to initialize tracking: {e}")
            self.tracking_initialized = False

    def _cleanup_tracking(self) -> None:
        """Cleanup tracking systems (MLflow, WandB, DVC Live)."""
        if not self.tracking_initialized:
            return

        try:
            if wandb.run is not None:
                wandb.finish()

            if mlflow.active_run():
                mlflow.end_run()

            if self.live:
                self.live.end()

        except Exception as e:
            logger.warning(f"Error during tracking cleanup: {e}")

    def _log_final_status(self, success: bool) -> None:
        """Log the final status of the pipeline stage and track success/failure.

        :param success: Indicates if the stage completed successfully
        :type success: bool
        """
        duration = time.time() - self.start_time
        logger.info(f"{self.stage_name.capitalize()} completed in {duration:.2f} seconds with success: {success}")

        # Log final status metrics
        metrics = {
            f"{self.stage_name}_setup_success": int(success),
            f"{self.stage_name}_duration_seconds": duration,
        }
        self.log_metrics(metrics)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to tracking systems (MLflow, WandB, DVC Live) if initialized.

        :param metrics: Metrics to log
        :type metrics: Dict[str, Any]
        """
        if not self.tracking_initialized:
            return

        stage_tracking = self.stage_config.get("tracking", {})
        try:
            # Log metrics to WandB, MLflow, and DVC Live if enabled
            if stage_tracking.get("wandb", {}).get("enabled", False):
                wandb.log(metrics)
            if stage_tracking.get("mlflow", {}).get("enabled", False):
                mlflow.log_metrics(metrics)
            if stage_tracking.get("dvclive", {}).get("enabled", False) and self.live:
                for key, value in metrics.items():
                    self.live.log(key, value)

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to tracking systems (MLflow, WandB, DVC Live) if initialized.

        :param local_path: Path to the local artifact file
        :type local_path: str
        :param artifact_path: Optional path within the artifact storage
        :type artifact_path: Optional[str]
        """
        if not self.tracking_initialized:
            return

        stage_tracking = self.stage_config.get("tracking", {})
        try:
            # Log artifacts to WandB and MLflow
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

            # Save artifact with DVC Live
            if stage_tracking.get("dvclive", {}).get("enabled", False) and self.live:
                self.live.log_artifact(local_path)

        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def get_path(self, key: str) -> Path:
        """Get path from configuration.

        :param key: Path key
        :type key: str
        :return: Resolved path
        :rtype: Path
        :raises KeyError: If the specified key does not exist in paths configuration.
        """
        try:
            path = Path(self.cfg.paths[key])
            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
            return path
        except KeyError as e:
            raise KeyError(f"Key '{key}' not found in paths configuration.") from e
