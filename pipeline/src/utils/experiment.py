# src/utils/experiment.py
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """MLflow experiment tracking wrapper."""

    def __init__(self, cfg: DictConfig, experiment_name: str):
        """Initialize experiment tracker.

        Args:
            cfg: Configuration object
            experiment_name: Name of the experiment
        """
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.run_id: Optional[str] = None
        self._setup_tracking()

    def _setup_tracking(self) -> None:
        """Setup MLflow tracking."""
        if self.cfg.tracking.enabled:
            mlflow.set_tracking_uri(self.cfg.tracking.uri)
            mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> None:
        """Start a new MLflow run."""
        if self.cfg.tracking.enabled:
            mlflow.start_run(run_name=run_name, tags=tags)
            self.run_id = mlflow.active_run().info.run_id
            logger.info(f"Started run: {self.run_id}")

            # Log configuration
            self.log_params(OmegaConf.to_container(self.cfg))

    def end_run(self) -> None:
        """End current MLflow run."""
        if self.cfg.tracking.enabled and mlflow.active_run():
            mlflow.end_run()
            logger.info(f"Ended run: {self.run_id}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if self.cfg.tracking.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if self.cfg.tracking.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log artifact to MLflow."""
        if self.cfg.tracking.enabled:
            mlflow.log_artifact(str(local_path))

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict] = None,
        code_paths: Optional[List[str]] = None,
    ) -> None:
        """Log model to MLflow."""
        if self.cfg.tracking.enabled:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                conda_env=conda_env,
                code_paths=code_paths
            )
