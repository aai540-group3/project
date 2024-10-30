"""Experiment tracking utilities for ML pipeline.

This module provides experiment tracking functionality using MLflow, handling
parameter logging, metrics tracking, and artifact management.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
from omegaconf import DictConfig


@dataclass
class ExperimentTracker:
    """MLflow experiment tracking implementation.

    :param cfg: Configuration object
    :type cfg: DictConfig
    :param experiment_name: Name of the experiment
    :type experiment_name: str
    :param run_id: Current run ID, defaults to None
    :type run_id: str, optional
    """

    cfg: DictConfig
    experiment_name: str
    run_id: Optional[str] = None
    _active_run: Optional[mlflow.ActiveRun] = None

    def __post_init__(self):
        """Initialize MLflow configuration."""
        if self.cfg.tracking.uri:
            mlflow.set_tracking_uri(self.cfg.tracking.uri)

        if self.cfg.registry.uri:
            mlflow.set_registry_uri(self.cfg.registry.uri)

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Start a new MLflow run.

        :param run_name: Name for the run, defaults to None
        :type run_name: str, optional
        :param tags: Run tags, defaults to None
        :type tags: Dict[str, str], optional
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self._active_run.info.run_id

    def end_run(self) -> None:
        """End current MLflow run."""
        if self._active_run:
            mlflow.end_run()
            self._active_run = None
            self.run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        :param params: Parameters to log
        :type params: Dict[str, Any]
        """
        if self._active_run:
            mlflow.log_params(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow.

        :param metrics: Metrics to log
        :type metrics: Dict[str, float]
        :param step: Step number, defaults to None
        :type step: int, optional
        """
        if self._active_run:
            mlflow.log_metrics(metrics, step=step)

    def log_artifacts(self, local_dir: Union[str, Path]) -> None:
        """Log artifacts to MLflow.

        :param local_dir: Directory containing artifacts
        :type local_dir: Union[str, Path]
        """
        if self._active_run:
            mlflow.log_artifacts(str(local_dir))

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict] = None,
        code_paths: Optional[list] = None,
    ) -> None:
        """Log ML model to MLflow.

        :param model: Model to log
        :type model: Any
        :param artifact_path: Artifact path
        :type artifact_path: str
        :param conda_env: Conda environment, defaults to None
        :type conda_env: Dict, optional
        :param code_paths: Additional code paths, defaults to None
        :type code_paths: list, optional
        """
        if self._active_run:
            mlflow.sklearn.log_model(
                model, artifact_path, conda_env=conda_env, code_paths=code_paths
            )
