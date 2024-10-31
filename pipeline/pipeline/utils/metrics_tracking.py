"""
Metrics Tracking
================

.. module:: pipeline.utils.metrics_tracking
   :synopsis: Unified metrics tracking and versioning

.. moduleauthor:: aai540-group3
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import wandb
from omegaconf import DictConfig

from .logging import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """Unified metrics tracking across multiple platforms.

    :param cfg: Tracking configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize metrics tracker.

        :param cfg: Tracking configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.run_id: Optional[str] = None
        self.metrics_history: List[Dict] = []
        self._setup_tracking()

    def _setup_tracking(self) -> None:
        """Configure experiment tracking systems."""
        if self.cfg.wandb.enabled:
            import wandb

            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=self.cfg.to_container(),
                tags=[self.__class__.__name__],
                mode=os.getenv("WANDB_MODE", "online"),
                settings=wandb.Settings(
                    silent=True, disable_git=True, disable_code=True
                ),
            )

        if self.cfg.mlflow.enabled:
            import mlflow

            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(self.cfg.experiment.name)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None, commit: bool = True
    ) -> None:
        """Log metrics to all configured platforms.

        :param metrics: Metrics to log
        :type metrics: Dict[str, float]
        :param step: Training step
        :type step: Optional[int]
        :param commit: Whether to commit immediately
        :type commit: bool
        """
        timestamp = datetime.now()

        # Add to history
        self.metrics_history.append(
            {"timestamp": timestamp.isoformat(), "step": step, "metrics": metrics}
        )

        # Log to MLflow
        if self.cfg.mlflow.enabled:
            mlflow.log_metrics(metrics, step=step)

        # Log to W&B
        if self.cfg.wandb.enabled:
            wandb.log(metrics, step=step, commit=commit)

        # Save to local file
        self._save_metrics_locally(metrics, timestamp)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to tracking platforms.

        :param params: Parameters to log
        :type params: Dict[str, Any]
        """
        if self.cfg.mlflow.enabled:
            mlflow.log_params(params)

        if self.cfg.wandb.enabled:
            wandb.config.update(params)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict] = None,
        code_paths: Optional[List[str]] = None,
    ) -> None:
        """Log model artifacts to tracking platforms.

        :param model: Model to log
        :type model: Any
        :param artifact_path: Path for artifact
        :type artifact_path: str
        :param conda_env: Conda environment specification
        :type conda_env: Optional[Dict]
        :param code_paths: Additional code paths to log
        :type code_paths: Optional[List[str]]
        """
        if self.cfg.mlflow.enabled:
            mlflow.sklearn.log_model(
                model, artifact_path, conda_env=conda_env, code_paths=code_paths
            )

        if self.cfg.wandb.enabled:
            artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)

    def log_figure(self, figure: Any, path: str) -> None:
        """Log figure to tracking platforms.

        :param figure: Figure to log
        :type figure: Any
        :param path: Path for figure
        :type path: str
        """
        if self.cfg.mlflow.enabled:
            mlflow.log_figure(figure, path)

        if self.cfg.wandb.enabled:
            wandb.log({path: figure})

    def _save_metrics_locally(
        self, metrics: Dict[str, float], timestamp: datetime
    ) -> None:
        """Save metrics to local file.

        :param metrics: Metrics to save
        :type metrics: Dict[str, float]
        :param timestamp: Timestamp for metrics
        :type timestamp: datetime
        """
        metrics_dir = Path(self.cfg.paths.metrics)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = metrics_dir / "metrics_history.json"

        try:
            if metrics_file.exists():
                with metrics_file.open("r") as f:
                    history = json.load(f)
            else:
                history = []

            history.append({"timestamp": timestamp.isoformat(), "metrics": metrics})

            with metrics_file.open("w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metrics locally: {e}")

    def get_best_metric(self, metric_name: str, mode: str = "max") -> Optional[float]:
        """Get best metric value.

        :param metric_name: Name of metric
        :type metric_name: str
        :param mode: 'max' or 'min'
        :type mode: str
        :return: Best metric value
        :rtype: Optional[float]
        """
        if not self.metrics_history:
            return None

        values = [
            entry["metrics"].get(metric_name)
            for entry in self.metrics_history
            if metric_name in entry["metrics"]
        ]

        if not values:
            return None

        return max(values) if mode == "max" else min(values)

    def get_metric_history(
        self, metric_name: str
    ) -> List[Dict[str, Union[str, float]]]:
        """Get history for specific metric.

        :param metric_name: Name of metric
        :type metric_name: str
        :return: Metric history
        :rtype: List[Dict[str, Union[str, float]]]
        """
        return [
            {
                "timestamp": entry["timestamp"],
                "value": entry["metrics"][metric_name],
                "step": entry.get("step"),
            }
            for entry in self.metrics_history
            if metric_name in entry["metrics"]
        ]

    def end_run(self) -> None:
        """End tracking run."""
        if self.cfg.mlflow.enabled:
            mlflow.end_run()

        if self.cfg.wandb.enabled:
            wandb.finish()
