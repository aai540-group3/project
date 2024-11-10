"""
Tracking Management
===================

.. module:: pipeline.utils.tracking.manager
   :synopsis: Factory and managers for experiment tracking systems

.. moduleauthor:: aai540-group3
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from omegaconf import DictConfig


class BaseTrackingManager(ABC):
    """Abstract base class for tracking managers."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class MLflowManager(BaseTrackingManager):
    """MLflow tracking manager."""

    def __init__(self, cfg: DictConfig, experiment_name: str):
        """Initialize MLflow manager."""
        import mlflow

        self.cfg = cfg
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        import mlflow

        mlflow.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to MLflow."""
        import mlflow

        mlflow.log_artifact(local_path, artifact_path)

    def cleanup(self) -> None:
        """End MLflow run."""
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()


class WandbManager(BaseTrackingManager):
    """Weights & Biases tracking manager."""

    def __init__(self, cfg: DictConfig, experiment_name: str):
        """Initialize W&B manager."""
        import wandb

        self.cfg = cfg
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=experiment_name,
            config=cfg,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        import wandb

        wandb.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to W&B."""
        import wandb

        wandb.config.update(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to W&B."""
        import wandb

        artifact = wandb.Artifact(name=Path(local_path).name, type="output")
        artifact.add_file(local_path)
        wandb.log_artifact(artifact)

    def cleanup(self) -> None:
        """Finish W&B run."""
        import wandb

        if wandb.run:
            wandb.finish()


class DVCManager(BaseTrackingManager):
    """DVC tracking manager."""

    def __init__(self, cfg: DictConfig, experiment_name: str):
        """Initialize DVC manager."""
        from dvclive import Live

        self.live = Live(
            dir=cfg.dvc.dir,
            resume=False,
            report=cfg.dvc.report,
            save_dvc_exp=True,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to DVC."""
        for name, value in metrics.items():
            self.live.log_metric(name, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to DVC."""
        self.live.log_params(params)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact to DVC."""
        self.live.log_artifact(local_path)

    def cleanup(self) -> None:
        """End DVC tracking."""
        if hasattr(self, "live"):
            self.live.end()


class TrackingManagerFactory:
    """Factory for creating tracking managers."""

    @staticmethod
    def create_managers(cfg: DictConfig, stage_name: str, stage_config: DictConfig) -> List[BaseTrackingManager]:
        """Create tracking managers based on configuration.

        Args:
            cfg: Global configuration
            stage_name: Name of the current stage
            stage_config: Stage-specific configuration

        Returns:
            List of tracking managers
        """
        managers = []
        experiment_name = f"{cfg.experiment.name}_{stage_name}"

        try:
            # Initialize MLflow if enabled
            if stage_config.tracking.mlflow.enabled:
                managers.append(MLflowManager(cfg, experiment_name))

            # Initialize W&B if enabled
            if stage_config.tracking.wandb.enabled:
                managers.append(WandbManager(cfg, experiment_name))

            # Initialize DVC if enabled
            if stage_config.tracking.dvc.enabled:
                managers.append(DVCManager(cfg, experiment_name))

        except Exception as e:
            logger.error(f"Failed to initialize tracking managers: {e}")

        return managers
