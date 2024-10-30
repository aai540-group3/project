import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manage ML experiments with MLflow."""

    def __init__(self, cfg: DictConfig):
        """Initialize experiment manager.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.client = None
        self.experiment = None
        self.run_id = None
        self._setup_tracking()

    def _setup_tracking(self) -> None:
        """Setup MLflow tracking."""
        if self.cfg.experiment.mlflow.enabled:
            mlflow.set_tracking_uri(self.cfg.experiment.mlflow.tracking_uri)
            self.client = MlflowClient()

            # Create or get experiment
            try:
                self.experiment = self.client.create_experiment(
                    name=self.cfg.experiment.name, tags=self.cfg.experiment.tags
                )
            except Exception:
                self.experiment = self.client.get_experiment_by_name(
                    self.cfg.experiment.name
                )

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict] = None
    ) -> None:
        """Start a new MLflow run.

        Args:
            run_name: Name of the run
            tags: Additional tags for the run
        """
        if self.cfg.experiment.mlflow.enabled:
            # Start run
            mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=run_name,
                tags=tags,
            )
            self.run_id = mlflow.active_run().info.run_id

            # Log configuration
            self._log_config()

            # Set run-level tags
            run_tags = {**self.cfg.experiment.mlflow.tags}
            if tags:
                run_tags.update(tags)
            self.set_tags(run_tags)

            logger.info(f"Started run: {self.run_id}")

    def end_run(self) -> None:
        """End current MLflow run."""
        if self.cfg.experiment.mlflow.enabled and mlflow.active_run():
            mlflow.end_run()
            logger.info(f"Ended run: {self.run_id}")
            self.run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Parameters to log
        """
        if self.cfg.experiment.mlflow.enabled:
            mlflow.log_params(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Metrics to log
            step: Step number for the metrics
        """
        if self.cfg.experiment.mlflow.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Union[str, Path]) -> None:
        """Log artifact to MLflow.

        Args:
            local_path: Path to artifact
        """
        if self.cfg.experiment.mlflow.enabled:
            mlflow.log_artifact(str(local_path))

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        conda_env: Optional[Dict] = None,
        code_paths: Optional[List[str]] = None,
    ) -> None:
        """Log model to MLflow.

        Args:
            model: Model to log
            artifact_path: Path in artifact store
            conda_env: Conda environment specification
            code_paths: Additional code paths to log
        """
        if self.cfg.experiment.mlflow.enabled:
            if self.cfg.experiment.mlflow.artifacts.log_model:
                mlflow.sklearn.log_model(
                    model, artifact_path, conda_env=conda_env, code_paths=code_paths
                )

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the current run.

        Args:
            tags: Tags to set
        """
        if self.cfg.experiment.mlflow.enabled:
            mlflow.set_tags(tags)

    def _log_config(self) -> None:
        """Log configuration to MLflow."""
        if self.cfg.experiment.mlflow.enabled:
            # Convert config to dictionary
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)

            # Save config to YAML
            config_path = Path("config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Log config file
            self.log_artifact(config_path)

            # Remove temporary file
            config_path.unlink()
