import logging
from pathlib import Path

import hydra
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig

from .base import PipelineStage

logger = logging.getLogger(__name__)


class TrainingStage(PipelineStage):
    """Model training stage."""

    def run(self) -> None:
        """Run model training."""
        self.tracker.start_run()

        try:
            # Initialize model
            model = instantiate(self.cfg.model)

            # Load data
            train_data = hydra.utils.get_original_cwd() / self.cfg.data.train_path
            val_data = hydra.utils.get_original_cwd() / self.cfg.data.val_path

            # Load best parameters
            best_params = self.registry.load_best_params(
                self.cfg.model.name, self.cfg.optimization.metric
            )

            # Train model
            model.train(train_data, val_data, **best_params)

            # Log metrics and model
            self.tracker.log_metrics(model.metrics)
            self.tracker.log_model(
                model, "model", conda_env=model.get_conda_env(), code_paths=["src"]
            )

            # Save model artifacts
            model_path = Path(self.cfg.paths.models) / self.cfg.model.name / "model.pkl"
            model.save(model_path)

        finally:
            self.tracker.end_run()
