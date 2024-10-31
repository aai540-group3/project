import logging
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate

from .base import PipelineStage

logger = logging.getLogger(__name__)


class ModelRegistrationStage(PipelineStage):
    """Model registration stage."""

    def run(self) -> None:
        """Run model registration."""
        self.tracker.start_run()

        try:
            # Load comparison metrics
            comparison_path = (
                Path(self.cfg.paths.metrics) / "evaluation" / "model_comparison.json"
            )
            comparison_metrics = pd.read_json(comparison_path)

            # Register best model
            best_model_name = self.registry.select_best_model(
                comparison_metrics,
                metric=self.cfg.registry.selection_metric,
                threshold=self.cfg.registry.selection_threshold,
            )

            if best_model_name:
                # Load best model
                model_path = Path(self.cfg.paths.models) / best_model_name / "model.pkl"
                model = instantiate(self.cfg.models[best_model_name])
                model.load(model_path)

                # Register model
                self.registry.register_model(
                    model=model,
                    name=f"{self.cfg.experiment.name}-{best_model_name}",
                    metrics=comparison_metrics[best_model_name],
                    tags={
                        "experiment_id": self.tracker.run_id,
                        "model_type": best_model_name,
                        "dataset": self.cfg.data.name,
                    },
                )

            # Save registry metadata
            self.registry.save_metadata(Path(self.cfg.paths.registry) / "metadata.json")

        finally:
            self.tracker.end_run()
