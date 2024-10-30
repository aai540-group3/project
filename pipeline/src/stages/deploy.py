import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..deploy.huggingface import HuggingFaceDeployer
from ..utils.experiment import ExperimentTracker

logger = logging.getLogger(__name__)


class DeploymentStage:
    """Model deployment stage."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tracker = ExperimentTracker(cfg, cfg.experiment.name)
        self.deployer = HuggingFaceDeployer(cfg)

    def run(self) -> None:
        """Run deployment pipeline."""
        self.tracker.start_run()

        try:
            # Load model and metrics
            model_path = Path(self.cfg.paths.models) / self.cfg.model.name / "model.pkl"
            metrics_path = (
                Path(self.cfg.paths.metrics) / "evaluation" / "model_comparison.json"
            )

            with open(metrics_path) as f:
                metrics = json.load(f)

            # Prepare deployment
            deploy_dir = Path(self.cfg.paths.deploy) / "huggingface"
            self.deployer.prepare_deployment(
                model_path=model_path,
                output_dir=deploy_dir,
                metrics=metrics[self.cfg.model.name],
                feature_info=self.cfg.features.feature_groups,
            )

            # Push to HuggingFace Hub
            self.deployer.push_to_hub(
                local_dir=deploy_dir,
                commit_message=f"Update {self.cfg.model.name} model",
            )

            # Log deployment information
            self.tracker.log_params(
                {
                    "deploy.repo_id": self.cfg.huggingface.repo_id,
                    "deploy.model": self.cfg.model.name,
                }
            )

        finally:
            self.tracker.end_run()
