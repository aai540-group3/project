import logging
from pathlib import Path
from typing import Optional

import hydra
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from pipeline.utils.experiment import ExperimentTracker
from pipeline.utils.versioning import DataVersioning

logger = logging.getLogger(__name__)


def setup_tracking(cfg: DictConfig) -> ExperimentTracker:
    """Setup experiment tracking."""
    return ExperimentTracker(
        cfg=cfg,
        experiment_name=cfg.experiment.name,
        tracking_uri=cfg.tracking.uri,
        registry_uri=cfg.registry.uri,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> Optional[float]:
    """Main training pipeline with DVC integration."""
    # Setup experiment tracking
    tracker = setup_tracking(cfg)

    # Initialize DVC versioning
    versioning = DataVersioning(get_original_cwd())

    # Start tracking run
    tracker.start_run(
        run_name=f"{cfg.model.name}_{cfg.experiment.name}",
        tags={
            "model_type": cfg.model.name,
            "experiment": cfg.experiment.name,
            "data_version": versioning.get_data_hash(cfg.data.path),
        },
    )

    try:
        # Convert relative paths to absolute
        cfg.paths = OmegaConf.create(
            {k: to_absolute_path(v) for k, v in cfg.paths.items()}
        )

        # Log configuration
        logger.info("\n" + OmegaConf.to_yaml(cfg))
        tracker.log_params(OmegaConf.to_container(cfg))

        # Verify data versions
        if not versioning.check_data_version(
            cfg.data.path, Path(cfg.paths.metadata) / "data_version.yaml"
        ):
            versioning.log_data_version(
                cfg.data.path,
                Path(cfg.paths.metadata) / "data_version.yaml",
                {"experiment": cfg.experiment.name},
            )

        # Run DVC pipeline stages
        for stage in cfg.pipeline.stages:
            if cfg.pipeline.stages[stage].enabled:
                logger.info(f"Running stage: {stage}")
                stage_module = instantiate(cfg.pipeline.stages[stage])
                stage_module.run()

        # Get final metrics
        metrics_path = Path(cfg.paths.metrics) / "evaluation" / "model_comparison.json"
        if metrics_path.exists():
            with metrics_path.open() as f:
                final_metrics = json.load(f)
                tracker.log_metrics(final_metrics)
                return final_metrics.get(cfg.evaluation.primary_metric)

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise

    finally:
        tracker.end_run()


if __name__ == "__main__":
    main()
