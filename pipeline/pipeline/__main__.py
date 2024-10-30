"""Main entry point for MLOps pipeline."""

import logging

import click
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute MLOps pipeline.

    :param cfg: Hydra configuration
    :type cfg: DictConfig
    """
    import subprocess
    from pathlib import Path

    logger.info("Starting MLOps pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    try:
        # Run DVC pipeline
        result = subprocess.run(
            ["dvc", "repro"], check=True, capture_output=True, text=True
        )
        logger.info(f"DVC output:\n{result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed:\n{e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
