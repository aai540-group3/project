import logging
from pathlib import Path
from typing import Dict

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def validate_configs() -> None:
    """Validate all configuration files."""
    config_dir = Path("conf")

    # Validate main config
    main_config = OmegaConf.load(config_dir / "config.yaml")
    validate_main_config(main_config)

    # Validate model configs
    model_dir = config_dir / "model"
    for model_config in model_dir.glob("*.yaml"):
        validate_model_config(OmegaConf.load(model_config))

    # Validate pipeline configs
    pipeline_dir = config_dir / "pipeline"
    for pipeline_config in pipeline_dir.glob("*.yaml"):
        validate_pipeline_config(OmegaConf.load(pipeline_config))

    logger.info("All configurations validated successfully")


def validate_main_config(config: Dict) -> None:
    """Validate main configuration."""
    required_keys = ["version", "seed", "paths", "model_types"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in main config: {key}")


def validate_model_config(config: Dict) -> None:
    """Validate model configuration."""
    required_keys = ["name", "type", "hyperparameters", "optimization", "training"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in model config: {key}")


def validate_pipeline_config(config: Dict) -> None:
    """Validate pipeline configuration."""
    required_keys = ["stages"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in pipeline config: {key}")


if __name__ == "__main__":
    validate_configs()
