"""
Configuration Management
========================

.. module:: pipeline.conf.config
   :synopsis: Configuration management with Hydra and structured configs

.. moduleauthor:: aai540-group3
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .schema import (
    BuildConfig,
    DeployConfig,
    EvaluateConfig,
    FeaturizeConfig,
    IngestConfig,
    OptimizeConfig,
    PreprocessConfig,
    StageExecutionConfig,
    TrainConfig,
)


class ConfigurationManager:
    """Manages configuration loading, validation, and access.

    This class provides a centralized way to handle configuration using Hydra
    with structured configs. It handles:
    - Configuration registration
    - Default configuration setup
    - Configuration validation
    - Path resolution
    - Environment variable interpolation
    """

    _instance = None
    _config_store = None
    _config: Optional[DictConfig] = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._register_configs()

    @classmethod
    def get_instance(cls) -> "ConfigurationManager":
        """Get singleton instance.

        :return: ConfigurationManager instance
        :rtype: ConfigurationManager
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_configs(self) -> None:
        """Register structured configurations with Hydra."""
        if self._config_store is None:
            self._config_store = ConfigStore.instance()

            # Register base config
            self._config_store.store(
                name="base_config",
                node=StageExecutionConfig,
            )

            # Register stage-specific configs
            stage_configs = {
                "build": BuildConfig,
                "ingest": IngestConfig,
                "preprocess": PreprocessConfig,
                "featurize": FeaturizeConfig,
                "train": TrainConfig,
                "optimize": OptimizeConfig,
                "evaluate": EvaluateConfig,
                "deploy": DeployConfig,
            }

            for stage_name, config_class in stage_configs.items():
                self._config_store.store(
                    group=f"pipeline/stages/{stage_name}",
                    name="config",
                    node=config_class,
                )

    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
    ) -> DictConfig:
        """Load configuration from file or defaults.

        :param config_path: Path to configuration file
        :type config_path: Optional[Union[str, Path]]
        :param overrides: Configuration overrides
        :type overrides: Optional[List[str]]
        :return: Loaded configuration
        :rtype: DictConfig
        :raises RuntimeError: If configuration loading fails
        """
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()

            # Initialize Hydra
            hydra.initialize(
                version_base=None,
                config_path=str(config_path) if config_path else "conf",
            )

            # Compose configuration
            cfg = hydra.compose(
                config_name="config",
                overrides=overrides or [],
                return_hydra_config=True,
            )

            if not cfg:
                raise RuntimeError("Empty configuration returned")

            # Validate configuration
            self._validate_config(cfg)

            # Resolve paths
            cfg = self._resolve_paths(cfg)

            # Store configuration
            self._config = cfg

            logger.info("Configuration loaded successfully")
            return cfg

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}")

    def _validate_config(self, cfg: DictConfig) -> None:
        """Validate configuration structure and values.

        :param cfg: Configuration to validate
        :type cfg: DictConfig
        :raises ValueError: If configuration is invalid
        """
        required_fields = [
            "pipeline",
            "paths",
            "experiment",
            "tracking",
        ]

        for field in required_fields:
            if field not in cfg:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate stage configurations
        if "stages" not in cfg.pipeline:
            raise ValueError("No pipeline stages configured")

        for stage_name, stage_cfg in cfg.pipeline.stages.items():
            if not stage_cfg.get("enabled", True):
                continue

            if "params" not in stage_cfg:
                raise ValueError(f"Missing params in stage configuration: {stage_name}")

    def _resolve_paths(self, cfg: DictConfig) -> DictConfig:
        """Resolve and validate paths in configuration.

        :param cfg: Configuration with paths
        :type cfg: DictConfig
        :return: Configuration with resolved paths
        :rtype: DictConfig
        """
        # Get runtime working directory
        cwd = Path.cwd()

        # Resolve paths relative to working directory
        resolved_paths = {}
        for key, path in cfg.paths.items():
            if isinstance(path, str):
                # Handle environment variable interpolation
                path = os.path.expandvars(path)

                # Resolve relative to working directory if not absolute
                resolved_path = Path(path)
                if not resolved_path.is_absolute():
                    resolved_path = cwd / path

                resolved_paths[key] = str(resolved_path)

        # Update configuration with resolved paths
        cfg.paths = OmegaConf.create(resolved_paths)

        return cfg

    def get_stage_config(self, stage_name: str) -> DictConfig:
        """Get configuration for a specific stage.

        :param stage_name: Name of the stage
        :type stage_name: str
        :return: Stage configuration
        :rtype: DictConfig
        :raises ValueError: If stage configuration is not found
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        try:
            return self._config.pipeline.stages[stage_name]
        except KeyError:
            raise ValueError(
                f"Stage configuration not found: {stage_name}. "
                f"Available stages: {list(self._config.pipeline.stages.keys())}"
            )

    def get_experiment_name(self) -> str:
        """Get current experiment name.

        :return: Experiment name
        :rtype: str
        :raises RuntimeError: If configuration is not loaded
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config.experiment.name

    def get_tracking_config(self) -> DictConfig:
        """Get tracking configuration.

        :return: Tracking configuration
        :rtype: DictConfig
        :raises RuntimeError: If configuration is not loaded
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config.tracking

    @property
    def config(self) -> DictConfig:
        """Get current configuration.

        :return: Current configuration
        :rtype: DictConfig
        :raises RuntimeError: If configuration is not loaded
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config


# Create singleton instance
config_manager = ConfigurationManager.get_instance()
