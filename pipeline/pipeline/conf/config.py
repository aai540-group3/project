"""
Configuration Management
========================

.. module:: pipeline.conf.config
   :synopsis: Configuration management module for handling pipeline settings and paths.

.. moduleauthor:: aai540-group3
"""

import os
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from loguru import logger


class Config:
    """
    Configuration management class for loading, setting, and saving pipeline configurations.

    This class provides a centralized way to manage configuration files and paths
    within the pipeline, using Hydra and OmegaConf.
    """

    @staticmethod
    def load() -> DictConfig:
        """Load the configuration file using Hydra.

        :return: The loaded configuration as a DictConfig object.
        :rtype: DictConfig
        :raises RuntimeError: If loading configuration fails or Hydra returns an empty config.
        """
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()

            # Initialize Hydra with the configuration directory
            hydra.initialize(version_base=None, config_path=".")

            # Compose configuration
            cfg = hydra.compose(
                config_name="config",
                return_hydra_config=True,  # Include Hydra config in output
            )

            if not cfg:
                logger.error("Hydra returned an empty configuration.")
                raise RuntimeError("Configuration load failed: Empty configuration returned.")

            logger.debug(f"Configuration loaded: {cfg}")

            # Validate hydra configuration presence
            if "hydra" not in cfg:
                raise RuntimeError("Configuration load failed. Missing hydra configuration.")

            # Ensure HydraConfig is initialized
            try:
                if not HydraConfig.initialized():
                    HydraConfig().set_config(cfg)
            except Exception as hc_error:
                logger.error(f"Failed to initialize HydraConfig: {hc_error}")
                raise RuntimeError(f"Failed to initialize HydraConfig: {hc_error}")

            return cfg

        except Exception as e:
            logger.error(f"Configuration load failed: {e}")
            raise RuntimeError(f"Configuration load failed: {e}")
