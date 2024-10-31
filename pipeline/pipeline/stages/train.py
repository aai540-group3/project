"""
Train Stage
=================

.. module:: pipeline.stages.train
   :synopsis: Model training and optimize stage

.. moduleauthor:: aai540-group3
"""

from typing import Optional

import pandas as pd
from hydra.utils import instantiate
from loguru import logger

from ..models.base import BaseModel
from .base import PipelineStage


class TrainStage(PipelineStage):
    """Model training stage implementation.

    Handles model training, validation, and artifact logging.
    Supports both quick validation and full training modes.

    :param cfg: Training configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def run(self) -> None:
        """Execute training pipeline.

        :raises RuntimeError: If training fails
        :raises IOError: If saving artifacts fails
        """
        self.logger.info(f"Starting training in {self.cfg.experiment.name} mode")

        try:
            # Load data
            train_data = self._load_data(self.cfg.data.train_path)
            val_data = self._load_data(self.cfg.data.val_path)

            # Initialize model
            model = self._initialize_model()

            # Train model
            self._train_model(model, train_data, val_data)

            # Save artifacts
            self._save_artifacts(model)

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training pipeline failed: {str(e)}")

    def _initialize_model(self) -> BaseModel:
        """Initialize model with configuration.

        :return: Initialized model instance
        :rtype: BaseModel
        :raises ValueError: If model configuration is invalid
        """
        try:
            model_cfg = self.cfg.model[self.cfg.model.name]
            return instantiate(model_cfg)
        except Exception as e:
            raise ValueError(f"Failed to initialize model: {e}")

    def _train_model(
        self,
        model: BaseModel,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train model with provided data.

        :param model: Model instance
        :type model: BaseModel
        :param train_data: Training data
        :type train_data: pd.DataFrame
        :param val_data: Validation data
        :type val_data: Optional[pd.DataFrame]
        :raises RuntimeError: If training fails
        """
        try:
            X_train = train_data.drop(self.cfg.data.target, axis=1)
            y_train = train_data[self.cfg.data.target]

            X_val = None
            y_val = None
            if val_data is not None:
                X_val = val_data.drop(self.cfg.data.target, axis=1)
                y_val = val_data[self.cfg.data.target]

            model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                **self.cfg.training.params,
            )

            self.log_metrics(model.metrics)
        except Exception as e:
            raise RuntimeError(f"Model training failed: {e}")
