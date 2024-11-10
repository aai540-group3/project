"""
AutoGluon Model Implementation
==============================

.. module:: pipeline.models.autogluon
   :synopsis: Configuration-driven AutoGluon model

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from loguru import logger
from omegaconf import DictConfig

from .base import BaseModel


class AutoGluonModel(BaseModel):
    """AutoGluon model with automated machine learning capabilities.

    :param cfg: Model configuration
    :type cfg: DictConfig
    """

    def __init__(self, cfg: DictConfig):
        """Initialize AutoGluon model.

        :param cfg: Model configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.predictor: Optional[TabularPredictor] = None

    def _get_predictor_args(self) -> Dict:
        """Get predictor configuration based on mode.

        :return: Predictor arguments
        :rtype: Dict
        """
        model_cfg = self.cfg.model
        return {
            "label": model_cfg.label,
            "problem_type": model_cfg.problem_type,
            "eval_metric": model_cfg.eval_metric,
            "path": str(Path(self.cfg.model_path) / f"autogluon_{self.cfg.experiment.name}"),
            "verbosity": model_cfg.verbosity,
        }

    def _get_training_args(self) -> Dict:
        """Get training configuration based on mode.

        :return: Training arguments
        :rtype: Dict
        """
        train_cfg = self.cfg.quick_mode.training if self.cfg.experiment.name == "quick" else self.cfg.training

        return {
            "time_limit": train_cfg.time_limit,
            "presets": train_cfg.presets,
            "hyperparameters": train_cfg.hyperparameters,
            "hyperparameter_tune_kwargs": train_cfg.get("hyperparameter_tune_kwargs"),
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """Train AutoGluon model.

        :param X_train: Training features
        :type X_train: pd.DataFrame
        :param y_train: Training labels
        :type y_train: pd.Series
        :param X_val: Validation features
        :type X_val: Optional[pd.DataFrame]
        :param y_val: Validation labels
        :type y_val: Optional[pd.Series]
        :param kwargs: Additional training parameters
        :raises RuntimeError: If training fails
        """
        try:
            # Prepare data
            train_data = X_train.copy()
            train_data[self.cfg.model.label] = y_train

            val_data = None
            if X_val is not None and y_val is not None:
                val_data = X_val.copy()
                val_data[self.cfg.model.label] = y_val

            # Initialize predictor
            self.predictor = TabularPredictor(**self._get_predictor_args())

            # Train model
            train_args = self._get_training_args()
            self.predictor.fit(train_data=train_data, tuning_data=val_data, **train_args, **kwargs)

            # Calculate validation metrics
            if val_data is not None:
                self.metrics = self.predictor.evaluate(val_data)

            self._is_fitted = True
            logger.info("AutoGluon training completed successfully")

        except Exception as e:
            logger.error(f"AutoGluon training failed: {str(e)}")
            raise RuntimeError(f"AutoGluon training failed: {str(e)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model.

        :param X: Input features
        :type X: pd.DataFrame
        :return: Predicted labels
        :rtype: np.ndarray
        :raises RuntimeError: If model is not trained
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.predictor.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        :param X: Input features
        :type X: pd.DataFrame
        :return: Class probabilities
        :rtype: np.ndarray
        :raises RuntimeError: If model is not trained
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before making predictions")
        return self.predictor.predict_proba(X)
