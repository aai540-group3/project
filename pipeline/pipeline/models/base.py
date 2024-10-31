"""
Base Model Interface
====================

.. module:: pipeline.models.base
   :synopsis: Base interface for all machine learning models

.. moduleauthor:: aai540-group3
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all machine learning models.

    Provides interface and common functionality for model implementations.
    Supports both quick validation and full training modes.

    :param cfg: Model configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize model.

        :param cfg: Model configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self._is_fitted = False
        self.metrics: Dict[str, float] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate model configuration.

        :raises ValueError: If required configuration fields are missing
        """
        required_fields = ["name", "type", "hyperparameters"]
        for field in required_fields:
            if field not in self.cfg:
                raise ValueError(f"Missing required config field: {field}")

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """Train the model.

        :param X_train: Training features
        :type X_train: pd.DataFrame
        :param y_train: Training labels
        :type y_train: pd.Series
        :param X_val: Validation features
        :type X_val: Optional[pd.DataFrame]
        :param y_val: Validation labels
        :type y_val: Optional[pd.Series]
        :param kwargs: Additional training parameters
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Train method must be implemented by subclasses")

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        :param X: Input features
        :type X: pd.DataFrame
        :return: Predicted labels
        :rtype: np.ndarray
        :raises NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Predict method must be implemented by subclasses")

    def save(self, path: Union[str, Path]) -> None:
        """Save model and metadata.

        :param path: Path to save model
        :type path: Union[str, Path]
        :raises IOError: If saving fails
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            model_data = {
                "model": self._model,
                "metrics": self.metrics,
                "config": self.cfg,
                "is_fitted": self._is_fitted,
            }
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError(f"Failed to save model: {e}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk.

        :param path: Path to load model from
        :type path: Union[str, Path]
        :raises FileNotFoundError: If model file doesn't exist
        :raises IOError: If loading fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            model_data = joblib.load(path)
            self._model = model_data["model"]
            self.metrics = model_data["metrics"]
            self.cfg = model_data["config"]
            self._is_fitted = model_data["is_fitted"]
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise IOError(f"Failed to load model: {e}")
