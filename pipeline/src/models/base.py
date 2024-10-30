import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from ..utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class Model(ABC):
    """Abstract base class for all models."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize model.

        Args:
            cfg: Model configuration
        """
        self.cfg = cfg
        self._model: Optional[BaseEstimator] = None
        self.metrics: Dict[str, float] = {}
        self.best_params: Dict[str, Any] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @abstractmethod
    def _create_model(self, **kwargs) -> BaseEstimator:
        """Create and return model instance."""
        pass

    @abstractmethod
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        pass

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Train model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional arguments passed to model.fit()
        """
        if X_val is not None and y_val is not None:
            logger.info("Starting hyperparameter optimization...")
            self.best_params = self.optimize(X_train, y_train, X_val, y_val)
            logger.info(f"Best parameters: {self.best_params}")
            self._model = self._create_model(**self.best_params)
        else:
            self._model = self._create_model(**self.cfg.hyperparameters)

        logger.info("Training model...")
        self._model.fit(X_train, y_train, **kwargs)
        self._is_fitted = True

        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before making predictions.")
        return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Features
            y: Labels

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        self.metrics = calculate_metrics(
            y_true=y, y_pred=y_pred, y_pred_proba=y_pred_proba
        )

        logger.info("Model evaluation metrics:")
        for metric, value in self.metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return self.metrics

    def save(self, path: Union[str, Path]) -> None:
        """Save model and metadata.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "metrics": self.metrics,
            "best_params": self.best_params,
            "config": self.cfg,
            "is_fitted": self._is_fitted,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load model and metadata.

        Args:
            path: Path to load model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)
        self._model = model_data["model"]
        self.metrics = model_data["metrics"]
        self.best_params = model_data["best_params"]
        self._is_fitted = model_data["is_fitted"]

        logger.info(f"Model loaded from {path}")
