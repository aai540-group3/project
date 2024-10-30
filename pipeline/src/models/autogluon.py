# src/models/autogluon.py
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from omegaconf import DictConfig

from .base import BaseModel

logger = logging.getLogger(__name__)

class AutoGluonModel(BaseModel):
    """AutoGluon model implementation."""

    def __init__(self, cfg: DictConfig):
        """Initialize AutoGluon model.

        Args:
            cfg: Model configuration
        """
        super().__init__(cfg)
        self.predictor: Optional[TabularPredictor] = None
        self.feature_importance_: Optional[Dict[str, float]] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional arguments passed to fit
        """
        logger.info("Training AutoGluon model...")

        # Combine features and target
        train_data = X_train.copy()
        train_data[self.cfg.target_column] = y_train

        if X_val is not None and y_val is not None:
            val_data = X_val.copy()
            val_data[self.cfg.target_column] = y_val
        else:
            val_data = None

        # Initialize predictor
        self.predictor = TabularPredictor(
            label=self.cfg.target_column,
            problem_type="binary",
            eval_metric="roc_auc",
            path=str(Path(self.cfg.model_path) / "autogluon")
        )

        # Train model
        self.predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            time_limit=self.cfg.time_limit,
            presets=self.cfg.presets,
            **self.cfg.hyperparameters,
            **kwargs
        )

        # Get feature importance
        self.feature_importance_ = self.predictor.feature_importance(train_data)

        # Calculate and store metrics
        if val_data is not None:
            self.metrics = self.predictor.evaluate(val_data)

        self._is_fitted = True
        logger.info("AutoGluon model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        self._check_is_fitted()
        return self.predictor.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions.

        Args:
            X: Features

        Returns:
            Predicted probabilities
        """
        self._check_is_fitted()
        return self.predictor.predict_proba(X)

    def feature_importance(self, X: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            X: Features (not used, kept for API consistency)

        Returns:
            Feature importance scores
        """
        self._check_is_fitted()
        return self.feature_importance_

    def save(self, path: Union[str, Path]) -> None:
        """Save model.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.predictor is not None:
            self.predictor.save(str(path))

        # Save additional metadata
        metadata = {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance_,
            "is_fitted": self._is_fitted,
            "config": self.cfg
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load model.

        Args:
            path: Path to load model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        # Load predictor
        self.predictor = TabularPredictor.load(str(path))

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.metrics = metadata["metrics"]
        self.feature_importance_ = metadata["feature_importance"]
        self._is_fitted = metadata["is_fitted"]
        self.cfg = metadata["config"]
