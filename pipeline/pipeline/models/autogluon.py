"""
AutoGluon Model Implementation
==============================

.. module:: pipeline.models.autogluon
   :synopsis: Configuration-driven AutoGluon model

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from loguru import logger
from omegaconf import DictConfig

from pipeline.models.base import Model
from pipeline.models.metrics import Metrics


class AutogluonModel(Model):
    """Concrete implementation of BaseModel using AutoGluon."""

    def __init__(self, cfg: DictConfig):
        """Initialize the AutogluonModel."""
        super().__init__(cfg)
        self.predictor: Optional[TabularPredictor] = None
        self.label_column: str = self.cfg.get("label_column", "readmitted")
        self.problem_type: str = self.cfg.autogluon.get("problem_type", "binary")
        self.eval_metric: str = self.cfg.autogluon.get("eval_metric", "roc_auc")
        self.model_params: Dict[str, Any] = self.cfg.autogluon.get("model_params", {})
        logger.info(f"AutogluonModel initialized with label column '{self.label_column}'.")

    def train(self) -> Tuple[Path, Metrics]:
        """Train the AutoGluon model."""
        logger.info("Starting training with AutoGluon.")
        train_data = pd.concat([self.cfg.data.X_train, self.cfg.data.y_train], axis=1)
        tuning_data = pd.concat([self.cfg.data.X_val, self.cfg.data.y_val], axis=1)

        self.predictor = TabularPredictor(
            label=self.label_column,
            path=str(self.models_dir),
            eval_metric=self.eval_metric,
            problem_type=self.problem_type,
        ).fit(
            train_data=train_data,
            tuning_data=tuning_data,
            hyperparameters=self.hyperparameters,
            time_limit=self.cfg.autogluon.get("time_limit", 60),
            presets=self.cfg.autogluon.get("presets", "medium_quality"),
            verbosity=2,
        )

        # Save the trained model
        model_path = self.save_model_path()

        # Evaluate on validation data
        y_val = self.cfg.data.y_val
        y_pred = self.predictor.predict(self.cfg.data.X_val)
        y_proba = self.predictor.predict_proba(self.cfg.data.X_val).iloc[:, 1]

        metrics = Metrics(
            y_true=y_val.tolist(),
            y_pred=y_pred.tolist(),
            y_proba=y_proba.tolist(),
        )

        logger.info("Training completed.")
        return model_path, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        return self.predictor.predict(X).values

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction probabilities using the trained model."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        return self.predictor.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Metrics:
        """Evaluate the model on the test set."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        y_pred = self.predictor.predict(X).tolist()
        y_proba = self.predictor.predict_proba(X).iloc[:, 1].tolist()
        metrics = Metrics(
            y_true=y.tolist(),
            y_pred=y_pred,
            y_proba=y_proba,
        )
        logger.info("Evaluation completed.")
        return metrics

    def optimize(self) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        logger.info("AutoGluon handles hyperparameter optimization internally during training.")
        logger.info("Hyperparameter optimization completed.")
        return self.hyperparameters

    def save_model_path(self) -> Path:
        """Save the trained AutoGluon predictor."""
        model_path = self.models_dir / "autogluon_predictor"
        self.predictor.save(model_path)
        logger.info(f"Model saved at '{model_path}'.")
        return model_path

    def load_model(self, source_path: Optional[Path] = None) -> None:
        """Load a trained AutoGluon predictor from disk."""
        source_path = source_path or (self.models_dir / "autogluon_predictor")
        self.predictor = TabularPredictor.load(str(source_path))
        logger.info(f"Model loaded from '{source_path}'.")
