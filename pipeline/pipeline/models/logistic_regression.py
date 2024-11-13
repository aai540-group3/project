"""
Logistic Regression Model
=========================

.. module:: pipeline.models.logistic_regression
   :synopsis: Logistic Regression model

.. moduleauthor:: aai540-group3
"""

import pathlib
import typing

import numpy as np
import pandas as pd
import sklearn
from loguru import logger

from .metrics import Metrics
from .model import Model


class LogisticRegression(Model):
    """Concrete implementation of BaseModel using Logistic Regression."""

    def __init__(self):
        """Initialize the LogisticRegressionModel."""
        super().__init__()
        self.predictor: typing.Optional[sklearn.linear_model.LogisticRegression] = None
        self.label_column: str = self.cfg.get("label_column", "readmitted")
        self.problem_type: str = self.cfg.logistic.get("problem_type", "binary")
        self.eval_metric: str = self.cfg.logistic.get("metric", "roc_auc")
        self.model_params: typing.Dict[str, typing.Any] = self.cfg.logistic.get("model_params", {})
        self.preprocessor: typing.Optional[sklearn.pipeline.Pipeline] = None
        logger.info(f"LogisticRegressionModel initialized with label column '{self.label_column}'.")

    def train(self) -> typing.Tuple[pathlib.Path, Metrics]:
        """Train the Logistic Regression model."""
        logger.info("Starting training with Logistic Regression.")
        data = pd.concat([self.cfg.data.X_train, self.cfg.data.y_train], axis=1)
        tuning_data = pd.concat([self.cfg.data.X_val, self.cfg.data.y_val], axis=1)

        # Initialize the Logistic Regression model
        self.predictor = LogisticRegression(**self.model_params)

        # Fit the model
        self.predictor.fit(self.cfg.data.X_train, self.cfg.data.y_train)

        # Save the trained model
        model_path = self.save_model_path()

        # Evaluate on validation data
        y_val = self.cfg.data.y_val
        y_pred = self.predict(self.cfg.data.X_val)
        y_proba = self.predict_proba(self.cfg.data.X_val).iloc[:, 1]

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
        return pd.DataFrame(self.predictor.predict_proba(X), columns=self.predictor.classes_)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Metrics:
        """Evaluate the model on the test set."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        y_pred = self.predict(X).tolist()
        y_proba = self.predict_proba(X).iloc[:, 1].tolist()
        metrics = Metrics(
            y_true=y.tolist(),
            y_pred=y_pred,
            y_proba=y_proba,
        )
        logger.info("Evaluation completed.")
        return metrics

    def optimize(self) -> typing.Dict[str, typing.Any]:
        """Perform hyperparameter optimization."""
        logger.info("Logistic Regression typically does not require extensive hyperparameter optimization.")
        logger.info("Returning current hyperparameters.")
        return self.hyperparameters

    def save_model_path(self) -> pathlib.Path:
        """Save the trained Logistic Regression model."""
        model_path = self.models_dir / "logistic_regression_model.joblib"
        # Use joblib to save the model
        import joblib

        joblib.dump(self.predictor, model_path)
        logger.info(f"Model saved at '{model_path}'.")
        return model_path

    def load_model(self, source_path: typing.Optional[pathlib.Path] = None) -> None:
        """Load a trained Logistic Regression model from disk."""
        source_path = source_path or (self.models_dir / "logistic_regression_model.joblib")
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found at '{source_path}'.")
        import joblib

        self.predictor = joblib.load(source_path)
        logger.info(f"Model loaded from '{source_path}'.")
