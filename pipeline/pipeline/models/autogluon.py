import pathlib
import typing

import pandas as pd
from autogluon.tabular import TabularPredictor
from loguru import logger
from omegaconf import OmegaConf

from .metrics import Metrics
from .model import Model


class Autogluon(Model):
    """Concrete AutoGluon Model implementation."""

    def __init__(self):
        """Initialize Autogluon."""
        super().__init__()
        self.predictor: typing.Optional[TabularPredictor] = None

    def train(self) -> typing.Tuple[pathlib.Path, Metrics]:
        """Train the AutoGluon model."""
        logger.info("Starting training with AutoGluon.")

        # Prepare training and tuning data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        tuning_data = pd.concat([self.X_val, self.y_val], axis=1)

        # Convert OmegaConf configurations to a standard Python dictionary
        hyperparameters = OmegaConf.to_container(self.model_config.get("hyperparameters", {}), resolve=True)

        # Initialize and fit the TabularPredictor
        self.predictor = TabularPredictor(
            label=self.label_column,
            path=str(self.models_dir),
            eval_metric=self.model_config.get("metric", "roc_auc"),
            problem_type=self.model_config.get("problem_type", "binary"),
        ).fit(
            train_data=train_data,
            tuning_data=tuning_data,
            hyperparameters=hyperparameters,
            time_limit=self.model_config.get("time_limit", 60),
            presets=self.model_config.get("presets", "medium_quality"),
            verbosity=2,
        )

        # Save the trained model path
        model_path = self.save_model_path()

        # Evaluate on validation data
        y_pred = self.predictor.predict(self.X_val)
        y_proba = self.predictor.predict_proba(self.X_val).iloc[:, 1]
        metrics = Metrics(
            y_true=self.y_val.tolist(),
            y_pred=y_pred.tolist(),
            y_proba=y_proba.tolist(),
        )

        logger.info("Training completed.")
        return model_path, metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        return self.predictor.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction probabilities using the trained model."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        return self.predictor.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Metrics:
        """Evaluate the model on the provided test set."""
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X).iloc[:, 1]
        metrics = Metrics(
            y_true=y.tolist(),
            y_pred=y_pred.tolist(),
            y_proba=y_proba.tolist(),
        )
        logger.info("Evaluation completed.")
        return metrics

    def save_model_path(self) -> pathlib.Path:
        """Save the trained AutoGluon predictor."""
        model_path = self.models_dir / "autogluon_predictor"
        self.predictor.save(model_path)
        logger.info(f"Model saved at '{model_path}'.")
        return model_path

    def load_model(self, source_path: typing.Optional[pathlib.Path] = None) -> None:
        """Load a trained AutoGluon predictor from disk."""
        source_path = source_path or (self.models_dir / "autogluon_predictor")
        if not source_path.exists():
            logger.error(f"Model file not found at '{source_path}'.")
            raise FileNotFoundError(f"Model file not found at '{source_path}'.")
        self.predictor = TabularPredictor.load(str(source_path))
        logger.info(f"Model loaded from '{source_path}'.")
