"""
Machine Learning Model Abstract Base
====================================

This module provides an abstract base class for machine learning models,
managing configurations, data processing, training, evaluation, and metrics.

.. module:: model
  :synopsis: Abstract base class for ML model pipelines with data preparation,
             training, and evaluation.

.. moduleauthor:: aai540-group3
"""

import json
import pickle
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dvclive import Live
from imblearn.over_sampling import SMOTE
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from .metrics import Metrics


class Model(ABC):
    """Abstract base class for ML models."""

    def __init__(self):
        """Initialize model configurations, data setup, and directories."""
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.live: Optional[Live] = None

        # Load model configuration
        self.cfg: DictConfig = OmegaConf.load("params.yaml")
        self.name = self.__class__.__name__.lower()
        self.mode = self.cfg.models.base.get("mode", "quick")
        seed = self.cfg.get("seed", 42)
        np.random.seed(seed)

        # Model and data setup
        self.model_config = self.load_model_config(self.name)
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.label_column = self.cfg.models.base.get("label", "target")
        if self.cfg.get("data"):
            self.prepare_data()

        # Setup directories for metrics, plots, and models
        self.metrics_dir = Path("metrics") / self.name
        self.plots_dir = Path("plots") / self.name
        self.models_dir = Path("models") / self.name
        for path in [self.metrics_dir, self.plots_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Thread locks for concurrency
        self._model_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

    def save_model(self, model_artifact: Any, filename: str = "model.pkl") -> Path:
        """Save a model artifact to disk in the specified directory.

        :param model_artifact: Model artifact to be saved.
        :type model_artifact: Any
        :param filename: Name of the file to save the model as, defaults to "model.pkl".
        :type filename: str, optional
        :return: Path to the saved model file.
        :rtype: Path
        """
        destination_path = self.models_dir / filename
        try:
            with open(destination_path, "wb") as f:
                pickle.dump(model_artifact, f)
            logger.info(f"Model saved successfully at '{destination_path}'.")
            return destination_path
        except IOError as e:
            logger.error(f"Failed to save model to '{destination_path}': {e}")
            raise IOError(f"Error saving model to '{destination_path}'") from e

    def load_model(self, filename: str = "model.pkl") -> Any:
        """Load a model artifact from disk.

        :param filename: Name of the file from which to load the model, defaults to "model.pkl".
        :type filename: str, optional
        :return: Loaded model artifact.
        :rtype: Any
        """
        source_path = self.models_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found at '{source_path}'")
        try:
            with open(source_path, "rb") as f:
                return pickle.load(f)
        except (IOError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load model from '{source_path}': {e}")
            raise

    def save_metrics(self, metrics_data: Optional[Dict[str, Any]] = None) -> None:
        """Save metrics data to a JSON file in a thread-safe manner.

        :param filename: Name of the file to save metrics.
        :type filename: str
        :param metrics_data: Dictionary of metrics data, defaults to None.
        :type metrics_data: Optional[Dict[str, Any]], optional
        """
        metrics_path = self.metrics_dir / "metrics.json"
        try:
            with self._metrics_lock:
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=4)
            logger.info(f"Metrics saved successfully at '{metrics_path}'.")
        except Exception as e:
            logger.error(f"Failed to save metrics to '{metrics_path}': {e}")
            raise

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split the dataset into training, validation, and test sets.

        :param X: Features DataFrame.
        :type X: pd.DataFrame
        :param y: Target Series.
        :type y: pd.Series
        :return: Tuple of split DataFrames and Series for training, validation, and test sets.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        """
        train_size = self.cfg.dataset.splits.train
        val_size = self.cfg.dataset.splits.val
        test_size = self.cfg.dataset.splits.test

        # Ensure that split sizes sum to 1
        assert np.isclose(train_size + val_size + test_size, 1.0), "Train, val, and test splits must sum to 1."

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_size), random_state=self.cfg.get("seed", 42), stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(test_size / (val_size + test_size)),
            random_state=self.cfg.get("seed", 42),
            stratify=y_temp,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def execute(self):
        """Execute model training, prediction, and evaluation."""
        try:
            logger.info(f"Starting {self.name} model.")
            if self.X_train is None or self.y_train is None:
                self.prepare_data()
            model_artifact = self.train()
            self.save_model(model_artifact)
            y_pred = self.predict(self.X_test)
            y_proba = self.predict_proba(self.X_test).iloc[:, 1]
            estimator = self.get_estimator()
            metrics = self.calculate_and_save_metrics(
                y_true=self.y_test.tolist(),
                y_pred=y_pred.tolist(),
                y_proba=y_proba.tolist(),
                feature_importance=self.get_feature_importance(),
                estimator=estimator,
            )
            if self.live:
                self.live.log_metrics(metrics.to_dict())
            logger.info(f"Model {self.name} execution completed in {datetime.now() - self.start_time}.")
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise

    def prepare_data(self):
        """Load, split, and balance dataset."""
        try:
            data = pd.read_parquet(Path(self.cfg.paths.processed) / "features.parquet")
            X, y = data.drop(columns=[self.label_column]), data[self.label_column]
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data(X, y)

            smote = SMOTE(random_state=self.cfg.get("seed", 42))
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logger.info("Data preparation complete.")

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def load_model_config(self, model_type: str) -> Dict[str, Any]:
        """Load model-specific configurations from the main configuration file.

        :param model_type: The type of the model as defined in the configuration.
        :type model_type: str
        :return: Configuration for the specific model type.
        :rtype: Dict[str, Any]
        """
        config = self.cfg.models.get(model_type, {}).get(self.mode, {})
        if not config:
            raise ValueError(f"No configuration found for model '{model_type}' in mode '{self.mode}'.")
        return config

    @abstractmethod
    def train(self) -> Any:
        """Abstract method to train the model.

        This method is implemented by subclasses to encapsulate the model training process.
        It returns the trained model artifact, which is typically saved for later use.

        :return: The trained model artifact.
        :rtype: Any
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the input data.

        :param X: Input feature set for prediction.
        :type X: pd.DataFrame
        :return: Predictions array.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction probabilities for the input data.

        :param X: Input feature set for probability prediction.
        :type X: pd.DataFrame
        :return: DataFrame of prediction probabilities.
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    def get_estimator(self) -> Any:
        """Retrieve the underlying estimator for feature importance and SHAP computations.

        :return: The underlying trained estimator.
        :rtype: Any
        """
        pass

    def calculate_and_save_metrics(
        self,
        y_true,
        y_pred,
        y_proba,
        feature_importance: Optional[pd.DataFrame] = None,
        estimator: Optional[Any] = None,
    ) -> Metrics:
        """Calculate metrics, save reports, generate plots, and explain feature importance."""

        metrics = Metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba, model=estimator, X=self.X_test)

        # Save metrics to JSON
        self.save_metrics(metrics.to_dict())

        # Generate and save the classification report
        classification_report_data = metrics.get_classification_report()
        report_path = self.metrics_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(classification_report_data, f, indent=4)
        logger.info(f"Classification report saved at '{report_path}'.")

        # Generate additional plots
        self.generate_plots(metrics)

        # Check for feature importance and save if available
        if feature_importance is not None and feature_importance:
            feature_importance_path = self.plots_dir / "feature_importance.csv"
            pd.DataFrame.from_dict(feature_importance, orient="index", columns=["importance"]).to_csv(
                feature_importance_path
            )
            logger.info(f"Feature importance data saved at '{feature_importance_path}'")
        else:
            logger.warning("Feature importance data is unavailable. Skipping saving 'feature_importance.csv'.")

        # Attempt SHAP explanation if model is compatible
        shap_values = metrics.compute_shap_values(background=self.X_train)
        if shap_values:
            shap_save_dir = self.metrics_dir / "shap"
            shap_save_dir.mkdir(parents=True, exist_ok=True)
            metrics.generate_shap_plots(shap_values, save_dir=shap_save_dir)
        else:
            logger.warning("SHAP values could not be computed for the model.")

        return metrics

    def generate_plots(self, metrics: Metrics) -> None:
        """Generate and save evaluation plots."""
        paths = {
            "confusion_matrix": self.plots_dir / "confusion_matrix.png",
            "roc_curve": self.plots_dir / "roc_curve.png",
            "precision_recall_curve": self.plots_dir / "precision_recall_curve.png",
            "calibration_curve": self.plots_dir / "calibration_curve.png",
            "probability_distribution": self.plots_dir / "probability_distribution.png",
        }

        # Pass `y_true` and `y_pred` for confusion matrix
        metrics.plot_confusion_matrix(
            y_true=self.y_test, y_pred=self.predict(self.X_test), save_path=paths["confusion_matrix"]
        )

        # Pass `y_true` and `y_proba` for the ROC curve
        metrics.plot_roc_curve(
            y_true=self.y_test, y_proba=self.predict_proba(self.X_test).iloc[:, 1], save_path=paths["roc_curve"]
        )

        # Pass `y_true` and `y_proba` for the precision-recall curve
        metrics.plot_precision_recall_curve(
            y_true=self.y_test,
            y_proba=self.predict_proba(self.X_test).iloc[:, 1],
            save_path=paths["precision_recall_curve"],
        )

        metrics.plot_calibration_curve(
            y_true=self.y_test, y_proba=self.predict_proba(self.X_test).iloc[:, 1], save_path=paths["calibration_curve"]
        )

        metrics.plot_probability_distribution(
            y_true=self.y_test,
            y_proba=self.predict_proba(self.X_test).iloc[:, 1],
            save_path=paths["probability_distribution"],
        )

        logger.info("All evaluation plots generated and saved.")
