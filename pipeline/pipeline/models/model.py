"""
Machine Learning Model Abstract Base
====================================

This module provides an abstract base class for machine learning models,
managing configurations, data processing, training, evaluation, and metrics.

.. module:: model
    :synopsis: Abstract base class for ML model pipelines

.. moduleauthor:: aai540-group3
"""

import json
import os
import random
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
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

        # Load model configuration
        self.cfg: DictConfig = OmegaConf.load("params.yaml")
        self.name = self.__class__.__name__.lower()
        self.mode = os.environ.get("MODE", self.cfg.get("mode", "quick"))
        seed = self.cfg.get("seed", 42)

        # Set seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Model and data setup
        self.model_config = self.load_model_config(self.name)
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.label_column = self.cfg.models.base.get("label", "target")

        # Setup directories for metrics, plots, and models using cfg
        self.metrics_dir = Path(self.cfg.paths.metrics) / self.name
        self.plots_dir = Path(self.cfg.paths.plots) / self.name
        self.models_dir = Path(self.cfg.paths.models) / self.name

        # Ensure the directories exist
        for path in [self.metrics_dir, self.plots_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Thread locks for concurrency
        self._model_lock = threading.Lock()
        self._metrics_lock = threading.Lock()

    @abstractmethod
    def save_model(self, model_artifact: Any, filename: str = "model.pkl") -> Path:
        """Save a model artifact to disk in the specified directory.

        :param model_artifact: Model artifact to be saved.
        :type model_artifact: Any
        :param filename: Name of the file to save the model as.
        :type filename: str
        :return: Path to the saved model file.
        :rtype: Path
        """
        pass

    @abstractmethod
    def load_model(self, filename: str = "model.pkl") -> Any:
        """Load a model artifact from disk.

        :param filename: Name of the file from which to load the model.
        :type filename: str
        :return: Loaded model artifact.
        :rtype: Any
        """
        pass

    def save_metrics(self, metrics_data: Optional[Dict[str, Any]] = None):
        """Save metrics data to a JSON file in a thread-safe manner.

        :param metrics_data: Dictionary of metrics data.
        :type metrics_data: Optional[Dict[str, Any]]
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
            self.predict(self.X_test)
            self.predict_proba(self.X_test)
            self.get_estimator()
            self.calculate_and_save_metrics(self.metrics_dir)
            logger.info(f"Model {self.name} execution completed in {datetime.now() - self.start_time}.")
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            raise

    def prepare_data(self):
        """Load, split, and balance dataset."""
        try:
            data_path = Path(self.cfg.paths.processed) / "features.parquet"
            data = pd.read_parquet(data_path)

            # Reduce dataset size in 'quick' mode
            if self.mode == "quick":
                sample_frac = self.cfg.models.base.quick.get("sample_fraction", 0.1)
                data = data.sample(frac=sample_frac, random_state=self.cfg.get("seed", 42))
                logger.info(f"Quick mode: using {sample_frac*100}% of the data.")

            X, y = data.drop(columns=[self.label_column]), data[self.label_column]
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data(X, y)

            # Balance the training data
            smote = SMOTE(random_state=self.cfg.get("seed", 42))
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logger.info("Data preparation complete.")

        except FileNotFoundError as e:
            logger.error(f"Data file not found at '{data_path}': {e}")
            raise
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

    def calculate_and_save_metrics(self, output_dir: Union[str, Path]) -> Metrics:
        """Calculate and save all metrics and visualizations.

        :param output_dir: Directory where metrics and plots will be saved
        :type output_dir: Union[str, Path]
        :return: The computed metrics object
        :rtype: Metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create predictions first
        y_pred = self.predict(self.X_test)
        y_proba = self.predict_proba(self.X_test)

        # For binary classification, take the probability of the positive class
        if isinstance(y_proba, pd.DataFrame):
            if y_proba.shape[1] == 2:
                y_proba = y_proba.iloc[:, 1]
            elif y_proba.shape[1] == 1:
                y_proba = y_proba.iloc[:, 0]
            else:
                raise ValueError("Unexpected shape for y_proba DataFrame")
        elif isinstance(y_proba, np.ndarray):
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            elif y_proba.ndim == 1 or y_proba.shape[1] == 1:
                y_proba = y_proba.ravel()
            else:
                raise ValueError("Unexpected shape for y_proba array")

        metrics = Metrics(
            y_true=self.y_test.tolist(),
            y_proba=y_proba.tolist(),
            y_pred=y_pred.tolist(),
            model=self,
            X=self.X_test,
            mode=self.mode,
        )

        # Save metrics to JSON
        metrics_dict = metrics.to_dict()
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Metrics saved successfully at '{metrics_path}'.")

        # Generate classification report
        report = metrics.get_classification_report()
        report_path = output_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Classification report saved at '{report_path}'.")

        # Generate plots
        self.generate_plots(metrics)

        # Feature importance
        feature_importance = self.get_feature_importance()
        if feature_importance:
            feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"])
            feature_importance_df = feature_importance_df.sort_values("importance", ascending=False)

            # Save feature importance data under metrics directory
            feature_importance_csv_path = self.metrics_dir / "feature_importance.csv"
            feature_importance_df.to_csv(feature_importance_csv_path, index=False)
            logger.info(f"Feature importance data saved at '{feature_importance_csv_path}'")

            # Save feature importance plot under plots directory
            feature_importance_plot_path = self.plots_dir / "feature_importance.png"
            Metrics.plot_feature_importance(feature_importance_df, feature_importance_plot_path)
            logger.info(f"Feature importance plot saved at '{feature_importance_plot_path}'")
        else:
            logger.warning("Feature importance could not be calculated.")

        # SHAP values
        if self.mode != "quick":
            shap_values = metrics.compute_shap_values()
            if shap_values is not None:
                shap_dir = output_dir / "shap"
                metrics.generate_shap_plots(shap_values, shap_dir)
            else:
                logger.warning("SHAP values could not be computed for the model.")
        else:
            logger.info("Skipping SHAP analysis in quick mode.")

        return metrics

    def generate_plots(self, metrics: Metrics):
        """Generate and save evaluation plots."""
        paths = {
            "confusion_matrix": self.plots_dir / "confusion_matrix.png",
            "roc_curve": self.plots_dir / "roc_curve.png",
            "precision_recall_curve": self.plots_dir / "precision_recall_curve.png",
            "calibration_curve": self.plots_dir / "calibration_curve.png",
            "probability_distribution": self.plots_dir / "probability_distribution.png",
        }

        # Use data from the metrics object
        metrics.plot_confusion_matrix(save_path=paths["confusion_matrix"])

        metrics.plot_roc_curve(save_path=paths["roc_curve"])

        metrics.plot_precision_recall_curve(
            save_path=paths["precision_recall_curve"],
        )

        metrics.plot_calibration_curve(save_path=paths["calibration_curve"])

        metrics.plot_probability_distribution(
            save_path=paths["probability_distribution"],
        )

        logger.info("All evaluation plots generated and saved.")
