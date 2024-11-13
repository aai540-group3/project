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
    """Abstract model class for machine learning models."""

    def __init__(self):
        """Initialize the Abstract Model."""
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None

        # Tracking attributes
        self.live: Optional[Live] = None

        # Load configuration
        self.cfg: DictConfig = OmegaConf.load("params.yaml")
        logger.debug(f"Configuration loaded:\n{self.cfg}")

        # Model identity and mode setup
        self.name: str = self.__class__.__name__.lower()
        logger.info(f"Initializing '{self.name}' model.")
        self.mode: str = self.cfg.models.base.get("mode", "quick")
        logger.info(f"Operating mode set to '{self.mode}'.")

        # Set random seed
        seed: int = self.cfg.get("seed", 42)
        np.random.seed(seed)
        logger.debug(f"Random seed set to {seed}.")

        # Load model-specific configurations based on model type and mode
        self.model_config = self.load_model_config(self.name)

        # Initialize data attributes
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Load and prepare data if specified in configuration
        self.label_column = self.cfg.models.base.get("label", "target")
        if self.cfg.get("data"):
            self.prepare_data()

        # Directory setup
        self.metrics_dir: Path = Path("metrics") / self.name
        self.plots_dir: Path = Path("plots") / self.name
        self.models_dir: Path = Path("models") / self.name
        for path in [self.metrics_dir, self.plots_dir, self.models_dir]:
            path.mkdir(parents=True, exist_ok=True)

        # Locks for thread safety
        self._model_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metadata_lock = threading.Lock()

    def prepare_data(self):
        """Load and split the dataset based on configuration, applying SMOTE to the training set if specified."""
        logger.info("Loading and preparing data.")
        data_path = Path(self.cfg.paths.processed) / "features.parquet"

        try:
            data = pd.read_parquet(data_path)
            X = data.drop(columns=[self.label_column])
            y = data[self.label_column]

            # Split data and assign to instance variables
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data(X, y)
            logger.info("Data preparation complete.")
            logger.debug(
                f"Data shapes - X_train: {self.X_train.shape}, y_train: {self.y_train.shape}, "
                f"X_val: {self.X_val.shape}, y_val: {self.y_val.shape}, X_test: {self.X_test.shape}, y_test: {self.y_test.shape}"
            )

            # Apply SMOTE to training data if enabled in configuration
            if self.cfg.get("apply_smote", False):
                logger.info("Applying SMOTE to the training set.")
                smote = SMOTE(random_state=self.cfg.get("seed", 42))
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                logger.debug(f"After SMOTE - X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into training, validation, and test sets."""
        logger.info("Splitting data into train, validation, and test sets.")
        train_size = self.cfg.dataset.splits.train
        val_size = self.cfg.dataset.splits.val
        test_size = self.cfg.dataset.splits.test

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

    def load_model_config(self, model_type: str) -> Dict[str, Any]:
        """Load model-specific configurations based on model type and mode."""
        config = self.cfg.models.get(model_type, {}).get(self.mode, {})
        if not config:
            raise ValueError(f"No configuration found for model '{model_type}' in mode '{self.mode}'.")
        logger.debug(f"Loaded configuration for model '{model_type}' in mode '{self.mode}': {config}")
        return config

    @abstractmethod
    def train(self) -> Tuple[Path, Metrics]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction probabilities."""
        pass

    def save_model(self, model_artifact: Any, filename: str = "model.pkl") -> None:
        """Save the model artifact to disk.

        :param model_artifact: The trained model object to save.
        :param filename: The filename to save the model under.
        """
        destination_path = self.models_dir / filename
        try:
            with open(destination_path, "wb") as f:
                pickle.dump(model_artifact, f)
            logger.info(f"Model saved successfully at '{destination_path}'.")
        except Exception as e:
            logger.error(f"Failed to save model to '{destination_path}': {e}")
            raise

    def load_model(self, filename: str = "model.pkl") -> Any:
        """Load a model artifact from disk.

        :param filename: The filename of the model to load.
        :return: The loaded model object.
        """
        source_path = self.models_dir / filename
        if not source_path.exists():
            logger.error(f"Model file not found at '{source_path}'.")
            raise FileNotFoundError(f"Model file not found at '{source_path}'")

        try:
            with open(source_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully from '{source_path}'.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from '{source_path}': {e}")
            raise

    def calculate_and_save_metrics(
        self, y_true, y_pred, y_proba, feature_importance: Optional[pd.DataFrame] = None
    ) -> Metrics:
        """Calculate metrics, generate plots, and save classification report, metrics, and feature importance.

        :param y_true: Ground truth labels
        :param y_pred: Predicted labels
        :param y_proba: Predicted probabilities
        :param feature_importance: DataFrame with feature names and importance scores, if applicable.
        :return: Metrics object with computed metrics
        """
        metrics = Metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
        metrics_data = metrics.to_dict()

        # Save evaluation metrics
        self.save_metrics("evaluation_metrics", metrics_data)

        # Generate and save plots
        self.generate_plots(metrics)

        # Save classification report
        report = metrics.get_classification_report()
        report_path = self.metrics_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Classification report saved at '{report_path}'.")

        # Save feature importance if provided
        if feature_importance is not None and not feature_importance.empty:
            self.save_feature_importance_csv(feature_importance)

        return metrics

    def save_feature_importance_csv(self, feature_importance: pd.DataFrame) -> None:
        """Save feature importance data to a CSV file.

        :param feature_importance: DataFrame with feature names and importance scores.
        """
        csv_path = self.metrics_dir / "feature_importance.csv"
        try:
            feature_importance.to_csv(csv_path, index=False)
            logger.info(f"Feature importance CSV saved at '{csv_path}'.")
        except Exception as e:
            logger.error(f"Failed to save feature importance to '{csv_path}': {e}")
            raise

    def generate_plots(self, metrics: Metrics) -> None:
        """Generate and save plots using metrics data.

        :param metrics: Metrics object containing y_true, y_pred, and y_proba
        """
        cm_path = self.plots_dir / "confusion_matrix.png"
        roc_path = self.plots_dir / "roc_curve.png"
        pr_path = self.plots_dir / "precision_recall_curve.png"
        calib_path = self.plots_dir / "calibration_curve.png"
        prob_dist_path = self.plots_dir / "probability_distribution.png"

        Metrics.plot_confusion_matrix(metrics.y_true, metrics.y_pred, save_path=cm_path)
        Metrics.plot_roc_curve(metrics.y_true, metrics.y_proba, save_path=roc_path)
        Metrics.plot_precision_recall_curve(metrics.y_true, metrics.y_proba, save_path=pr_path)
        Metrics.plot_calibration_curve(metrics.y_true, metrics.y_proba, save_path=calib_path)
        Metrics.plot_probability_distribution(metrics.y_true, metrics.y_proba, save_path=prob_dist_path)

        logger.info("All evaluation plots generated and saved.")

    def save_metrics(self, filename: str, metrics_data: Optional[Dict[str, Any]] = None) -> None:
        """Save computed metrics to a JSON file.

        :param filename: The name of the file to save metrics to (without extension).
        :param metrics_data: Dictionary containing metrics data to save. If None, saves the current self.metrics data.
        """
        metrics_path = self.metrics_dir / "metrics.json"

        if metrics_data is None and self.metrics is not None:
            metrics_data = self.metrics.to_dict()

        try:
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=4)
            logger.info(f"Metrics saved successfully at '{metrics_path}'.")
        except Exception as e:
            logger.error(f"Failed to save metrics to '{metrics_path}': {e}")
            raise
