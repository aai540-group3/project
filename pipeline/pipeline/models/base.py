"""
Base Model Implementation
========================

.. module:: pipeline.models.base
   :synopsis: Abstract base class for machine learning models

.. moduleauthor:: aai540-group3
"""

import json
import os
import pickle
import sys
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dvclive import Live
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .metrics import Metrics


class Model(ABC):
    """Abstract base class for machine learning models."""

    def __init__(self):
        """Initialize the Abstract Model."""
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None

        self.cfg: DictConfig = OmegaConf.load("params.yaml")
        logger.debug(f"Configuration loaded:\n{self.cfg}")

        self.name: str = self.__class__.__name__.lower()
        logger.info(f"Initializing '{self.name}' model.")

        mode = os.getenv("MODE") or self.cfg.get("mode", None)
        if not mode:
            logger.warning("No mode specified in configuration. Defaulting to quick mode")
            mode = "quick"

        self.mode: str = mode
        logger.info(f"Operating mode set to '{self.mode}'.")

        # SET NP RANDOM SEED
        seed: int = self.cfg.get("seed", None)
        if seed is None:
            logger.warning("No seed specified in configuration. Defaulting to 42")
            seed = 42
        np.random.seed(seed)

        # OPTIMIZATION
        optimize: bool = self.cfg.get("optimize", None)
        if optimize is None:
            logger.warning("No optimization flag specified in configuration. Defaulting to False")
            optimize = False

        self.hyperparameters: Dict[str, Any] = self.cfg.get("hyperparameters", {})

        # Data Attributes
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

        # Metrics and Live Tracking
        self.metrics: Optional[Metrics] = None
        self.live: Optional[Live] = None

        # Directory Setup
        self.metrics_dir: Path = Path("metrics") / self.name
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file: Path = self.metrics_dir / "metrics.json"

        self.plots_dir: Path = Path("plots") / self.name
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir: Path = Path("models") / self.name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_file: Path = self.models_dir / "model.pkl"

        # Timing Attributes
        self.training_time: Optional[float] = None
        self.evaluation_time: Optional[float] = None
        self.optimization_time: Optional[float] = None

        # Locks for thread safety
        self._model_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metadata_lock = threading.Lock()

    @abstractmethod
    def train(self, *args, **kwargs) -> Tuple[Path, Metrics]:
        raise NotImplementedError("Method 'train' must be implemented by subclasses.")

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Method 'predict' must be implemented by subclasses.")

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Metrics:
        raise NotImplementedError("Method 'evaluate' must be implemented by subclasses.")

    @abstractmethod
    def optimize(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Method 'optimize' must be implemented by subclasses.")

    def execute(self, X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> None:
        """Execute the model's workflow in a separate thread with logging and error handling."""
        self.X_test = X_test
        self.y_test = y_test
        logger.info(f"EXECUTING MODEL: {self.name}")

        def thread_run(X_test: pd.DataFrame, y_test: pd.Series, **kwargs):
            try:
                # Optimization
                if kwargs.get("enable_optimization", self.optimize):
                    logger.info("Starting hyperparameter optimization.")
                    optimize_start = datetime.now()
                    self.hyperparameters = kwargs.get("optimize_func", self.optimize)()
                    self.optimization_time = (datetime.now() - optimize_start).total_seconds()
                    logger.info(f"Optimized hyperparameters: {self.hyperparameters}")

                # Training
                logger.info("Starting model training.")
                train_start = datetime.now()
                train_func = kwargs.get("train_func", self.train)
                model_path, metrics = train_func()
                self.training_time = (datetime.now() - train_start).total_seconds()

                # End time
                self.end_time = datetime.now()
                duration = (self.end_time - self.start_time).total_seconds()
                logger.info(f"COMPLETED MODEL: {self.name} in {duration:.2f}s")

                # Save Model
                with self._model_lock:
                    save_model_func = kwargs.get("save_model_func", self.save_model)
                    save_model_func(model_path)

                # Metrics
                with self._metrics_lock:
                    self.metrics = metrics
                    save_metrics_func = kwargs.get("save_metrics_func", self.save_metrics)
                    save_metrics_func()

                # Metadata
                with self._metadata_lock:
                    save_metadata_func = kwargs.get("save_metadata_func", self.save_metadata)
                    save_metadata_func()

            except Exception:
                logger.error(f"FAILED MODEL: {self.name}")
                logger.exception("Error during model execution.")
                sys.exit(1)
            finally:
                logger.complete()

        # Run the thread
        thread = threading.Thread(target=thread_run, args=(X_test, y_test), kwargs=kwargs)
        thread.start()
        thread.join()

    def save_model(self, source_path: Path) -> None:
        """Save the trained model to disk."""
        destination_path = self.models_dir / "model.pkl"
        try:
            with self._model_lock:
                source_path.rename(destination_path)
            logger.info(f"Model saved successfully at '{destination_path}'.")
        except Exception as e:
            logger.error(f"Failed to save model to '{destination_path}': {e}")
            raise

    def load_model(self, source_path: Optional[Union[str, Path]] = None) -> Any:
        """Load a trained model from disk."""
        source_path = Path(source_path) if source_path else self.model_file
        try:
            with open(source_path, "rb") as f:
                model = pickle.load(f)
            self.__dict__.update(model.__dict__)
            logger.info(f"Model loaded successfully from '{source_path}'.")
            return self
        except Exception as e:
            logger.error(f"Failed to load model from '{source_path}': {e}")
            raise

    def save_metadata(self) -> None:
        """Save metadata information about the model training and evaluation."""
        if self.metrics:
            metadata = {
                "model": self.name,
                "execution_start": self.start_time.isoformat(),
                "execution_end": self.end_time.isoformat() if self.end_time else None,
                "total_time": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
                "training_time": self.training_time,
                "evaluation_time": self.evaluation_time,
                "optimization_time": self.optimization_time,
                "metrics": self.metrics.to_dict(),
                "filepath": str(self.model_file),
                "filesize": self.model_file.stat().st_size if self.model_file.exists() else None,
            }

            metadata_path = self.models_dir / "metadata.json"
            try:
                with self._metadata_lock:
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=4)
                logger.info(f"Metadata saved at '{metadata_path}'.")
            except Exception as e:
                logger.error(f"Failed to save metadata to '{metadata_path}': {e}")
                raise
        else:
            logger.warning("No metrics available to save in metadata.")
