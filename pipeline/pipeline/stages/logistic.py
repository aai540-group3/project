import hashlib
import os
import warnings
from functools import wraps
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from dvclive import Live
from pipeline.stages.base import PipelineStage


def track(func):
    """Decorator for managing DVC Live tracking around a training process."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.live = Live(dir=str(Path(self.cfg.paths.metrics) / "logistic"), dvcyaml=False)
        try:
            result = func(self, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e
        finally:
            if self.live:
                self.live.end()
            logger.info("Training completed")
        return result

    return wrapper


class Logistic(PipelineStage):
    """Pipeline stage for Logistic Regression model training with DVC Live tracking."""

    def __init__(self):
        """Initialize the Logistic Regression pipeline stage."""
        super().__init__()
        self.live = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.scaler = None

    @logger.catch()
    def run(self):
        """Entry point for running model training."""
        self.train()

    @logger.catch()
    @track
    def train(self):
        """Prepare data, initialize and train the Logistic Regression model, and evaluate results."""
        warnings.filterwarnings("ignore")
        np.random.seed(self.cfg.seed)

        # Load configuration
        mode = os.getenv("TRAIN_MODE", "quick")
        config = self.cfg.logistic[mode]

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        self.model_params = config.get("hyperparameters", {})
        self.label_column = self.cfg.logistic.model.get("label", "readmitted")

        # Load data and preprocess
        data = self.load_data("features.parquet", subdir="data/processed")
        X = data.drop(columns=[self.label_column])
        y = data[self.label_column]
        data_hash = hashlib.md5(pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=True).values).hexdigest()

        # Split data and log parameters
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        self.X_test, self.y_test = X_test, y_test
        self.log_params(config, data_hash, X_train, X_val, X_test)

        # Run optimization if specified and enabled
        if "optimize" in config and config["optimize"].get("enabled", False):
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val, config["optimize"])
            self.model_params.update(best_params)

        # Initialize, scale, and train the model
        self.model = self.initialize_model()
        self.train_model(X_train, y_train)

        # Generate predictions and calculate metrics
        y_pred, y_pred_proba = self.generate_predictions(X_test)
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Log metrics and create evaluation plots
        self.log_metrics(metrics)
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba, metrics)

    def initialize_model(self):
        """Initialize the Logistic Regression model with specified hyperparameters."""
        return LogisticRegression(**self.model_params)

    def train_model(self, X_train, y_train):
        """Scale features and train the Logistic Regression model."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def generate_predictions(self, X_test):
        """Generate predictions and prediction probabilities using the trained model."""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        return y_pred, y_pred_proba

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, optimize_config):
        """Optimize hyperparameters using Optuna."""

        def objective(trial):
            params = {}
            for param_name, param_cfg in optimize_config["param_space"].items():
                if param_cfg["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_cfg["low"], param_cfg["high"], log=param_cfg.get("log", False)
                    )
                elif param_cfg["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_cfg["choices"])

            model = LogisticRegression(**params)
            model.fit(self.scaler.fit_transform(X_train), y_train)
            y_pred_proba = model.predict_proba(self.scaler.transform(X_val))[:, 1]
            return roc_auc_score(y_val, y_pred_proba)

        study = optuna.create_study(direction=optimize_config["direction"])
        study.optimize(objective, n_trials=optimize_config["n_trials"], timeout=optimize_config["timeout"])

        return study.best_params
