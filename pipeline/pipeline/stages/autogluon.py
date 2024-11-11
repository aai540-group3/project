import hashlib
import os
import warnings
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dvclive import Live
from pipeline.stages.base import PipelineStage


def track(func):
    """Decorator for managing DVC Live tracking around a training process."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.live = Live(dir=str(Path(self.cfg.paths.metrics) / "autogluon"), dvcyaml=False)
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


class Autogluon(PipelineStage):
    """Pipeline stage for AutoGluon model training with DVC Live tracking."""

    def __init__(self):
        """Initialize the AutoGluon pipeline stage with base configurations."""
        super().__init__()
        self.live = None
        self.X_test = None
        self.y_test = None

    @logger.catch()
    def run(self):
        """Entry point for running model training."""
        self.train()

    @logger.catch()
    @track
    def train(self):
        """Perform data preparation, model training, and evaluation."""
        warnings.filterwarnings("ignore")
        np.random.seed(self.cfg.seed)

        # Load configuration
        mode = os.getenv("TRAIN_MODE", "quick")
        config = self.cfg.autogluon[mode]

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        self.model_params = config.get("model", {})
        self.label_column = self.model_params.get("label", "readmitted")

        # Load data and process
        data = self.load_data("features.parquet", subdir="data/processed")
        X = data.drop(columns=[self.label_column])
        y = data[self.label_column]
        data_hash = hashlib.md5(pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=True).values).hexdigest()

        # Split data and log parameters
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        self.X_test, self.y_test = X_test, y_test
        self.log_params(config, data_hash, X_train, X_val, X_test)

        # Initialize and train the predictor
        predictor = self.initialize_predictor()
        self.train_model(predictor, X_train, y_train, X_val, y_val, config)

        # Generate predictions and calculate metrics
        y_pred, y_pred_proba = self.generate_predictions(predictor, X_test)
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Log metrics and create evaluation plots
        self.log_metrics(metrics)
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba, metrics)
        self.plot_feature_importance(predictor)

        # Save model outputs and metrics
        self.save_outputs(predictor, metrics)

    def initialize_predictor(self):
        """Initialize the AutoGluon predictor with configured parameters."""
        return TabularPredictor(
            label=self.label_column,
            path=str(Path(self.cfg.paths.models) / "autogluon"),
            eval_metric=self.model_params.get("metric", "roc_auc"),
            problem_type=self.model_params.get("problem_type", "binary"),
        )

    def train_model(self, predictor, X_train, y_train, X_val, y_val, config):
        """Train the model with the specified configuration and data."""
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        hyperparameters = config.get("hyperparameters", {})

        predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            time_limit=config.get("time_limit", 60),
            hyperparameters=hyperparameters,
            presets=config.get("presets", "medium_quality"),
            verbosity=2,
        )

    def generate_predictions(self, predictor, X_test):
        """Generate predictions and probabilities for the test set."""
        y_pred = predictor.predict(X_test)
        y_pred_proba = predictor.predict_proba(X_test)[1]
        return y_pred, y_pred_proba

    def save_outputs(self, predictor, metrics):
        """Save model leaderboard, feature importance, and training summary."""
        leaderboard = predictor.leaderboard(extra_info=True)
        best_model_name = leaderboard.iloc[0]["model"]
        best_model_score = float(leaderboard.iloc[0]["score_val"])
        importance_df = predictor.feature_importance(pd.concat([self.X_test, self.y_test], axis=1))
        fit_summary = predictor.fit_summary(verbosity=0)
        training_time = fit_summary.get("training_time", None)
        hyperparameters = fit_summary["model_hyperparams"][best_model_name]

        # Aggregate model information for saving
        model_info = {
            "model_path": str(Path(self.cfg.paths.models) / "autogluon"),
            "best_model": {
                "name": best_model_name,
                "score": best_model_score,
                "hyperparameters": hyperparameters,
            },
            "leaderboard": leaderboard.to_dict(),
            "feature_importance": importance_df.to_dict(),
            "training_time": training_time,
            "metrics": metrics,
            "model_params": self.model_params,
        }

        # Save model information and feature importance
        self.save_metrics("model_info", model_info)
        self.save_output(
            importance_df.reset_index(),
            "feature_importance.csv",
            subdir=str(Path(self.cfg.paths.metrics) / self.name),
        )
