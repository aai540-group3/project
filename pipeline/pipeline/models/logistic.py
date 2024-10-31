"""
Logistic Regression Model Implementation
========================================

.. module:: pipeline.models.logistic
   :synopsis: Configuration-driven logistic regression model

.. moduleauthor:: aai540-group3
"""

from typing import Dict, Optional

import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class LogisticRegressionModel(BaseModel):
    """Logistic regression model with hyperparameter optimization.

    :param cfg: Model configuration
    :type cfg: DictConfig
    """

    def __init__(self, cfg: DictConfig):
        """Initialize logistic regression model.

        :param cfg: Model configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.scaler: Optional[StandardScaler] = None
        self._model: Optional[LogisticRegression] = None

    def _create_model(self, params: Optional[Dict] = None) -> LogisticRegression:
        """Create logistic regression model from configuration.

        :param params: Optional parameter overrides
        :type params: Optional[Dict]
        :return: Configured model instance
        :rtype: LogisticRegression
        """
        model_cfg = (
            self.cfg.quick_mode.model
            if self.cfg.experiment.name == "quick"
            else self.cfg.model
        )

        # Merge configuration with optional parameter overrides
        model_params = dict(model_cfg)
        if params:
            model_params.update(params)

        return LogisticRegression(**model_params)

    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        """Optimize hyperparameters using Optuna.

        :param X_train: Training features
        :type X_train: np.ndarray
        :param y_train: Training labels
        :type y_train: np.ndarray
        :param X_val: Validation features
        :type X_val: np.ndarray
        :param y_val: Validation labels
        :type y_val: np.ndarray
        :return: Best parameters
        :rtype: Dict
        """
        opt_cfg = (
            self.cfg.quick_mode.optimize
            if self.cfg.experiment.name == "quick"
            else self.cfg.optimize
        )

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, space in opt_cfg.param_space.items():
                # Check conditions if specified
                if "condition" in space:
                    if not eval(space.condition, {"params": params}):
                        continue

                if space.type == "float":
                    params[name] = trial.suggest_float(
                        name, space.low, space.high, log=space.get("log", False)
                    )
                elif space.type == "categorical":
                    params[name] = trial.suggest_categorical(name, space.choices)

            model = self._create_model(params)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return self._calculate_metric(y_val, y_pred_proba, opt_cfg.metric)

        study = optuna.create_study(
            direction=opt_cfg.direction,
            sampler=optuna.samplers.TPESampler(seed=self.cfg.get("seed", 42)),
        )
        study.optimize(objective, n_trials=opt_cfg.n_trials, timeout=opt_cfg.timeout)

        return study.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """Train logistic regression model.

        :param X_train: Training features
        :type X_train: pd.DataFrame
        :param y_train: Training labels
        :type y_train: pd.Series
        :param X_val: Validation features
        :type X_val: Optional[pd.DataFrame]
        :param y_val: Validation labels
        :type y_val: Optional[pd.Series]
        :param kwargs: Additional training parameters
        :raises RuntimeError: If training fails
        """
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

            # Optimize hyperparameters if validation data is provided
            if (
                X_val is not None
                and y_val is not None
                and not self.cfg.experiment.name == "quick"
            ):
                best_params = self._optimize_hyperparameters(
                    X_train_scaled, y_train, X_val_scaled, y_val
                )
                self._model = self._create_model(best_params)
            else:
                self._model = self._create_model()

            # Train model
            self._model.fit(X_train_scaled, y_train)

            # Calculate validation metrics
            if X_val is not None and y_val is not None:
                y_pred = self._model.predict(X_val_scaled)
                y_pred_proba = self._model.predict_proba(X_val_scaled)[:, 1]
                self.metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)

            self._is_fitted = True
            logger.info("Logistic regression training completed successfully")

        except Exception as e:
            logger.error(f"Logistic regression training failed: {str(e)}")
            raise RuntimeError(f"Logistic regression training failed: {str(e)}")
