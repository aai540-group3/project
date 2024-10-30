import logging
from typing import Any, Dict

import numpy as np
import optuna
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .base import Model

logger = logging.getLogger(__name__)


class LogisticRegressionModel(Model):
    """Logistic Regression model implementation."""

    def _create_model(self, **kwargs) -> LogisticRegression:
        """Create Logistic Regression model."""
        params = {**self.cfg.hyperparameters, **kwargs}
        return LogisticRegression(**params)

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "C": trial.suggest_float(
                    "C",
                    self.cfg.optimization.param_space.C.low,
                    self.cfg.optimization.param_space.C.high,
                    log=True,
                ),
                "penalty": trial.suggest_categorical(
                    "penalty", self.cfg.optimization.param_space.penalty.choices
                ),
                "solver": trial.suggest_categorical(
                    "solver", self.cfg.optimization.param_space.solver.choices
                ),
            }

            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = trial.suggest_float(
                    "l1_ratio",
                    self.cfg.optimization.param_space.l1_ratio.low,
                    self.cfg.optimization.param_space.l1_ratio.high,
                )

            model = self._create_model(**params)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg.get("seed", 42)),
        )

        study.optimize(
            objective, n_trials=self.cfg.optimization.n_trials, show_progress_bar=True
        )

        return study.best_params
