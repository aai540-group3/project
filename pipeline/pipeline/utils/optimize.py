"""
Hyperparameter Optimization
===========================

.. module:: pipeline.utils.optimize
   :synopsis: Hyperparameter optimize utilities

.. moduleauthor:: aai540-group3
"""

from typing import Any, Callable, Dict, Optional

import optuna
from omegaconf import DictConfig

from .logging import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """Hyperparameter optimize using Optuna."""

    def __init__(self, cfg: DictConfig):
        """Initialize optimizer.

        :param cfg: Optimization configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.study = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate optimize configuration."""
        required = ["n_trials", "metric", "direction"]
        if not all(hasattr(self.cfg, param) for param in required):
            raise ValueError(f"Missing required parameters: {required}")

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run optimize.

        :param objective: Objective function
        :type objective: Callable[[Dict[str, Any]], float]
        :param n_trials: Number of trials
        :type n_trials: Optional[int]
        :param timeout: Optimization timeout
        :type timeout: Optional[int]
        :return: Best parameters
        :rtype: Dict[str, Any]
        """
        study = optuna.create_study(
            direction=self.cfg.direction,
            sampler=optuna.samplers.TPESampler(seed=self.cfg.seed),
        )

        study.optimize(
            objective,
            n_trials=n_trials or self.cfg.n_trials,
            timeout=timeout or self.cfg.timeout,
        )

        self.study = study
        return study.best_params

    def get_best_value(self) -> float:
        """Get best optimize value.

        :return: Best value
        :rtype: float
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")
        return self.study.best_value

    def get_best_trial(self) -> optuna.Trial:
        """Get best trial.

        :return: Best trial
        :rtype: optuna.Trial
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")
        return self.study.best_trial

    def get_param_importances(self) -> Dict[str, float]:
        """Get parameter importances.

        :return: Parameter importances
        :rtype: Dict[str, float]
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")
        return optuna.importance.get_param_importances(self.study)
