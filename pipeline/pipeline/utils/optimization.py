"""
Hyperparameter Optimization
========================

.. module:: pipeline.utils.optimization
   :synopsis: Hyperparameter optimization utilities

.. moduleauthor:: aai540-group3
"""

from typing import Any, Callable, Dict, Optional, Union

import optuna
from omegaconf import DictConfig

from .logging import get_logger

logger = get_logger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna.

    :param cfg: Optimization configuration
    :type cfg: DictConfig
    """

    def __init__(self, cfg: DictConfig):
        """Initialize optimizer.

        :param cfg: Optimization configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.study: Optional[optuna.Study] = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate optimization configuration.

        :raises ValueError: If configuration is invalid
        """
        required_fields = ['n_trials', 'metric', 'direction', 'param_space']
        for field in required_fields:
            if field not in self.cfg:
                raise ValueError(f"Missing required config field: {field}")

    def _create_study(self) -> optuna.Study:
        """Create Optuna study.

        :return: Configured study
        :rtype: optuna.Study
        """
        return optuna.create_study(
            direction=self.cfg.direction,
            sampler=optuna.samplers.TPESampler(
                seed=self.cfg.get('seed', 42)
            )
        )

    def _suggest_params(
        self,
        trial: optuna.Trial,
        param_space: Dict
    ) -> Dict:
        """Suggest parameters based on configuration.

        :param trial: Optuna trial
        :type trial: optuna.Trial
        :param param_space: Parameter space configuration
        :type param_space: Dict
        :return: Suggested parameters
        :rtype: Dict
        """
        params = {}
        for name, space in param_space.items():
            # Check conditions if specified
            if 'condition' in space:
                if not eval(space.condition, {'params': params}):
                    continue

            if space.type == 'float':
                params[name] = trial.suggest_float(
                    name,
                    space.low,
                    space.high,
                    log=space.get('log', False)
                )
            elif space.type == 'int':
                params[name] = trial.suggest_int(
                    name,
                    space.low,
                    space.high,
                    log=space.get('log', False)
                )
            elif space.type == 'categorical':
                params[name] = trial.suggest_categorical(
                    name,
                    space.choices
                )
            else:
                raise ValueError(f"Unknown parameter type: {space.type}")

        return params

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run optimization.

        :param objective: Objective function to minimize/maximize
        :type objective: Callable[[Dict[str, Any]], float]
        :param n_trials: Number of trials, defaults to None
        :type n_trials: Optional[int]
        :param timeout: Timeout in seconds, defaults to None
        :type timeout: Optional[int]
        :return: Best parameters
        :rtype: Dict[str, Any]
        """
        self.study = self._create_study()

        def wrapped_objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, self.cfg.param_space)
            return objective(params)

        try:
            self.study.optimize(
                wrapped_objective,
                n_trials=n_trials or self.cfg.n_trials,
                timeout=timeout or self.cfg.get('timeout'),
                callbacks=[self._log_callback]
            )
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

        return self.study.best_params

    def _log_callback(
        self,
        study: optuna.Study,
        trial: optuna.FrozenTrial
    ) -> None:
        """Log optimization progress.

        :param study: Optuna study
        :type study: optuna.Study
        :param trial: Current trial
        :type trial: optuna.FrozenTrial
        """
        logger.info(
            f"Trial {trial.number} finished with value: {trial.value:.4f} "
            f"and parameters: {trial.params}"
        )
