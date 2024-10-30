# src/utils/optimization.py
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Optimize model hyperparameters using Optuna."""

    def __init__(self, cfg: DictConfig):
        """Initialize optimizer.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.study = None
        self.best_params = {}
        self.best_value = None

    def optimize(
        self,
        objective: Callable[[optuna.Trial], float],
        direction: str = "maximize"
    ) -> Dict[str, Any]:
        """Run optimization.

        Args:
            objective: Objective function to optimize
            direction: Optimization direction ("maximize" or "minimize")

        Returns:
            Best parameters found
        """
        # Create study
        self.study = optuna.create_study(
            study_name=self.cfg.optimization.study_name,
            direction=direction,
            sampler=self._create_sampler()
        )

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.cfg.optimization.n_trials,
            timeout=self.cfg.optimization.timeout,
            show_progress_bar=True
        )

        # Store results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Save results
        self._save_results()

        return self.best_params

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        sampler_name = self.cfg.optimization.sampler.name.lower()
        sampler_params = self.cfg.optimization.sampler.params

        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(**sampler_params)
        elif sampler_name == "random":
            return optuna.samplers.RandomSampler(**sampler_params)
        elif sampler_name == "grid":
            return optuna.samplers.GridSampler(**sampler_params)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

    def _save_results(self) -> None:
        """Save optimization results."""
        if not self.study:
            raise RuntimeError("No optimization results to save")

        # Create results directory
        results_dir = Path(self.cfg.optimization.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save study statistics
        stats = {
            "study_name": self.study.study_name,
            "direction": self.study.direction.name,
            "n_trials": len(self.study.trials),
            "best_params": self.best_params,
            "best_value": self.best_value
        }

        with open(results_dir / "study_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Save all trials information
        trials = []
        for trial in self.study.trials:
            trials.append({
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name
            })

        with open(results_dir / "trials.json", "w") as f:
            json.dump(trials, f, indent=2)

        # Create visualization
        try:
            import plotly.graph_objects as go

            # Parameter importance plot
            importances = optuna.importance.get_param_importances(self.study)
            fig = go.Figure([go.Bar(
                x=list(importances.keys()),
                y=list(importances.values())
            )])
            fig.update_layout(
                title="Parameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance Score"
            )
            fig.write_html(str(results_dir / "parameter_importance.html"))

            # Optimization history plot
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_html(str(results_dir / "optimization_history.html"))

        except ImportError:
            logger.warning("plotly not installed. Skipping visualization.")
