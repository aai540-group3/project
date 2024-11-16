import json
import logging
from pathlib import Path
from typing import Dict

import optuna
from hydra.utils import instantiate

from ..utils.optimize import HyperparameterOptimizer
from .base import PipelineStage

logger = logging.getLogger(__name__)


class OptimizeStage(PipelineStage):
    """Hyperparameter optimization stage."""

    def run(self) -> None:
        """Execute optimization pipeline."""
        self.tracker.start_run(run_name=f"optimize_{self.cfg.model.name}")

        try:
            # Load data
            train_data = self._load_data(self.cfg.data.train_path)
            val_data = self._load_data(self.cfg.data.val_path)

            # Initialize model and optimizer
            model = instantiate(self.cfg.model)
            optimizer = HyperparameterOptimizer(self.cfg.optimize)

            # Define objective function
            def objective(trial: optuna.Trial) -> float:
                # Suggest parameters
                params = self._suggest_parameters(trial)

                # Train model
                model.train(
                    train_data.drop(self.cfg.data.target, axis=1),
                    train_data[self.cfg.data.target],
                    val_data.drop(self.cfg.data.target, axis=1),
                    val_data[self.cfg.data.target],
                    **params,
                )

                # Get validation metric
                metric_value = model.metrics[self.cfg.optimize.metric]

                # Log to MLflow
                self.tracker.log_metrics({f"trial_{trial.number}_{self.cfg.optimize.metric}": metric_value})

                return metric_value

            # Run optimization
            best_params = optimizer.optimize(objective, direction=self.cfg.optimize.direction)

            # Log best parameters
            self.tracker.log_params({"best_params": best_params, "best_value": optimizer.best_value})

            # Save results
            self._save_results(optimizer)

            logger.info(f"Best {self.cfg.optimize.metric}: " f"{optimizer.best_value:.4f}")

        finally:
            self.tracker.end_run()

    def _suggest_parameters(self, trial: optuna.Trial) -> Dict:
        """Suggest parameters based on configuration.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of suggested parameters
        """
        params = {}

        for name, space in self.cfg.optimize.param_space.items():
            if space.type == "float":
                params[name] = trial.suggest_float(name, space.low, space.high, log=space.get("log", False))
            elif space.type == "int":
                params[name] = trial.suggest_int(name, space.low, space.high, log=space.get("log", False))
            elif space.type == "categorical":
                params[name] = trial.suggest_categorical(name, space.choices)
            else:
                raise ValueError(f"Unknown parameter type: {space.type}")

        return params

    def _save_results(self, optimizer: HyperparameterOptimizer) -> None:
        """Save optimization results.

        Args:
            optimizer: Hyperparameter optimizer
        """
        # Create output directory
        output_dir = Path(self.cfg.paths.optimize) / self.cfg.model.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save study statistics
        stats = {
            "study_name": optimizer.study.study_name,
            "direction": optimizer.study.direction.name,
            "n_trials": len(optimizer.study.trials),
            "best_params": optimizer.best_params,
            "best_value": optimizer.best_value,
        }

        with open(output_dir / "study_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Save all trials information
        trials = []
        for trial in optimizer.study.trials:
            trials.append(
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                }
            )

        with open(output_dir / "trials.json", "w") as f:
            json.dump(trials, f, indent=2)

        # Create visualizations
        try:
            import plotly.graph_objects as go

            # Parameter importance plot
            importances = optuna.importance.get_param_importances(optimizer.study)
            fig = go.Figure([go.Bar(x=list(importances.keys()), y=list(importances.values()))])
            fig.update_layout(
                title="Parameter Importance",
                xaxis_title="Parameter",
                yaxis_title="Importance Score",
            )
            fig.write_html(str(output_dir / "parameter_importance.html"))

            # Optimization history plot
            fig = optuna.visualization.plot_optimization_history(optimizer.study)
            fig.write_html(str(output_dir / "optimization_history.html"))

        except ImportError:
            logger.warning("plotly not installed. Skipping visualization.")
