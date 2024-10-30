# src/stages/evaluate.py
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

from ..utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
from .base import PipelineStage

logger = logging.getLogger(__name__)

class EvaluationStage(PipelineStage):
    """Model evaluation stage."""

    def run(self) -> None:
        """Execute evaluation pipeline."""
        self.tracker.start_run(run_name="evaluate")

        try:
            # Load test data
            test_data = self._load_data(self.cfg.data.test_path)
            X_test = test_data.drop(self.cfg.data.target, axis=1)
            y_test = test_data[self.cfg.data.target]

            # Evaluate each model
            results = {}
            for model_name in self.cfg.model_types:
                # Load model
                model = self._load_model(model_name)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                results[model_name] = metrics

                # Generate plots
                self._generate_plots(
                    model_name,
                    y_test,
                    y_pred,
                    y_pred_proba,
                    model.feature_importance(X_test)
                )

            # Save results
            self._save_results(results)

            # Log best model
            best_model = self._select_best_model(results)
            self.tracker.log_params({
                "best_model": best_model,
                "best_score": results[best_model][self.cfg.evaluation.metric]
            })

            logger.info(f"Best model: {best_model}")

        finally:
            self.tracker.end_run()

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba)
        }

    def _generate_plots(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        feature_importance: Dict[str, float]
    ) -> None:
        """Generate evaluation plots.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            feature_importance: Feature importance scores
        """
        output_dir = Path(self.cfg.paths.plots) / "evaluation" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            output_dir / "confusion_matrix.png",
            labels=["No Readmission", "Readmission"]
        )

        # ROC curve
        plot_roc_curve(
            y_true,
            y_pred_proba,
            output_dir / "roc_curve.png"
        )

        # Feature importance
        plot_feature_importance(
            pd.Series(feature_importance),
            output_dir / "feature_importance.png",
            n_features=20
        )

    def _save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save evaluation results.

        Args:
            results: Dictionary of model results
        """
        output_dir = Path(self.cfg.paths.evaluation)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "model_comparison.json", "w") as f:
            json.dump(results, f, indent=2)

    def _select_best_model(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> str:
        """Select best model based on primary metric.

        Args:
            results: Dictionary of model results

        Returns:
            Name of best model
        """
        metric = self.cfg.evaluation.metric
        return max(
            results.keys(),
            key=lambda k: results[k][metric]
        )
