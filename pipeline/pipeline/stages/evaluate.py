"""
Evaluate Stage
===================

.. module:: pipeline.stages.evaluate
   :synopsis: Model evaluation and comparison stage

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd
from loguru import logger

from pipeline.stages.base import PipelineStage

from ..utils.metrics import calculate_metrics, get_confusion_matrix
from ..utils.visualization import plot_confusion_matrices, plot_feature_importance, plot_roc_curves


class EvaluateStage(PipelineStage):
    """Model evaluation and comparison stage.

    :param cfg: Stage configuration
    :type cfg: DictConfig
    """

    def run(self) -> None:
        """Execute evaluation pipeline.

        :raises RuntimeError: If evaluation fails
        """
        logger.info(f"Starting evaluation in {self.cfg.experiment.name} mode")

        try:
            # Load test data
            test_data = self._load_test_data()
            X_test = test_data.drop(columns=[self.cfg.data.target])
            y_test = test_data[self.cfg.data.target]

            results = {}
            feature_importance = {}
            predictions = {}

            # Evaluate each model
            for model_name in self.cfg.model_types:
                model_results = self._evaluate_model(model_name, X_test, y_test)
                results[model_name] = model_results["metrics"]
                feature_importance[model_name] = model_results["feature_importance"]
                predictions[model_name] = model_results["predictions"]

            # Generate comparison plots
            self._generate_comparison_plots(y_test, predictions, feature_importance)

            # Save and log results
            self._save_results(results)
            self._log_best_model(results)

            logger.info("Evaluation completed successfully")

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def _evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate single model.

        :param model_name: Name of the model
        :type model_name: str
        :param X_test: Test features
        :type X_test: pd.DataFrame
        :param y_test: Test labels
        :type y_test: pd.Series
        :return: Evaluation results
        :rtype: Dict
        """
        logger.info(f"Evaluating model: {model_name}")

        # Load model
        model = self._load_model(model_name)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, self.cfg.evaluation.metrics)

        # Calculate confusion matrix
        cm, cm_metrics = get_confusion_matrix(y_test, y_pred)
        metrics.update(cm_metrics)

        # Get feature importance if available
        feature_importance = model.feature_importance(X_test) if hasattr(model, "feature_importance") else None

        return {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "predictions": {"y_pred": y_pred, "y_pred_proba": y_pred_proba},
        }

    def _generate_comparison_plots(self, y_test: pd.Series, predictions: Dict, feature_importance: Dict) -> None:
        """Generate comparison plots.

        :param y_test: True labels
        :type y_test: pd.Series
        :param predictions: Model predictions
        :type predictions: Dict
        :param feature_importance: Feature importance scores
        :type feature_importance: Dict
        """
        plot_dir = Path(self.cfg.paths.plots) / "evaluation"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ROC curves
        plot_roc_curves(
            y_test,
            {name: pred["y_pred_proba"] for name, pred in predictions.items()},
            save_path=plot_dir / "roc_curves.png",
        )

        # Confusion matrices
        plot_confusion_matrices(
            y_test,
            {name: pred["y_pred"] for name, pred in predictions.items()},
            save_path=plot_dir / "confusion_matrices.png",
        )

        # Feature importance comparison
        if any(fi is not None for fi in feature_importance.values()):
            plot_feature_importance(
                feature_importance,
                top_n=self.cfg.evaluation.feature_importance.top_n,
                save_path=plot_dir / "feature_importance.png",
            )

    def _save_results(self, results: Dict) -> None:
        """Save evaluation results.

        :param results: Evaluation results
        :type results: Dict
        """
        output_dir = Path(self.cfg.paths.evaluation)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        pd.DataFrame(results).to_json(output_dir / "model_comparison.json")

        # Log to MLflow
        if self.cfg.mlflow.enabled:
            for model_name, metrics in results.items():
                mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items()})

    def _log_best_model(self, results: Dict) -> None:
        """Log best model information.

        :param results: Evaluation results
        :type results: Dict
        """
        metric = self.cfg.evaluation.primary_metric
        best_model = max(results.keys(), key=lambda k: results[k][metric])
        best_score = results[best_model][metric]

        logger.info(f"Best model: {best_model} ({metric}={best_score:.4f})")

        # Log to tracking systems
        self.log_metrics({"best_model_score": best_score, "best_model_name": best_model})
