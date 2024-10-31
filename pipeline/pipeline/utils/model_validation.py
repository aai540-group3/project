"""
Model Validation Utilities
==========================

.. module:: pipeline.utils.model_validation
   :synopsis: Model validation and testing utilities

.. moduleauthor:: aai540-group3
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score


class ModelValidator:
    """Model validation utility."""

    def __init__(self, cfg: DictConfig):
        """Initialize model validator.

        :param cfg: Validation configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.validation_results = {}

    def validate_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict:
        """Validate model comprehensively.

        :param model: Model to validate
        :type model: Any
        :param X: Training features
        :type X: Union[np.ndarray, pd.DataFrame]
        :param y: Training labels
        :type y: Union[np.ndarray, pd.Series]
        :param X_val: Validation features
        :type X_val: Optional[Union[np.ndarray, pd.DataFrame]]
        :param y_val: Validation labels
        :type y_val: Optional[Union[np.ndarray, pd.Series]]
        :return: Validation results
        :rtype: Dict
        """
        results = {}

        # Input validation
        input_errors = self._validate_input(X)
        results["input_validation"] = {
            "passed": len(input_errors) == 0,
            "errors": input_errors,
        }

        # Performance validation
        perf_results = self._validate_performance(model, X, y, X_val, y_val)
        results["performance_validation"] = perf_results

        # Resource validation
        resource_results = self._validate_resources(model, X)
        results["resource_validation"] = resource_results

        # Bias validation
        if self.cfg.bias.enabled:
            bias_results = self._validate_bias(model, X, y, protected_attributes=self.cfg.bias.protected_attributes)
            results["bias_validation"] = bias_results

        # Stability validation
        if self.cfg.stability.enabled:
            stability_results = self._validate_stability(model, X, y)
            results["stability_validation"] = stability_results

        self.validation_results = results
        return results

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> List[str]:
        """Validate input data.

        :param X: Input features
        :type X: Union[np.ndarray, pd.DataFrame]
        :return: List of validation errors
        :rtype: List[str]
        """
        errors = []

        if isinstance(X, pd.DataFrame):
            # Check schema
            if self.cfg.input.check_schema:
                for col, dtype in self.cfg.input.schema.features.items():
                    if col not in X.columns:
                        errors.append(f"Missing column: {col}")
                    elif str(X[col].dtype) != dtype:
                        errors.append(f"Invalid dtype for {col}: " f"expected {dtype}, got {X[col].dtype}")

            # Check ranges
            if self.cfg.input.check_range:
                for col, (min_val, max_val) in self.cfg.input.ranges.items():
                    if col in X.columns:
                        if X[col].min() < min_val or X[col].max() > max_val:
                            errors.append(f"Values out of range for {col}: " f"[{min_val}, {max_val}]")

        # Check missing values
        if self.cfg.input.check_missing:
            missing = np.isnan(X).sum()
            if missing.any():
                errors.append(f"Found {missing.sum()} missing values")

        return errors

    def _validate_performance(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Dict:
        """Validate model performance.

        :param model: Model to validate
        :type model: Any
        :param X: Training features
        :type X: Union[np.ndarray, pd.DataFrame]
        :param y: Training labels
        :type y: Union[np.ndarray, pd.Series]
        :param X_val: Validation features
        :type X_val: Optional[Union[np.ndarray, pd.DataFrame]]
        :param y_val: Validation labels
        :type y_val: Optional[Union[np.ndarray, pd.Series]]
        :return: Performance validation results
        :rtype: Dict
        """
        results = {}

        # Cross-validation if enabled
        if self.cfg.performance.cross_validation.enabled:
            cv_scores = {}
            for metric in self.cfg.performance.metrics:
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.cfg.performance.cross_validation.n_splits,
                    scoring=metric,
                )
                cv_scores[metric] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "passed": scores.mean() >= self.cfg.performance.thresholds[metric],
                }
            results["cross_validation"] = cv_scores

        # Validation set performance
        if X_val is not None and y_val is not None:
            val_scores = {}
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            for metric in self.cfg.performance.metrics:
                if metric == "accuracy":
                    score = accuracy_score(y_val, y_pred)
                elif metric == "precision":
                    score = precision_score(y_val, y_pred)
                elif metric == "recall":
                    score = recall_score(y_val, y_pred)
                elif metric == "f1":
                    score = f1_score(y_val, y_pred)
                elif metric == "roc_auc":
                    score = roc_auc_score(y_val, y_pred_proba)

                val_scores[metric] = {
                    "value": score,
                    "passed": score >= self.cfg.performance.thresholds[metric],
                }
            results["validation"] = val_scores

        return results

    def _validate_resources(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> Dict:
        """Validate resource usage.

        :param model: Model to validate
        :type model: Any
        :param X: Input features
        :type X: Union[np.ndarray, pd.DataFrame]
        :return: Resource validation results
        :rtype: Dict
        """
        import sys
        import time

        import psutil

        results = {}

        # Check memory usage
        if self.cfg.resources.check_memory:
            memory_usage = sys.getsizeof(model) / (1024 * 1024)  # MB
            results["memory"] = {
                "usage_mb": memory_usage,
                "passed": memory_usage <= self.cfg.resources.limits.max_memory_mb,
            }

        # Check inference time
        if self.cfg.resources.check_inference_time:
            start_time = time.time()
            model.predict(X[:100])  # Use small batch
            inference_time = (time.time() - start_time) * 1000  # ms
            results["inference_time"] = {
                "time_ms": inference_time,
                "passed": inference_time <= self.cfg.resources.limits.max_inference_time_ms,
            }

        return results

    def _validate_bias(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        protected_attributes: List[str],
    ) -> Dict:
        """Validate model bias.

        :param model: Model to validate
        :type model: Any
        :param X: Input features
        :type X: Union[np.ndarray, pd.DataFrame]
        :param y: True labels
        :type y: Union[np.ndarray, pd.Series]
        :param protected_attributes: Protected attribute names
        :type protected_attributes: List[str]
        :return: Bias validation results
        :rtype: Dict
        """
        results = {}

        if not isinstance(X, pd.DataFrame):
            return results

        y_pred = model.predict(X)

        for attr in protected_attributes:
            if attr not in X.columns:
                continue

            attr_results = {}
            groups = X[attr].unique()

            # Demographic parity
            group_predictions = {group: y_pred[X[attr] == group].mean() for group in groups}
            max_diff = max(group_predictions.values()) - min(group_predictions.values())
            attr_results["demographic_parity"] = {
                "value": max_diff,
                "passed": max_diff <= self.cfg.bias.thresholds.demographic_parity,
            }

            # Equal opportunity
            group_recall = {}
            for group in groups:
                mask = X[attr] == group
                group_recall[group] = recall_score(y[mask], y_pred[mask])
            max_diff = max(group_recall.values()) - min(group_recall.values())
            attr_results["equal_opportunity"] = {
                "value": max_diff,
                "passed": max_diff <= self.cfg.bias.thresholds.equal_opportunity,
            }

            results[attr] = attr_results

        return results

    def _validate_stability(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict:
        """Validate model stability.

        :param model: Model to validate
        :type model: Any
        :param X: Input features
        :type X: Union[np.ndarray, pd.DataFrame]
        :param y: True labels
        :type y: Union[np.ndarray, pd.Series]
        :return: Stability validation results
        :rtype: Dict
        """
        from scipy import stats

        results = {}

        # Prediction drift detection
        if self.cfg.stability.check_prediction_drift:
            # Split data into two parts
            mid = len(X) // 2
            y_pred_1 = model.predict_proba(X[:mid])[:, 1]
            y_pred_2 = model.predict_proba(X[mid:])[:, 1]

            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(y_pred_1, y_pred_2)
            results["prediction_drift"] = {
                "ks_statistic": ks_statistic,
                "p_value": p_value,
                "passed": p_value >= self.cfg.stability.drift_detection.threshold,
            }

        # Feature drift detection
        if self.cfg.stability.check_feature_drift and isinstance(X, pd.DataFrame):
            feature_drift = {}
            for column in X.columns:
                if X[column].dtype in [np.number]:
                    ks_statistic, p_value = stats.ks_2samp(X[column][:mid], X[column][mid:])
                    feature_drift[column] = {
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "passed": p_value >= self.cfg.stability.drift_detection.threshold,
                    }
            results["feature_drift"] = feature_drift

        return results

    def generate_validation_report(self) -> str:
        """Generate validation report.

        :return: Validation report
        :rtype: str
        """
        if not self.validation_results:
            return "No validation results available"

        report = ["Model Validation Report", "=" * 50, ""]

        # Input validation
        input_results = self.validation_results.get("input_validation", {})
        report.append("Input Validation:")
        report.append("-" * 20)
        report.append(f"Passed: {input_results.get('passed', False)}")
        if input_results.get("errors"):
            report.append("Errors:")
            for error in input_results["errors"]:
                report.append(f"  - {error}")
        report.append("")

        # Performance validation
        perf_results = self.validation_results.get("performance_validation", {})
        report.append("Performance Validation:")
        report.append("-" * 20)

        if "cross_validation" in perf_results:
            report.append("Cross-validation Results:")
            for metric, results in perf_results["cross_validation"].items():
                report.append(
                    f"  {metric}: {results['mean']:.4f} ± {results['std']:.4f} "
                    f"({'PASSED' if results['passed'] else 'FAILED'})"
                )

        if "validation" in perf_results:
            report.append("Validation Set Results:")
            for metric, results in perf_results["validation"].items():
                report.append(f"  {metric}: {results['value']:.4f} " f"({'PASSED' if results['passed'] else 'FAILED'})")
        report.append("")

        # Resource validation
        resource_results = self.validation_results.get("resource_validation", {})
        report.append("Resource Validation:")
        report.append("-" * 20)
        for resource, results in resource_results.items():
            report.append(
                f"  {resource}: {results.get('usage_mb', results.get('time_ms', 'N/A'))} "
                f"({'PASSED' if results['passed'] else 'FAILED'})"
            )
        report.append("")

        # Bias validation
        bias_results = self.validation_results.get("bias_validation", {})
        if bias_results:
            report.append("Bias Validation:")
            report.append("-" * 20)
            for attr, results in bias_results.items():
                report.append(f"  {attr}:")
                for metric, values in results.items():
                    report.append(
                        f"    {metric}: {values['value']:.4f} " f"({'PASSED' if values['passed'] else 'FAILED'})"
                    )
            report.append("")

        # Stability validation
        stability_results = self.validation_results.get("stability_validation", {})
        if stability_results:
            report.append("Stability Validation:")
            report.append("-" * 20)
            if "prediction_drift" in stability_results:
                pred_drift = stability_results["prediction_drift"]
                report.append(
                    f"  Prediction Drift: {pred_drift['ks_statistic']:.4f} "
                    f"(p={pred_drift['p_value']:.4f}) "
                    f"({'PASSED' if pred_drift['passed'] else 'FAILED'})"
                )
            if "feature_drift" in stability_results:
                report.append("  Feature Drift:")
                for feature, results in stability_results["feature_drift"].items():
                    report.append(
                        f"    {feature}: {results['ks_statistic']:.4f} "
                        f"(p={results['p_value']:.4f}) "
                        f"({'PASSED' if results['passed'] else 'FAILED'})"
                    )

        return "\n".join(report)
