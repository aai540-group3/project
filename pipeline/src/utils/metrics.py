# src/utils/metrics.py
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, float]:
    """Calculate classification metrics.

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
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }


def calculate_model_comparison(
    models: Dict,
    test_data: np.ndarray,
    metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for model comparison.

    Args:
        models: Dictionary of models
        test_data: Test dataset
        metrics: List of metrics to calculate

    Returns:
        Dictionary of metrics for each model
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(test_data)
        y_pred_proba = model.predict_proba(test_data)[:, 1]

        results[name] = calculate_metrics(
            test_data.target,
            y_pred,
            y_pred_proba
        )

    return results
