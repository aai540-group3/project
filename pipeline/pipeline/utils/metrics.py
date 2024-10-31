"""
Metrics Utilities
=================

.. module:: pipeline.utils.metrics
   :synopsis: Metrics calculation and tracking utilities

.. moduleauthor:: aai540-group3
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate standard classification metrics.

    :param y_true: True labels
    :type y_true: np.ndarray
    :param y_pred: Predicted labels
    :type y_pred: np.ndarray
    :param y_pred_proba: Predicted probabilities
    :type y_pred_proba: Optional[np.ndarray]
    :return: Dictionary of calculated metrics
    :rtype: Dict[str, float]
    :raises ValueError: If input shapes don't match
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics.update(
            {
                "roc_auc": roc_auc_score(y_true, y_pred_proba),
                "avg_precision": average_precision_score(y_true, y_pred_proba),
            }
        )

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calculate confusion matrix and derived metrics.

    :param y_true: True labels
    :type y_true: np.ndarray
    :param y_pred: Predicted labels
    :type y_pred: np.ndarray
    :return: Tuple of confusion matrix and derived metrics
    :rtype: Tuple[np.ndarray, Dict[str, float]]
    :raises ValueError: If input shapes don't match
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    derived_metrics = {
        "true_negative_rate": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }

    return cm, derived_metrics
