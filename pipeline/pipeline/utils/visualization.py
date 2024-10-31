"""
Visualization Utilities
====================

.. module:: pipeline.utils.visualization
   :synopsis: Plotting and visualization utilities

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from .logging import get_logger


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probas: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot ROC curves for multiple models.

    :param y_true: True labels
    :type y_true: np.ndarray
    :param y_pred_probas: Dictionary of model predictions
    :type y_pred_probas: Dict[str, np.ndarray]
    :param save_path: Path to save plot
    :type save_path: Optional[Path]
    :param figsize: Figure size
    :type figsize: tuple
    """
    plt.figure(figsize=figsize)

    for model_name, y_pred_proba in y_pred_probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_preds: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (15, 5),
) -> None:
    """Plot confusion matrices for multiple models.

    :param y_true: True labels
    :type y_true: np.ndarray
    :param y_preds: Dictionary of model predictions
    :type y_preds: Dict[str, np.ndarray]
    :param save_path: Path to save plot
    :type save_path: Optional[Path]
    :param figsize: Figure size
    :type figsize: tuple
    """
    n_models = len(y_preds)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, y_pred) in zip(axes, y_preds.items()):
        cm = pd.crosstab(y_true, y_pred, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=ax)
        ax.set_title(f"{model_name}\nConfusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_feature_importance(
    feature_importance: Dict[str, Dict[str, float]],
    top_n: int = 20,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> None:
    """Plot feature importance comparison.

    :param feature_importance: Feature importance scores by model
    :type feature_importance: Dict[str, Dict[str, float]]
    :param top_n: Number of top features to show
    :type top_n: int
    :param save_path: Path to save plot
    :type save_path: Optional[Path]
    :param figsize: Figure size
    :type figsize: tuple
    """
    # Combine and normalize feature importance scores
    combined_scores = {}
    for model_name, scores in feature_importance.items():
        if scores is None:
            continue
        normalized_scores = StandardScaler().fit_transform(np.array(list(scores.values())).reshape(-1, 1)).ravel()
        combined_scores[model_name] = dict(zip(scores.keys(), normalized_scores))

    if not combined_scores:
        logger.warning("No feature importance scores available")
        return

    # Get top features across all models
    all_scores = pd.DataFrame(combined_scores)
    mean_importance = all_scores.mean(axis=1)
    top_features = mean_importance.nlargest(top_n).index

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(all_scores.loc[top_features], annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Feature Importance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Feature")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
