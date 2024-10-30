# src/utils/visualization.py
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

def set_style() -> None:
    """Set plotting style."""
    plt.style.use("seaborn")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    labels: Optional[List[str]] = None,
) -> None:
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Output path
        labels: Class labels
    """
    set_style()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC curve.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Output path
    """
    set_style()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(
    importance: pd.Series,
    output_path: Path,
    n_features: int = 20,
) -> None:
    """Plot feature importance.

    Args:
        importance: Feature importance series
        output_path: Output path
        n_features: Number of features to plot
    """
    set_style()

    plt.figure(figsize=(10, 6))
    importance.nlargest(n_features).plot(kind="barh")
    plt.title(f"Top {n_features} Feature Importance")
    plt.xlabel("Importance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot model comparison.

    Args:
        metrics: Dictionary of model metrics
        output_path: Output path
    """
    set_style()

    df = pd.DataFrame(metrics).T

    plt.figure(figsize=(12, 6))
    df.plot(kind="bar")
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
