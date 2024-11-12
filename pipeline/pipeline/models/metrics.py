"""
Metrics Implementation
=====================

.. module:: pipeline.models.metrics
   :synopsis: Implementation of metrics computation and visualization for machine learning models

.. moduleauthor:: aai540-group3
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class Metrics:
    """Class for computing and storing classification metrics.

    This class provides functionality for computing various classification metrics,
    generating visualizations, and exporting results. It handles both binary and
    multiclass classification scenarios.

    :param y_true: Ground truth labels
    :type y_true: Optional[List[int]]
    :param y_proba: Predicted probabilities (binary: float list, multiclass: float list of lists)
    :type y_proba: Optional[Union[List[float], List[List[float]]]]
    :param y_pred: Predicted class labels
    :type y_pred: Optional[List[int]]

    :ivar binary: Indicates if the classification is binary (automatically determined)
    :type binary: bool
    """

    y_true: Optional[List[int]] = field(default=None, repr=False)
    y_proba: Optional[Union[List[float], List[List[float]]]] = field(default=None, repr=False)
    y_pred: Optional[List[int]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the Metrics instance."""
        self.binary = self.is_binary()

    def is_binary(self) -> bool:
        """Check if the classification task is binary.

        :return: True if classification is binary (2 classes), False otherwise
        :rtype: bool
        """
        return len(set(self.y_true)) == 2 if self.y_true else False

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert metrics to a dictionary format.

        :return: Dictionary containing all computed metrics
        :rtype: Dict[str, Optional[float]]
        """
        return self.get_metrics()

    def to_json(self, filepath: str) -> None:
        """Save metrics to a JSON file.

        :param filepath: Path where the JSON file will be saved
        :type filepath: str
        """
        with open(filepath, "w") as f:
            json.dump(self.get_metrics(), f, indent=4)
        logger.info(f"Metrics saved to JSON at '{filepath}'.")

    def get_metrics(self) -> Dict[str, Optional[float]]:
        """Compute and return classification metrics.

        :return: Dictionary containing various classification metrics
        :rtype: Dict[str, Optional[float]]
        """
        metrics = {}
        if self.y_true and self.y_pred:
            metrics.update(
                {
                    "accuracy": accuracy_score(self.y_true, self.y_pred),
                    "precision": precision_score(self.y_true, self.y_pred, average="binary"),
                    "recall": recall_score(self.y_true, self.y_pred, average="binary"),
                    "f1_score": f1_score(self.y_true, self.y_pred, average="binary"),
                    "specificity": self.calculate_specificity(),
                    "balanced_accuracy": balanced_accuracy_score(self.y_true, self.y_pred),
                    "mcc": matthews_corrcoef(self.y_true, self.y_pred),
                }
            )

        if self.y_proba and self.y_true and self.is_binary():
            try:
                metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_proba)
                metrics["average_precision"] = average_precision_score(self.y_true, self.y_proba)
                metrics["log_loss"] = log_loss(self.y_true, self.y_proba)
                metrics["brier_score"] = brier_score_loss(self.y_true, self.y_proba)
            except ValueError as e:
                logger.error(f"Error computing probabilistic metrics: {e}")

        return metrics

    def calculate_specificity(self) -> float:
        """Calculate the specificity metric for binary classification.

        :return: Specificity score (true negative rate)
        :rtype: float
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        if cm.shape == (2, 2):
            tn, fp, _, _ = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        logger.warning("Specificity is not defined for non-binary classification.")
        return 0.0

    @staticmethod
    def plot_confusion_matrix(
        y_true: List[int], y_pred: List[int], save_path: Path, title: str = "Confusion Matrix"
    ) -> None:
        """Generate and save a confusion matrix visualization.

        :param y_true: Ground truth labels
        :type y_true: List[int]
        :param y_pred: Predicted labels
        :type y_pred: List[int]
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param title: Title for the plot
        :type title: str
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        )

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved at '{save_path}'.")

    @staticmethod
    def plot_roc_curve(y_true: List[int], y_proba: List[float], save_path: Path, title: str = "ROC Curve") -> None:
        """Generate and save a ROC curve plot.

        :param y_true: Ground truth labels
        :type y_true: List[int]
        :param y_proba: Predicted probabilities for the positive class
        :type y_proba: List[float]
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param title: Title for the plot
        :type title: str
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"ROC curve plot saved at '{save_path}'.")

    @staticmethod
    def plot_precision_recall_curve(
        y_true: List[int], y_proba: List[float], save_path: Path, title: str = "Precision-Recall Curve"
    ) -> None:
        """Generate and save a precision-recall curve plot.

        :param y_true: Ground truth labels
        :type y_true: List[int]
        :param y_proba: Predicted probabilities for the positive class
        :type y_proba: List[float]
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param title: Title for the plot
        :type title: str
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {avg_precision:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Precision-Recall curve plot saved at '{save_path}'.")

    @staticmethod
    def plot_calibration_curve(
        y_true: List[int], y_proba: List[float], save_path: Path, n_bins: int = 10, title: str = "Calibration Plot"
    ) -> None:
        """Generate and save a calibration curve plot.

        :param y_true: Ground truth labels
        :type y_true: List[int]
        :param y_proba: Predicted probabilities for the positive class
        :type y_proba: List[float]
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param n_bins: Number of bins for calibration
        :type n_bins: int
        :param title: Title for the plot
        :type title: str
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker="o", color="darkorange", label="Calibration curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="navy", label="Perfect calibration")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Calibration plot saved at '{save_path}'.")

    @staticmethod
    def plot_probability_distribution(
        y_true: List[int], y_proba: List[float], save_path: Path, title: str = "Probability Distribution"
    ) -> None:
        """Generate and save probability distribution plots.

        :param y_true: Ground truth labels
        :type y_true: List[int]
        :param y_proba: Predicted probabilities for the positive class
        :type y_proba: List[float]
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param title: Title for the plot
        :type title: str
        """
        plt.figure(figsize=(10, 6))

        # Convert to numpy arrays for easier manipulation
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        # Plot distributions for positive and negative classes
        sns.kdeplot(data=y_proba[y_true == 1], label="Positive Class", color="forestgreen")
        sns.kdeplot(data=y_proba[y_true == 0], label="Negative Class", color="crimson")

        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Probability distribution plot saved at '{save_path}'.")

    @staticmethod
    def plot_feature_importance(
        feature_importance: pd.DataFrame, save_path: Path, title: str = "Top 20 Features by Importance"
    ) -> None:
        """Generate and save a feature importance visualization.

        :param feature_importance: DataFrame with feature names and importance scores
        :type feature_importance: pd.DataFrame
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param title: Title for the plot
        :type title: str

        .. note::
           The DataFrame must have 'feature' and 'importance' columns.
        """
        top_features = feature_importance.nlargest(20, "importance")

        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x="importance", y="feature")
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Feature importance plot saved at '{save_path}'.")

    def get_classification_report(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Generate a classification report for precision, recall, f1-score, and support.

        :return: Dictionary with classification report metrics for each class
        :rtype: Optional[Dict[str, Dict[str, float]]]
        """
        if self.y_true and self.y_pred:
            return classification_report(self.y_true, self.y_pred, output_dict=True)
        logger.warning("Classification report could not be generated due to missing data.")
        return None
