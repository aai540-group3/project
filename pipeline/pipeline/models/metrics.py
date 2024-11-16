import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
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
    """Class for computing and storing classification metrics and feature importance.

    This class provides functionality for computing various classification metrics,
    generating visualizations, computing feature importances, and integrating SHAP
    explanations for models.

    :param y_true: Ground truth labels
    :type y_true: Optional[List[int]]
    :param y_proba: Predicted probabilities (binary: float list, multiclass: float list of lists)
    :type y_proba: Optional[Union[List[float], List[List[float]]]]
    :param y_pred: Predicted class labels
    :type y_pred: Optional[List[int]]
    :param model: Trained classifier model (base estimator)
    :type model: Optional[Any]
    :param X: Feature data used for predictions
    :type X: Optional[pd.DataFrame]
    """

    y_true: Optional[List[int]] = field(default=None, repr=False)
    y_proba: Optional[Union[List[float], List[List[float]]]] = field(default=None, repr=False)
    y_pred: Optional[List[int]] = field(default=None, repr=False)
    model: Optional[Any] = field(default=None, repr=False)
    X: Optional[pd.DataFrame] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the Metrics instance."""
        self.binary = self.is_binary()

    def is_binary(self) -> bool:
        """Check if the target variable is binary.

        :return: True if the target variable has two unique values, otherwise False.
        :rtype: bool
        """
        # Ensure y_true is a Series for consistent attribute access
        y_true = pd.Series(self.y_true) if isinstance(self.y_true, list) else self.y_true
        return len(set(y_true)) == 2 if not y_true.empty else False

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

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Union[List[int], List[float]]],
        model: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
    ) -> "Metrics":
        """Create a Metrics instance from a dictionary.

        :param data: Dictionary containing 'y_true', 'y_pred', and 'y_proba' as keys
        :type data: Dict[str, Union[List[int], List[float]]]
        :param model: Trained classifier model
        :type model: Optional[Any]
        :param X: Feature data used for predictions
        :type X: Optional[pd.DataFrame]
        :return: Instance of Metrics class populated with values from the dictionary
        :rtype: Metrics
        """
        y_true = data.get("y_true")
        y_pred = data.get("y_pred")
        y_proba = data.get("y_proba")

        logger.info("Creating Metrics instance from dictionary.")
        return cls(y_true=y_true, y_pred=y_pred, y_proba=y_proba, model=model, X=X)

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
                    "precision": precision_score(self.y_true, self.y_pred, average="binary", zero_division=0),
                    "recall": recall_score(self.y_true, self.y_pred, average="binary", zero_division=0),
                    "f1_score": f1_score(self.y_true, self.y_pred, average="binary", zero_division=0),
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
        y_true_np = np.array(y_true)
        y_proba_np = np.array(y_proba)

        # Plot distributions for positive and negative classes
        sns.kdeplot(data=y_proba_np[y_true_np == 1], label="Positive Class", color="forestgreen")
        sns.kdeplot(data=y_proba_np[y_true_np == 0], label="Negative Class", color="crimson")

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
        """Generate and save feature importance visualization and data.

        :param feature_importance: DataFrame with feature names and importance scores
        :type feature_importance: pd.DataFrame
        :param save_path: Path where the plot and CSV will be saved (without extension)
        :type save_path: Path
        :param title: Title for the plot
        :type title: str

        .. note::
        The DataFrame must have 'feature' and 'importance' columns.
        The method will save both a .png visualization and a .csv file.
        If save_path includes a .png extension, it will be removed before saving.
        """
        if feature_importance is None or feature_importance.empty:
            logger.warning("Feature importance data is empty. Skipping plot and CSV export.")
            return

        # Remove .png extension if present and create base path
        base_path = save_path.with_suffix("") if save_path.suffix == ".png" else save_path

        # Prepare paths for both files
        plot_path = base_path.with_suffix(".png")
        csv_path = base_path.with_suffix(".csv")

        # Get top features
        top_features = feature_importance.nlargest(20, "importance")

        # Save the plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x="importance", y="feature")
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Feature importance plot saved at '{plot_path}'")

        # Save the CSV
        top_features.to_csv(csv_path, index=False)
        logger.info(f"Feature importance data saved at '{csv_path}'")

    @staticmethod
    def plot_shap_summary(
        shap_values: shap.Explanation,
        X: pd.DataFrame,
        save_path: Path,
        plot_type: str = "dot",
        title: str = "SHAP Summary Plot",
    ) -> None:
        """Generate and save a SHAP summary plot.

        :param shap_values: SHAP values explanation object
        :type shap_values: shap.Explanation
        :param X: Feature data used for SHAP
        :type X: pd.DataFrame
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param plot_type: Type of SHAP plot ('dot', 'bar', etc.)
        :type plot_type: str
        :param title: Title for the plot
        :type title: str
        """
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"SHAP summary plot saved at '{save_path}'.")
        except Exception as e:
            logger.error(f"Error generating SHAP summary plot: {e}")

    @staticmethod
    def plot_shap_dependence(
        shap_values: shap.Explanation, feature: str, X: pd.DataFrame, save_path: Path, title: Optional[str] = None
    ) -> None:
        """Generate and save a SHAP dependence plot for a specific feature.

        :param shap_values: SHAP values explanation object
        :type shap_values: shap.Explanation
        :param feature: Feature name for dependence plot
        :type feature: str
        :param X: Feature data used for SHAP
        :type X: pd.DataFrame
        :param save_path: Path where the plot will be saved
        :type save_path: Path
        :param title: Title for the plot
        :type title: Optional[str]
        """
        try:
            plt.figure(figsize=(10, 8))
            shap.dependence_plot(feature, shap_values.values, X, show=False)
            plt.title(title if title else f"SHAP Dependence Plot for {feature}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"SHAP dependence plot for '{feature}' saved at '{save_path}'.")
        except Exception as e:
            logger.error(f"Error generating SHAP dependence plot for '{feature}': {e}")

    def compute_model_based_importance(self) -> Optional[pd.DataFrame]:
        """Compute feature importance using the trained model's inherent feature_importances_ attribute.

        :return: DataFrame containing features and their importance scores
        :rtype: Optional[pd.DataFrame]
        """
        if self.model is None or self.X is None:
            logger.warning("Model or feature data not provided. Cannot compute model-based feature importance.")
            return None

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_names = self.X.columns
            feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            logger.info("Computed model-based feature importance.")
            return feature_importance_df
        else:
            logger.warning("The provided model does not have a 'feature_importances_' attribute.")
            return None

    def compute_permutation_importance_metrics(
        self, scoring: str = "accuracy", n_repeats: int = 10, random_state: int = 42
    ) -> Optional[pd.DataFrame]:
        """Compute permutation feature importance.

        :param scoring: Scoring metric to evaluate feature importance
        :type scoring: str
        :param n_repeats: Number of times to permute a feature
        :type n_repeats: int
        :param random_state: Random state for reproducibility
        :type random_state: int
        :return: DataFrame containing features and their permutation importance scores
        :rtype: Optional[pd.DataFrame]
        """
        if self.model is None or self.X is None or self.y_true is None:
            logger.warning(
                "Model, feature data, or true labels not provided. Cannot compute permutation feature importance."
            )
            return None

        try:
            result = permutation_importance(
                self.model, self.X, self.y_true, scoring=scoring, n_repeats=n_repeats, random_state=random_state
            )
            feature_importance_df = pd.DataFrame({"feature": self.X.columns, "importance": result.importances_mean})
            logger.info("Computed permutation feature importance.")
            return feature_importance_df
        except Exception as e:
            logger.error(f"Error computing permutation feature importance: {e}")
            return None

    def get_feature_importance(self, method: str = "model_based") -> Optional[pd.DataFrame]:
        """Retrieve feature importance based on the specified method.

        :param method: Method to compute feature importance ('model_based' or 'permutation')
        :type method: str
        :return: DataFrame containing features and their importance scores
        :rtype: Optional[pd.DataFrame]
        """
        if method == "model_based":
            feature_importance = self.compute_model_based_importance()
            if feature_importance is None:
                logger.warning("Model-based feature importance not available. Falling back to permutation importance.")
                feature_importance = self.compute_permutation_importance_metrics()
        elif method == "permutation":
            feature_importance = self.compute_permutation_importance_metrics()
        else:
            logger.error(f"Unsupported feature importance method: {method}")
            feature_importance = None
        return feature_importance

    def compute_shap_values(self, background: Optional[pd.DataFrame] = None) -> Optional[shap.Explanation]:
        """Compute SHAP values for the model and feature data.

        :param background: Background dataset for SHAP (used for deep models)
        :type background: Optional[pd.DataFrame]
        :return: SHAP values explanation object
        :rtype: Optional[shap.Explanation]
        """
        # Ensure that both model and feature data are available
        if self.model is None or self.X is None or self.X.empty:
            logger.warning("Model or feature data not provided. Cannot compute SHAP values.")
            return None

        try:
            explainer = shap.Explainer(self.model, background if background is not None else self.X)
            shap_values = explainer(self.X)
            logger.info("Computed SHAP values.")
            return shap_values
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None

    def generate_shap_plots(
        self, shap_values: shap.Explanation, save_dir: Path, title: Optional[str] = "SHAP Summary Plot"
    ) -> None:
        """Generate SHAP summary and dependence plots.

        :param shap_values: SHAP values explanation object
        :type shap_values: shap.Explanation
        :param save_dir: Directory where SHAP plots will be saved
        :type save_dir: Path
        :param title: Title for the summary plot
        :type title: Optional[str]
        """
        if shap_values is None:
            logger.warning("SHAP values not provided. Cannot generate SHAP plots.")
            return

        save_dir.mkdir(parents=True, exist_ok=True)

        # SHAP Summary Plot
        self.plot_shap_summary(
            shap_values=shap_values, X=self.X, save_path=save_dir / "shap_summary.png", plot_type="dot", title=title
        )

        # SHAP Dependence Plots for Top Features
        try:
            # Determine top features based on mean absolute SHAP values
            shap_abs = np.abs(shap_values.values).mean(axis=0)
            top_indices = np.argsort(shap_abs)[-10:]
            top_features = self.X.columns[top_indices]

            for feature in top_features:
                self.plot_shap_dependence(
                    shap_values=shap_values,
                    feature=feature,
                    X=self.X,
                    save_path=save_dir / f"shap_dependence_{feature}.png",
                    title=f"SHAP Dependence Plot for {feature}",
                )
        except Exception as e:
            logger.error(f"Error generating SHAP dependence plots: {e}")

    def get_classification_report(self) -> Dict[str, Any]:
        """Generate a classification report as a dictionary.

        :return: Classification report with precision, recall, and f1-score for each class
        :rtype: Dict[str, Any]
        """
        if self.y_true and self.y_pred:
            report = classification_report(self.y_true, self.y_pred, output_dict=True)
            logger.info("Generated classification report.")
            return report
        else:
            logger.warning("True and predicted labels are required to generate classification report.")
            return {}
