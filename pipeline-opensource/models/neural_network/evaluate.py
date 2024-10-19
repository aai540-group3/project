"""
.. module:: models.neural_network.evaluate
   :synopsis: Evaluate the trained neural network model and log metrics, including feature importances.

This script evaluates the neural network model on the test data and includes:

- Loading the trained model and test data.
- Making predictions and calculating evaluation metrics.
- Generating and saving confusion matrix, ROC curve, and feature importances plots.
- Computing feature importances using SHAP values.
- Logging metrics and artifacts using DVCLive.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from dvclive import Live

from tensorflow.keras.models import load_model
import shap
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bool_to_int(X):
    """
    Converts boolean features to integers (True -> 1, False -> 0).
    """
    return X.astype(int)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Main function to evaluate the neural network model.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during evaluation.
    """
    try:
        cfg.model.name = "neural_network"
        logger.info(f"Starting evaluation for neural network model {cfg.model.name}...")
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Paths
        artifacts_dir = Path(
            to_absolute_path(cfg.paths.models.neural_network.artifacts)
        )
        model_path = Path(to_absolute_path(cfg.paths.models.neural_network.model_file))
        metrics_output_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.metrics_file)
        )
        confusion_matrix_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.confusion_matrix)
        )
        roc_curve_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.roc_curve)
        )
        feature_importance_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.feature_importances)
        )
        preprocessor_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.preprocessor)
        )
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Load preprocessed test data
        X_test = np.load(artifacts_dir / "X_test.npy")
        y_test = np.load(artifacts_dir / "y_test.npy")

        # Load the trained neural network model
        model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load the preprocessor to get feature names
        preprocessor = joblib.load(preprocessor_path)
        # Retrieve feature names
        try:
            feature_names_num = preprocessor.named_transformers_[
                "num"
            ].get_feature_names_out()
        except AttributeError:
            feature_names_num = preprocessor.transformers_[0][
                2
            ]  # Use the column names directly

        feature_names_bin = preprocessor.transformers_[1][2]
        feature_names = list(feature_names_num) + list(feature_names_bin)

        # Make predictions
        y_pred_proba = model.predict(X_test).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate evaluation metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": 2 * (precision * recall) / (precision + recall + 1e-7),
        }

        # Initialize DVCLive
        with Live(dir=str(artifacts_dir)) as live:
            try:
                # Log evaluation parameters
                live.log_params(
                    {
                        "model_name": cfg.model.name,
                        "metrics_output_path": str(metrics_output_path),
                        "evaluation_date": pd.Timestamp.now().isoformat(),
                        "dataset_version": cfg.dataset.version,
                    }
                )

                # Log evaluation metrics
                for metric_name, metric_value in metrics.items():
                    live.log_metric(metric_name, metric_value)
                logger.info(f"Logged Metrics: {metrics}")

                # Save metrics to JSON
                with open(metrics_output_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Metrics saved to {metrics_output_path}")

                # Log metrics JSON as artifact
                live.log_artifact(
                    str(metrics_output_path),
                    type="metrics",
                    name=f"{cfg.model.name}_metrics",
                )

                # Generate and log Confusion Matrix plot
                conf_matrix = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 5))
                plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                tick_marks = [0, 1]
                plt.xticks(tick_marks, tick_marks)
                plt.yticks(tick_marks, tick_marks)
                plt.xlabel("Predicted")
                plt.ylabel("True")

                # Add labels to each cell
                thresh = conf_matrix.max() / 2.0
                for i, j in np.ndindex(conf_matrix.shape):
                    plt.text(
                        j,
                        i,
                        format(conf_matrix[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black",
                    )

                plt.tight_layout()
                plt.savefig(confusion_matrix_path)
                plt.close()
                live.log_image("confusion_matrix", str(confusion_matrix_path))
                logger.info(f"Confusion matrix plot saved to {confusion_matrix_path}")

                # Generate and log ROC Curve plot
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc_value = auc(fpr, tpr)

                plt.figure()
                plt.plot(
                    fpr,
                    tpr,
                    color="darkorange",
                    lw=2,
                    label=f"ROC curve (area = {roc_auc_value:.2f})",
                )
                plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic")
                plt.legend(loc="lower right")
                plt.savefig(roc_curve_path)
                plt.close()
                live.log_image("roc_curve", str(roc_curve_path))
                logger.info(f"ROC curve plot saved to {roc_curve_path}")

                # Compute and plot Feature Importances using SHAP
                logger.info("Calculating feature importances using SHAP...")

                # Compute and plot Feature Importances using SHAP
                logger.info("Calculating feature importances using SHAP...")

                # Due to computational constraints, use a subset of the data
                X_sample = X_test[:100]
                background = X_test[:100]

                # Create SHAP explainer
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(X_sample)

                # Debugging output
                logger.info(f"Type of shap_values: {type(shap_values)}")
                logger.info(f"shap_values shape: {shap_values.shape}")
                logger.info(f"X_sample shape: {X_sample.shape}")
                logger.info(f"Feature names length: {len(feature_names)}")

                # Adjust shap_values for plotting
                shap_values_to_plot = np.squeeze(
                    shap_values
                )  # Remove the singleton dimension
                logger.info(
                    f"After squeezing, shap_values shape: {shap_values_to_plot.shape}"
                )

                # Ensure shapes match
                assert (
                    shap_values_to_plot.shape == X_sample.shape
                ), f"Mismatch in shapes between shap_values ({shap_values_to_plot.shape}) and X_sample ({X_sample.shape}) after squeezing"

                # Plot feature importances
                shap.summary_plot(
                    shap_values_to_plot,
                    features=X_sample,
                    feature_names=feature_names,
                    show=False,
                )
                plt.savefig(feature_importance_path)
                plt.close()
                live.log_image("feature_importances", str(feature_importance_path))
                logger.info(
                    f"Feature importances plot saved to {feature_importance_path}"
                )
            finally:
                live.end()

        logger.info("Evaluation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
