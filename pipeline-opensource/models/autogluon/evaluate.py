"""
.. module:: models.autogluon.evaluate
   :synopsis: Evaluate the trained AutoGluon model and log metrics.

This script loads a trained AutoGluon model, evaluates it on the test dataset, and logs various metrics
including accuracy, precision, recall, ROC-AUC, and F1-score. It also generates and logs a confusion
matrix, ROC curve, and feature importances plot as image artifacts using DVCLive.
"""

import json
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Evaluate the trained AutoGluon model and log metrics.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during evaluation.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Use Hydra-managed paths
        test_data_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.preprocessed_data_test)
        )
        artifacts_dir = Path(to_absolute_path(cfg.paths.models.autogluon.artifacts))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        metrics_output_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.metrics_file)
        )
        confusion_matrix_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.confusion_matrix)
        )
        roc_curve_path = Path(to_absolute_path(cfg.paths.models.autogluon.roc_curve))
        feature_importance_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.feature_importances)
        )

        logger.info(f"Evaluating AutoGluon model {cfg.model.name}...")

        # Load preprocessed test data
        test_df = pd.read_csv(test_data_path)
        y_true = test_df["readmitted"]
        X_test = test_df.drop(columns=["readmitted"])

        # Load the trained AutoGluon model
        predictor = TabularPredictor.load(str(artifacts_dir))
        y_pred = predictor.predict(X_test)
        y_pred_proba = predictor.predict_proba(X_test)

        # Get the probability of the positive class
        positive_class = predictor.class_labels[
            -1
        ]  # Assuming the last class is positive
        y_pred_proba_positive = y_pred_proba[positive_class]

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba_positive, average="weighted"),
            "f1_score": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        # Initialize DVCLive
        with Live(dir=str(artifacts_dir)) as live:
            try:
                # Log evaluation parameters
                live.log_params(
                    {
                        "model_name": cfg.model.name,
                        "metrics_output_path": str(metrics_output_path),
                        "test_data_path": str(test_data_path),
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
                logger.info(f"Metrics JSON logged as artifact at {metrics_output_path}")

                # Generate and log Confusion Matrix plot
                conf_matrix = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 6))
                cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
                plt.colorbar(cax)

                class_labels = list(predictor.class_labels)
                tick_marks = np.arange(len(class_labels))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_labels)
                ax.set_yticklabels(class_labels)

                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")

                # Annotate the confusion matrix
                for (i, j), val in np.ndenumerate(conf_matrix):
                    ax.text(j, i, f"{val}", ha="center", va="center", color="red")

                plt.savefig(confusion_matrix_path)
                plt.close()

                # Log the confusion matrix image
                live.log_image("confusion", str(confusion_matrix_path))
                logger.info("Confusion matrix logged as image artifact.")

                # Generate and log ROC Curve plot
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba_positive)
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

                # Log the ROC curve image
                live.log_image("roc", str(roc_curve_path))
                logger.info("ROC curve logged as image artifact.")

                # Log feature importance
                logger.info("Calculating feature importance... This may take a while.")
                print("Calculating feature importance...", end="", flush=True)
                feature_importances = predictor.feature_importance(data=test_df)
                print(" Done!")

                # Assuming the feature importance is in the first column
                importance_column = feature_importances.columns[0]

                unused_features = feature_importances[
                    feature_importances[importance_column] == 0
                ].index.tolist()
                logger.info(f"Unused features: {unused_features}")

                sorted_importances = feature_importances[
                    feature_importances[importance_column] > 0
                ].sort_values(by=importance_column, ascending=False)
                plt.figure(figsize=(10, 6))
                sorted_importances.plot(kind="bar", y=importance_column)
                plt.title("Feature Importances")
                plt.tight_layout()
                plt.savefig(feature_importance_path)
                plt.close()

                # Log the feature importances image
                live.log_image("importance", str(feature_importance_path))
                logger.info("Feature importances logged as image artifact.")

            finally:
                # Always end the DVC Live run
                live.end()

        logger.info(f"Evaluation completed for AutoGluon model {cfg.model.name}.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
