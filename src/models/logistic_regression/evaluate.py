import json
import logging
from pathlib import Path

import hydra
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dvclive import Live
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        data_paths = cfg.dataset.path
        test_data_path = Path(to_absolute_path(f"{data_paths.processed}/test.csv"))
        model_path = Path(to_absolute_path(f"models/{cfg.model.name}/model.pkl"))
        metrics_output_path = Path(to_absolute_path(f"reports/metrics/{cfg.model.name}_metrics.json"))

        logger.info(f"Evaluating Logistic Regression model {cfg.model.name}...")

        # Load test data
        test_df = pd.read_csv(test_data_path)
        y_true = test_df["readmitted"]
        X_test = test_df.drop(columns=["readmitted"])

        # Load the trained Logistic Regression model
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

        # Calculate evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba, average='weighted'),
            "f1_score": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Initialize DVCLive
        with Live() as live:
            # Log evaluation parameters
            live.log_params({
                "model_name": cfg.model.name,
                "metrics_output_path": str(metrics_output_path),
                "test_data_path": str(test_data_path),
                "evaluation_date": pd.Timestamp.now().isoformat(),
                "dataset_version": cfg.dataset.version
            })

            # Log evaluation metrics
            for metric_name, metric_value in metrics.items():
                live.log_metric(metric_name, metric_value)
            logger.info(f"Logged Metrics: {metrics}")

            # Save metrics to JSON
            metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_output_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_output_path}")

            # Log metrics JSON as artifact
            live.log_artifact(str(metrics_output_path), type="metrics", name=f"{cfg.model.name}_metrics")
            logger.info(f"Metrics JSON logged as artifact at {metrics_output_path}")

            # Generate and log Confusion Matrix plot
            conf_matrix = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
            plt.colorbar(cax)

            ax.set_xticklabels([''] + [str(i) for i in range(2)])  # Assuming binary classification
            ax.set_yticklabels([''] + [str(i) for i in range(2)])

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Annotate the confusion matrix
            for (i, j), val in np.ndenumerate(conf_matrix):
                ax.text(j, i, f"{val}", ha='center', va='center', color='red')

            conf_matrix_path = metrics_output_path.parent / "confusion_matrix.png"
            plt.savefig(conf_matrix_path)
            plt.close()

            # Log the confusion matrix image
            live.log_image("confusion", str(conf_matrix_path))
            logger.info("Confusion matrix logged as image artifact.")

            # Generate and log ROC Curve plot
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc_value = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange',
                     lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            roc_curve_path = metrics_output_path.parent / "roc_curve.png"
            plt.savefig(roc_curve_path)
            plt.close()

            # Log the ROC curve image
            live.log_image("roc", str(roc_curve_path))
            logger.info("ROC curve logged as image artifact.")

            # Log feature importance (coefficients for logistic regression)
            feature_importances = pd.Series(model.coef_[0], index=X_test.columns)
            sorted_importances = feature_importances.abs().sort_values(ascending=False)
            plt.figure(figsize=(10, 6))
            sorted_importances.plot(kind='bar')
            plt.title('Feature Importances (Absolute Coefficients)')
            plt.tight_layout()
            feature_importance_path = metrics_output_path.parent / "feature_importances.png"
            plt.savefig(feature_importance_path)
            plt.close()

            # Log the feature importances image
            live.log_image("importance", str(feature_importance_path))
            logger.info("Feature importances logged as image artifact.")

        logger.info(f"Evaluation completed for Logistic Regression model {cfg.model.name}.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise

if __name__ == "__main__":
    main()