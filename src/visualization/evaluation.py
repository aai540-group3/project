import logging
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)

from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_int(X):
    return X.astype(int)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.data.path
        model_output_path = Path(to_absolute_path(cfg.model.model_output_path))
        test_data_path = Path(to_absolute_path(f"{data_paths.processed}/test.csv"))
        preprocessor_path = Path(
            to_absolute_path(f"{data_paths.processed}/preprocessor.joblib")
        )
        feature_names_path = Path(
            to_absolute_path(f"{data_paths.processed}/feature_names.joblib")
        )

        logger.info("Loading test data...")
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop(columns=["readmitted"])
        y_test = test_df["readmitted"]

        logger.info("Loading model, preprocessor, and feature names...")
        clf = joblib.load(model_output_path)
        preprocessor = joblib.load(preprocessor_path)
        feature_names = joblib.load(feature_names_path)

        logger.info("Preprocessing test data...")
        X_test_preprocessed = preprocessor.transform(X_test)
        X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names)

        logger.info("Making predictions...")
        y_pred = clf.predict(X_test_preprocessed)

        logger.info("Computing metrics...")
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "roc_auc": roc_auc_score(y_test, y_pred),
        }

        for metric, value in metrics.items():
            logger.info(f"{metric.capitalize()}: {value * 100:.2f}%")

        logger.info("Logging metrics using DVCLive...")
        with Live(dir="dvclive_evaluate") as live:
            for metric, value in metrics.items():
                live.log_metric(metric, value)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()