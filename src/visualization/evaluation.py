import json
import os

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_paths = cfg.data.path
    model_output_path = to_absolute_path(cfg.model.model_output_path)
    test_data_path = to_absolute_path(f"{data_paths.processed}/test.csv")
    metrics_output_path = to_absolute_path(cfg.evaluation.metrics_output_path)

    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=["readmitted"])
    y_test = test_df["readmitted"]

    print("Loading model...")
    clf = joblib.load(model_output_path)

    print("Making predictions...")
    y_pred = clf.predict(X_test)

    print("Computing metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc * 100:.2f}%")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }

    print("Saving metrics...")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
