# models/autogluon/train.py
import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from autogluon.tabular import TabularPredictor

from dvclive import Live

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import shap  # Import SHAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train and evaluate AutoGluon model."""
    logger.info("Training AutoGluon model...")

    # Calculate hash of training data
    with open("data/processed/train.csv", "rb") as f:
        input_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()

    live = Live(dir="models/autogluon", dvcyaml=False)

    # Check if model exists and hash matches
    skip_training = False
    if os.path.exists("models/autogluon/model.pkl") and os.path.exists("models/autogluon/input_hash.txt"):
        with open("models/autogluon/input_hash.txt", "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == input_hash:
            logger.info("AutoGluon model already exists. Skipping training.")
            predictor = joblib.load("models/autogluon/model.pkl")
            skip_training = True

    if not skip_training:
        train_data = pd.read_csv("data/processed/train.csv")

        # Split into train/test/validate
        X = train_data.drop(columns=["readmitted"])
        y = train_data["readmitted"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        # Combine features and target
        train_data = pd.concat([X_train, y_train], axis=1)

        predictor = TabularPredictor(
            label="readmitted", path="models/autogluon", problem_type="binary"
        )
        predictor.fit(
            train_data=train_data, # Pass the combined DataFrame
            time_limit=3600,
            presets="best_quality",
            hyperparameters={"GBM": {"num_boost_round": 100}},
            verbosity=2,
        )
        # Save the model
        joblib.dump(predictor, "models/autogluon/model.pkl")
        logger.info(f"AutoGluon model saved to models/autogluon/model.pkl")
        # Save the hash
        with open("models/autogluon/input_hash.txt", "w") as f:
            f.write(input_hash)

    # Evaluate AutoGluon
    logger.info("Evaluating AutoGluon model...")
    y_pred = predictor.predict(X_test)
    y_pred_proba = predictor.predict_proba(X_test)
    positive_class = predictor.class_labels[-1]
    y_pred_proba_positive = y_pred_proba[positive_class]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba_positive, average="weighted"),
        "f1_score": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    with open("models/autogluon/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    for metric_name, metric_value in metrics.items():
        live.log_metric(metric_name, metric_value)

    # Log confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(predictor.class_labels))
    plt.xticks(tick_marks, predictor.class_labels)
    plt.yticks(tick_marks, predictor.class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/autogluon/confusion_matrix.png")
    plt.close()
    live.log_image("confusion_matrix", "models/autogluon/confusion_matrix.png")

    # Log ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_positive)
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
    plt.savefig("models/autogluon/roc_curve.png")
    plt.close()
    live.log_image("roc_curve", "models/autogluon/roc_curve.png")

    # Compute and plot feature importances using SHAP
    logger.info("Calculating feature importances using SHAP...")
    explainer = shap.Explainer(predictor.predict, X_train)
    shap_values = explainer(X_test)

    # Summary plot (bar plot of mean absolute SHAP values)
    shap.summary_plot(
        shap_values,
        features=X_test,
        feature_names=X_test.columns,
        show=False
    )
    plt.savefig("models/autogluon/feature_importances.png")
    plt.close()
    live.log_image("feature_importances", "models/autogluon/feature_importances.png")

    live.end()

if __name__ == "__main__":
    main()