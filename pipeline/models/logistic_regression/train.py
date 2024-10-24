# models/logistic_regression/train.py
import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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
import shap # Import SHAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train and evaluate Logistic Regression model."""
    logger.info("Training Logistic Regression model...")

    # Calculate hash of training data
    with open("data/processed/train.csv", "rb") as f:
        input_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()
    live = Live(dir="models/logistic_regression", dvcyaml=False)

    # Check if model exists and hash matches
    skip_training = False
    if os.path.exists("models/logistic_regression/model.pkl") and os.path.exists("models/logistic_regression/input_hash.txt"):
        with open("models/logistic_regression/input_hash.txt", "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == input_hash:
            logger.info("Logistic Regression model already exists. Skipping training.")
            model = joblib.load("models/logistic_regression/model.pkl")
            skip_training = True

    if not skip_training:
        # Train Logistic Regression
        logger.info("Training Logistic Regression model...")
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

        model = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
        )

        # Define parameter grid for GridSearchCV
        param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, scoring="roc_auc", cv=5
        )
        grid_search.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        model = grid_search.best_estimator_

        # Save the model
        joblib.dump(model, "models/logistic_regression/model.pkl")
        logger.info(f"Logistic Regression model saved to models/logistic_regression/model.pkl")
        # Save the hash
        with open("models/logistic_regression/input_hash.txt", "w") as f:
            f.write(input_hash)

    # Evaluate Logistic Regression
    logger.info("Evaluating Logistic Regression model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Assuming binary classification

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    with open("models/logistic_regression/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    for metric_name, metric_value in metrics.items():
        live.log_metric(metric_name, metric_value)

    # Log confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(model.classes_))
    plt.xticks(tick_marks, model.classes_)
    plt.yticks(tick_marks, model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/logistic_regression/confusion_matrix.png")
    plt.close()
    live.log_image("confusion_matrix", "models/logistic_regression/confusion_matrix.png")

    # Log ROC curve
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
    plt.savefig("models/logistic_regression/roc_curve.png")
    plt.close()
    live.log_image("roc_curve", "models/logistic_regression/roc_curve.png")

    # Compute and plot feature importances using SHAP
    logger.info("Calculating feature importances using SHAP...")
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)

    # Summary plot (bar plot of mean absolute SHAP values)
    shap.summary_plot(
        shap_values,
        features=X_test,
        feature_names=X_test.columns,
        show=False
    )
    plt.savefig("models/logistic_regression/feature_importances.png")
    plt.close()
    live.log_image("feature_importances", "models/logistic_regression/feature_importances.png")

    live.end()

if __name__ == "__main__":
    main()