#!/usr/bin/env python3
"""
pipeline.models.logistic_regression.model

Logistic Regression Model Training Script

This script trains a logistic regression model using configuration-specified hyperparameter optimization, data balancing,
and evaluation methods. Designed for easy customization via configuration dictionaries.

Attributes:
    logger (Logger): Configured logger for logging messages.
"""

import hashlib
import json
import logging
import os
import shutil
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage for sklearn

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from dvclive import Live
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_logistic_regression(CONFIG):
    """Trains a logistic regression model based on the provided configuration.

    This function loads data, applies SMOTE to handle class imbalance, optimizes hyperparameters using Optuna,
    and evaluates the model on the validation and test datasets. Artifacts, metrics, and plots are saved in
    directories specified by the CONFIG dictionary.

    Args:
        CONFIG (dict): Configuration dictionary specifying paths, model parameters, optimization settings,
                       and plotting options.

    Raises:
        Exception: If an error occurs during data processing, training, or artifact generation.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    np.random.seed(42)
    warnings.filterwarnings("ignore")

    # Set up artifact directories
    artifacts_path = CONFIG["paths"]["artifacts"]
    if artifacts_path.exists():
        shutil.rmtree(artifacts_path)
    for subdir in CONFIG["paths"]["subdirs"]:
        (artifacts_path / subdir).mkdir(parents=True, exist_ok=True)

    # Initialize DVC Live for tracking metrics
    live = Live(dir=str(artifacts_path / "metrics"), dvcyaml=False)

    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_path = CONFIG["paths"]["data"]
        df = pd.read_parquet(data_path)

        # Calculate data hash for reproducibility tracking
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()

        # Separate features and target
        columns_to_drop = ["event_timestamp", "created_timestamp"]
        X = df.drop(columns=[CONFIG["model"]["target"]] + columns_to_drop)
        y = df[CONFIG["model"]["target"]]

        # Log data info
        class_distribution = y.value_counts().to_dict()
        imbalance_ratio = max(class_distribution.values()) / min(class_distribution.values())
        live.log_params({
            "data_hash": data_hash,
            "n_samples": len(df),
            "n_features": len(X.columns),
            "class_distribution_before_smote": class_distribution,
            "imbalance_ratio_before_smote": float(imbalance_ratio),
        })

        # Train, validation, and test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=CONFIG["splits"]["test_size"],
            random_state=CONFIG["splits"]["random_state"], stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=CONFIG["splits"]["val_size"],
            random_state=CONFIG["splits"]["random_state"], stratify=y_train_val)

        # Apply SMOTE on training data to address imbalance
        logger.info("Applying SMOTE to training data...")
        smote = SMOTE(random_state=CONFIG["model"]["random_state"])
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter optimization using Optuna
        logger.info("Starting hyperparameter optimization...")

        def objective(trial):
            """Objective function for Optuna hyperparameter optimization."""
            # Define hyperparameters to tune
            C = trial.suggest_float("C", CONFIG["optimization"]["param_space"]["C"]["low"],
                                    CONFIG["optimization"]["param_space"]["C"]["high"],
                                    log=CONFIG["optimization"]["param_space"]["C"]["log"])
            penalty = trial.suggest_categorical("penalty", CONFIG["optimization"]["param_space"]["penalty"])
            solver = trial.suggest_categorical("solver", CONFIG["optimization"]["param_space"]["solver"])
            l1_ratio = trial.suggest_float("l1_ratio", CONFIG["optimization"]["param_space"]["l1_ratio"]["low"],
                                           CONFIG["optimization"]["param_space"]["l1_ratio"]["high"]) if penalty == "elasticnet" else None

            # Build model and evaluate on validation set
            model = LogisticRegression(C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio,
                                       max_iter=1000, random_state=CONFIG["model"]["random_state"])
            model.fit(X_train_scaled, y_train_balanced)
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
            return roc_auc_score(y_val, val_pred)

        # Run optimization study
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CONFIG["model"]["random_state"]))
        study.optimize(objective, n_trials=CONFIG["model"]["optimization_trials"], show_progress_bar=True)

        # Train final model with best hyperparameters
        best_params = study.best_trial.params
        logger.info("Training final model with best parameters...")
        final_model = LogisticRegression(
            C=best_params["C"],
            penalty=best_params["penalty"],
            solver=best_params["solver"],
            l1_ratio=best_params.get("l1_ratio"),
            max_iter=1000,
            random_state=CONFIG["model"]["random_state"]
        )
        final_model.fit(X_train_scaled, y_train_balanced)

        # Evaluate the model and generate metrics
        logger.info("Evaluating model...")
        val_pred = final_model.predict_proba(X_val_scaled)[:, 1]
        test_pred = final_model.predict_proba(X_test_scaled)[:, 1]
        val_pred_classes = (val_pred > 0.5).astype(int)
        test_pred_classes = (test_pred > 0.5).astype(int)

        # Calculate and log metrics
        metrics = {
            "val_accuracy": accuracy_score(y_val, val_pred_classes),
            "val_precision": precision_score(y_val, val_pred_classes),
            "val_recall": recall_score(y_val, val_pred_classes),
            "val_f1": f1_score(y_val, val_pred_classes),
            "val_auc": roc_auc_score(y_val, val_pred),
            "test_accuracy": accuracy_score(y_test, test_pred_classes),
            "test_precision": precision_score(y_test, test_pred_classes),
            "test_recall": recall_score(y_test, test_pred_classes),
            "test_f1": f1_score(y_test, test_pred_classes),
            "test_auc": roc_auc_score(y_test, test_pred),
        }
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)

        # Save final model and scaler
        joblib.dump(final_model, artifacts_path / "model" / "model.joblib")
        joblib.dump(scaler, artifacts_path / "model" / "scaler.joblib")

        # Generate and save plots
        logger.info("Generating visualizations...")

        # Confusion Matrix
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        cm = confusion_matrix(y_test, test_pred_classes)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig(artifacts_path / "plots" / "confusion_matrix.png", dpi=CONFIG["plots"]["figure"]["dpi"])
        plt.close()

        # ROC Curve
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
        plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (AUC = {metrics["test_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(artifacts_path / "plots" / "roc_curve.png", dpi=CONFIG["plots"]["figure"]["dpi"])
        plt.close()

        # Save metrics and hyperparameters
        with open(artifacts_path / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        with open(artifacts_path / "model" / "params.json", "w") as f:
            json.dump(best_params, f, indent=4)

        live.end()
        logger.info("Training pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        live.end()
        raise


def quick_run():
    """Runs an ultra-minimal configuration for instant feedback.
    Optimized for absolute minimum runtime - useful for code testing and debugging."""
    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/logistic_regression/artifacts"),
            "subdirs": ["model", "metrics", "plots"],
        },
        "model": {
            "target": "readmitted",
            "random_state": 42,
            "optimization_trials": 1,
            "cv_folds": 1,
        },
        "training": {
            "epochs": 1,
            "patience": 1,
            "batch_size": 512,
        },
        "splits": {
            "test_size": 0.2,
            "val_size": 0.25,
            "random_state": 42,
        },
        "optimization": {
            "param_space": {
                "C": {"low": 1.0, "high": 1.0, "log": True},
                "penalty": ["l2"],
                "solver": ["saga"],
                "l1_ratio": {"low": 0.5, "high": 0.5},
                "batch_size": [512],
                "learning_rate": {"low": 0.01, "high": 0.01, "log": True},
                "n_layers": {"low": 1, "high": 1},
                "units_first": {"low": 32, "high": 32, "step": 32},
                "units_factor": {"low": 0.5, "high": 0.5},
                "dropout": {"low": 0.1, "high": 0.1},
                "activation": ["relu"],
                "optimizer": ["adam"],
            }
        },
        "plots": {
            "style": "seaborn-v0_8-bright",
            "context": "paper",
            "font_scale": 1.2,
            "figure": {"figsize": (10, 6), "dpi": 300},
            "colors": {
                "primary": "#2196F3",
                "secondary": "#FF9800",
                "error": "#F44336",
                "success": "#4CAF50",
            },
        },
    }
    train_logistic_regression(CONFIG)


def full_run():
    """Runs a comprehensive training configuration optimized for model performance.
    Includes extensive hyperparameter search and validation."""
    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/logistic_regression/artifacts"),
            "subdirs": ["model", "metrics", "plots"],
        },
        "model": {
            "target": "readmitted",
            "random_state": 42,
            "optimization_trials": 100,
            "cv_folds": 5,
        },
        "training": {
            "epochs": 100,
            "patience": 10,
            "batch_size": 64,
        },
        "splits": {
            "test_size": 0.2,
            "val_size": 0.25,
            "random_state": 42,
        },
        "optimization": {
            "param_space": {
                "C": {"low": 1e-4, "high": 1e4, "log": True},
                "penalty": ["l1", "l2", "elasticnet", None],
                "solver": ["saga"],
                "l1_ratio": {"low": 0.0, "high": 1.0},
                "batch_size": [32, 64, 128, 256],
                "learning_rate": {"low": 1e-5, "high": 1e-2, "log": True},
                "n_layers": {"low": 1, "high": 5},
                "units_first": {"low": 32, "high": 256, "step": 32},
                "units_factor": {"low": 0.25, "high": 1.0},
                "dropout": {"low": 0.1, "high": 0.5},
                "activation": ["relu", "elu", "selu"],
                "optimizer": ["adam", "sgd"],
            }
        },
        "plots": {
            "style": "seaborn-v0_8-bright",
            "context": "paper",
            "font_scale": 1.2,
            "figure": {"figsize": (10, 6), "dpi": 300},
            "colors": {
                "primary": "#2196F3",
                "secondary": "#FF9800",
                "error": "#F44336",
                "success": "#4CAF50",
            },
        },
    }
    train_logistic_regression(CONFIG)


if __name__ == "__main__":
    MODE = os.getenv("MODE", "quick").lower()

    if MODE == "quick":
        quick_run()
    elif MODE == "full":
        full_run()
    else:
        print("Invalid mode. Please choose 'quick' or 'full'.")
