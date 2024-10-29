#!/usr/bin/env python3
"""
pipeline.models.autogluon.model

This script trains machine learning models using AutoGluon's TabularPredictor. It includes functions
for both quick and comprehensive model training with various hyperparameters and model configurations.

Attributes:
    logger (Logger): Configured logger for logging messages.

Example:
    To train models in quick mode, run:

        $ MODE=quick python model.py

    To train models in full mode, run:

        $ MODE=full python model.py
"""

import hashlib
import json
import logging
import os
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from dvclive import Live
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(42)

# Set plot styles for all visualizations
plt.style.use("seaborn-v0_8-bright")
sns.set_theme()
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})


def train_autogluon(CONFIG):
    """Trains AutoGluon models based on the provided configuration.

    This function loads the dataset, configures AutoGluon, trains models,
    and saves model artifacts, metrics, and visualizations.

    Args:
        CONFIG (dict): Configuration dictionary specifying paths, model parameters,
                       hyperparameters, and plotting settings.

    Raises:
        Exception: If an unexpected error occurs during training, model evaluation, or file operations.
    """
    artifacts_path = CONFIG["paths"]["artifacts"]

    # Prepare artifact directories
    if os.path.exists(artifacts_path):
        shutil.rmtree(artifacts_path)
    for subdir in CONFIG["paths"]["subdirs"]:
        os.makedirs(os.path.join(artifacts_path, subdir), exist_ok=True)

    # Initialize DVC Live for tracking metrics
    live = Live(dir=os.path.join(artifacts_path, "metrics"), dvcyaml=False)

    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_parquet(CONFIG["paths"]["data"])
        logger.info(f"Data shape: {df.shape}")

        # Calculate data hash for tracking
        with open(CONFIG["paths"]["data"], "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()

        # Split features and target
        y = df[CONFIG["model"]["label"]]
        X = df.drop(columns=[CONFIG["model"]["label"]])

        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=CONFIG["training"]["splits"]["train_test"],
            random_state=CONFIG["training"]["splits"]["random_state"],
            stratify=y,
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=CONFIG["training"]["splits"]["val_test"],
            random_state=CONFIG["training"]["splits"]["random_state"],
            stratify=y_temp,
        )

        # Prepare data for AutoGluon
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Log data split information
        live.log_params({
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "n_features": X.shape[1],
            "data_hash": data_hash,
        })

        # Initialize and train the AutoGluon predictor
        logger.info("Training AutoGluon models...")
        model_path = os.path.join(artifacts_path, "model")
        predictor = TabularPredictor(
            label=CONFIG["model"]["label"],
            path=model_path,
            eval_metric=CONFIG["model"]["eval_metric"],
            problem_type=CONFIG["model"]["problem_type"],
        )

        # Use holdout bagging if specified, otherwise combine train and validation
        if CONFIG["training"]["use_bag_holdout"]:
            predictor.fit(
                train_data=train_data,
                tuning_data=val_data,
                time_limit=CONFIG["training"]["time_limit"],
                hyperparameters=CONFIG["hyperparameters"],
                presets=CONFIG["model"]["presets"],
                num_bag_folds=CONFIG["training"]["bag_folds"],
                num_stack_levels=CONFIG["training"]["stack_levels"],
                verbosity=2,
                **CONFIG["training"]["extra_params"],
            )
        else:
            combined_data = pd.concat([train_data, val_data], axis=0)
            predictor.fit(
                train_data=combined_data,
                time_limit=CONFIG["training"]["time_limit"],
                hyperparameters=CONFIG["hyperparameters"],
                presets=CONFIG["model"]["presets"],
                num_bag_folds=CONFIG["training"]["bag_folds"],
                num_stack_levels=CONFIG["training"]["stack_levels"],
                verbosity=2,
                **CONFIG["training"]["extra_params"],
            )

        # Save model information
        model_info = predictor.info()
        with open(os.path.join(artifacts_path, "model", "model_info.txt"), "w") as f:
            json.dump(model_info, f, indent=4)

        # Get best model predictions and calculate metrics
        test_pred = predictor.predict(test_data)
        test_pred_proba = predictor.predict_proba(test_data)
        test_pred_proba_class1 = (
            test_pred_proba[1]
            if isinstance(test_pred_proba, pd.DataFrame)
            else test_pred_proba
        )

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_precision": float(precision_score(y_test, test_pred)),
            "test_recall": float(recall_score(y_test, test_pred)),
            "test_f1": float(f1_score(y_test, test_pred)),
            "test_auc": float(roc_auc_score(y_test, test_pred_proba_class1)),
            "test_pr_auc": float(average_precision_score(y_test, test_pred_proba_class1)),
        }

        # Log metrics to DVC Live
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)
            logger.info(f"{metric_name}: {value:.4f}")

        # Save metrics
        with open(os.path.join(artifacts_path, "metrics", "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Generate feature importance plot
        try:
            feature_importance = predictor.feature_importance(test_data)
            feature_importance.to_csv(os.path.join(artifacts_path, "model", "feature_importance.csv"))

            plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
            importance_data = feature_importance["importance"].sort_values().tail(20)
            sns.barplot(x=importance_data.values, y=importance_data.index, palette="viridis")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(artifacts_path, "plots", "feature_importance.png"), dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {str(e)}")

        # Generate confusion matrix plot
        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_path, "plots", "confusion_matrix.png"), dpi=300)
        plt.close()

        # Generate ROC curve plot
        fpr, tpr, _ = roc_curve(y_test, test_pred_proba_class1)
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["test_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_path, "plots", "roc_curve.png"), dpi=300)
        plt.close()

        live.end()
        logger.info("Training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        live.end()
        raise


def quick_run():
    """Runs a minimal configuration of the training pipeline for testing and debugging."""
    CONFIG = {
        "training": {
            "time_limit": 10,
            "bag_folds": 0,
            "stack_levels": 0,
            "use_bag_holdout": False,
            "splits": {"train_test": 0.2, "val_test": 0.5, "random_state": 42},
            "extra_params": {"save_space": True},
        },
        "model": {
            "label": "readmitted",
            "eval_metric": "roc_auc",
            "problem_type": "binary",
            "presets": None,
        },
        "hyperparameters": {
            "GBM": [{"extra_trees": False, "learning_rate": 0.1, "num_boost_round": 10}],
        },
        "paths": {
            "artifacts": "models/autogluon/artifacts",
            "data": "data/interim/data_featured.parquet",
            "subdirs": ["model", "metrics", "plots"],
        },
        "plots": {
            "style": "seaborn-v0_8-darkgrid",
            "context": "paper",
            "figure": {"figsize": (10, 6), "dpi": 300},
        },
    }
    train_autogluon(CONFIG)


def full_run():
    """Runs a comprehensive configuration of the training pipeline with extensive hyperparameter tuning."""
    CONFIG = {
        "training": {
            "time_limit": 7200,
            "bag_folds": 5,
            "stack_levels": 2,
            "use_bag_holdout": True,
            "splits": {"train_test": 0.2, "val_test": 0.5, "random_state": 42},
            "extra_params": {
                "dynamic_stacking": True,
                "ds_args": {
                    "enable_ray_logging": True,
                },
                "num_gpus": -1,
                "feature_generator": "auto",
                "auto_stack": True,
                "save_space": False,
                "hyperparameter_tune_kwargs": {
                    "search_strategy": "auto",
                    "num_trials": 100,
                    "scheduler": "local",
                    "searcher": "random",
                },
            },
        },
        "model": {
            "label": "readmitted",
            "eval_metric": "roc_auc",
            "problem_type": "binary",
            "presets": "best_quality",
        },
        "hyperparameters": {
            "GBM": {
                "extra_trees": True,
                "ag_args": {"name_suffix": "ExtraTrees"},
                "learning_rate": 0.05,
                "num_boost_round": 500,
                "num_leaves": 128,
                "feature_fraction": 0.8,
                "min_data_in_leaf": 20,
            },
            "CAT": {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 8,
                "l2_leaf_reg": 3,
                "random_strength": 1,
                "min_data_in_leaf": 20,
            },
            "XGB": {
                "learning_rate": 0.05,
                "n_estimators": 500,
                "max_depth": 8,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
            },
            "RF": {
                "criterion": "gini",
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
            },
            "XT": {
                "n_estimators": 500,
                "max_depth": 15,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
            },
            "NN_TORCH": {
                "num_epochs": 100,
                "learning_rate": 0.001,
                "dropout_prob": 0.1,
                "weight_decay": 0.01,
                "batch_size": 512,
                "optimizer": "adam",
                "activation": "relu",
                "layers": [200, 100],
            },
            "FASTAI": {
                "layers": [200, 100],
                "bs": 512,
                "epochs": 50,
                "lr": 0.01,
            },
        },
        "paths": {
            "artifacts": "models/autogluon/artifacts",
            "data": "data/interim/data_featured.parquet",
            "subdirs": ["model", "metrics", "plots"],
        },
        "plots": {
            "style": "seaborn-v0_8-darkgrid",
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
    train_autogluon(CONFIG)

if __name__ == "__main__":
    MODE = os.getenv("MODE", "quick").lower()

    if MODE == "quick":
        quick_run()
    elif MODE == "full":
        full_run()
    else:
        print("Invalid mode. Please choose 'quick' or 'full'.")
