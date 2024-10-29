#!/usr/bin/env python3
"""
pipeline.models.neural_network.model
Neural Network Model Training Script

This script trains a neural network model with hyperparameter tuning, data balancing,
and evaluation using configurations specified in a JSON or dictionary format.

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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for sklearn

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import tensorflow as tf
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from imblearn.over_sampling import SMOTE
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model


def train_neural_network(CONFIG):
    """Trains a neural network model based on the provided configuration.

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
    tf.random.set_seed(42)
    warnings.filterwarnings("ignore")

    artifacts_path = CONFIG["paths"]["artifacts"]
    if artifacts_path.exists():
        shutil.rmtree(artifacts_path)
    for subdir in CONFIG["paths"]["subdirs"]:
        (artifacts_path / subdir).mkdir(parents=True, exist_ok=True)

    live = Live(dir=str(artifacts_path / "metrics"), dvcyaml=False)

    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data_path = CONFIG["paths"]["data"]
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        df = pd.read_parquet(data_path)

        columns_to_drop = ["event_timestamp", "created_timestamp"]
        X = df.drop(columns=[CONFIG["model"]["target"]] + columns_to_drop)
        y = df[CONFIG["model"]["target"]]

        class_distribution = y.value_counts().to_dict()
        imbalance_ratio = max(class_distribution.values()) / min(class_distribution.values())
        live.log_params({
            "data_hash": data_hash,
            "n_samples": len(df),
            "n_features": len(X.columns),
            "class_distribution_before_smote": class_distribution,
            "imbalance_ratio_before_smote": float(imbalance_ratio),
        })

        # Data split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=CONFIG["splits"]["test_size"],
            random_state=CONFIG["splits"]["random_state"], stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=CONFIG["splits"]["val_size"],
            random_state=CONFIG["splits"]["random_state"], stratify=y_train_val)

        # Apply SMOTE
        logger.info("Applying SMOTE to training data...")
        smote = SMOTE(random_state=CONFIG["model"]["random_state"])
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        y_train = y_train_balanced

        live.log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_size_after_smote": len(X_train_balanced),
        })

        # Hyperparameter Optimization
        logger.info("Starting hyperparameter optimization...")

        def create_model(trial, input_dim):
            """Builds a neural network model based on trial suggestions for hyperparameters.

            Args:
                trial (optuna.Trial): Optuna trial object containing suggestions for hyperparameters.
                input_dim (int): Number of input features.

            Returns:
                model (Sequential): Compiled Keras model.
                batch_size (int): Suggested batch size for training.
            """
            # Suggest hyperparameters
            n_layers = trial.suggest_int("n_layers", CONFIG["optimization"]["param_space"]["n_layers"]["low"],
                                         CONFIG["optimization"]["param_space"]["n_layers"]["high"])
            units_first = trial.suggest_int("units_first", CONFIG["optimization"]["param_space"]["units_first"]["low"],
                                            CONFIG["optimization"]["param_space"]["units_first"]["high"],
                                            CONFIG["optimization"]["param_space"]["units_first"]["step"])
            units_factor = trial.suggest_float("units_factor", CONFIG["optimization"]["param_space"]["units_factor"]["low"],
                                               CONFIG["optimization"]["param_space"]["units_factor"]["high"])
            dropout = trial.suggest_float("dropout", CONFIG["optimization"]["param_space"]["dropout"]["low"],
                                          CONFIG["optimization"]["param_space"]["dropout"]["high"])
            learning_rate = trial.suggest_float("learning_rate", CONFIG["optimization"]["param_space"]["learning_rate"]["low"],
                                                CONFIG["optimization"]["param_space"]["learning_rate"]["high"],
                                                log=CONFIG["optimization"]["param_space"]["learning_rate"]["log"])
            batch_size = trial.suggest_categorical("batch_size", CONFIG["optimization"]["param_space"]["batch_size"])
            activation = trial.suggest_categorical("activation", CONFIG["optimization"]["param_space"]["activation"])
            optimizer_name = trial.suggest_categorical("optimizer", CONFIG["optimization"]["param_space"]["optimizer"])

            model = Sequential()
            model.add(Dense(units_first, activation=activation, input_shape=(input_dim,)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            units = units_first
            for _ in range(n_layers - 1):
                units = max(int(units * units_factor), 1)
                model.add(Dense(units, activation=activation))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            model.add(Dense(1, activation="sigmoid"))

            optimizer = Adam(learning_rate=learning_rate) if optimizer_name == "adam" else SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

            return model, batch_size

        def objective(trial):
            """Objective function for Optuna hyperparameter optimization."""
            model, batch_size = create_model(trial, input_dim=X_train_scaled.shape[1])
            early_stopping = EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True)
            history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                                epochs=CONFIG["training"]["epochs"], batch_size=batch_size, callbacks=[early_stopping], verbose=0)
            return max(history.history["val_auc"])

        # Run optimization study
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=CONFIG["model"]["random_state"]),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2, interval_steps=1))
        study.optimize(objective, n_trials=CONFIG["model"]["optimization_trials"], show_progress_bar=True)

        best_params = study.best_trial.params
        for param_name, param_value in best_params.items():
            live.log_param(f"best_{param_name}", str(param_value))

        # Train final model
        logger.info("Training final model with best parameters...")

        final_model, batch_size = create_model(trial=optuna.trial.FixedTrial(best_params), input_dim=X_train_scaled.shape[1])

        plot_model(final_model, to_file=str(artifacts_path / "plots" / "model_architecture.png"), show_shapes=True, show_layer_names=True)

        with open(artifacts_path / "model" / "model_summary.txt", "w") as f:
            final_model.summary(print_fn=lambda x: f.write(x + "\n"))

        early_stopping = EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True)
        dvc_callback = DVCLiveCallback(live=live, model_file=str(artifacts_path / "model" / "model.keras"))

        final_history = final_model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                                        epochs=CONFIG["training"]["epochs"], batch_size=batch_size,
                                        callbacks=[early_stopping, dvc_callback], verbose=1)

        # Evaluation
        logger.info("Evaluating model...")

        val_pred = final_model.predict(X_val_scaled).flatten()
        test_pred = final_model.predict(X_test_scaled).flatten()
        val_pred_classes = (val_pred > 0.5).astype(int)
        test_pred_classes = (test_pred > 0.5).astype(int)

        metrics = {
            "val_accuracy": float(accuracy_score(y_val, val_pred_classes)),
            "val_precision": float(precision_score(y_val, val_pred_classes)),
            "val_recall": float(recall_score(y_val, val_pred_classes)),
            "val_f1": float(f1_score(y_val, val_pred_classes)),
            "val_auc": float(roc_auc_score(y_val, val_pred)),
            "val_avg_precision": float(average_precision_score(y_val, val_pred)),
            "test_accuracy": float(accuracy_score(y_test, test_pred_classes)),
            "test_precision": float(precision_score(y_test, test_pred_classes)),
            "test_recall": float(recall_score(y_test, test_pred_classes)),
            "test_f1": float(f1_score(y_test, test_pred_classes)),
            "test_auc": float(roc_auc_score(y_test, test_pred)),
            "test_avg_precision": float(average_precision_score(y_test, test_pred)),
        }

        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)

        # Plotting and artifact saving
        logger.info("Saving artifacts and plots...")

        final_model.save(artifacts_path / "model" / "model.keras")
        joblib.dump(scaler, artifacts_path / "model" / "scaler.joblib")
        with open(artifacts_path / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        with open(artifacts_path / "model" / "params.json", "w") as f:
            json.dump(best_params, f, indent=4)
        with open(artifacts_path / "metrics" / "history.json", "w") as f:
            json.dump(final_history.history, f, indent=4)

        live.end()
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        live.end()
        raise




def quick_run():
    """Runs the neural network pipeline with quick configuration."""
    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/neural_network/artifacts"),
            "subdirs": ["model", "metrics", "plots"],
        },
        "model": {
            "target": "readmitted",
            "random_state": 42,
            "optimization_trials": 1,
            "cv_folds": 1,
        },
        "training": {"epochs": 1},
        "splits": {
            "test_size": 0.2,
            "val_size": 0.25,
            "random_state": 42,
        },
        "optimization": {
            "param_space": {
                "n_layers": {"low": 2, "high": 2},
                "units_first": {"low": 32, "high": 64, "step": 16},
                "units_factor": {"low": 0.7, "high": 0.8},
                "dropout": {"low": 0.2, "high": 0.3},
                "learning_rate": {
                    "low": 1e-3,
                    "high": 1e-2,
                    "log": True,
                },
                "batch_size": [32],
                "activation": ["relu"],
                "optimizer": ["adam"],
            }
        },
        "plots": {
            "style": "seaborn-v0_8-bright",
            "context": "paper",
            "font_scale": 1.2,
            "figure": {
                "figsize": (10, 6),
                "dpi": 300,
            },
            "colors": {
                "primary": "#2196F3",
                "secondary": "#FF9800",
                "error": "#F44336",
                "success": "#4CAF50",
            },
        },
    }
    train_neural_network(CONFIG)


def full_run():
    """Runs the neural network pipeline with the full configuration."""
    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/neural_network/artifacts"),
            "subdirs": ["model", "metrics", "plots"],
        },
        "model": {
            "target": "readmitted",
            "random_state": 42,
            "optimization_trials": 50,
            "cv_folds": 3,
        },
        "training": {"epochs": 100},
        "splits": {
            "test_size": 0.2,
            "val_size": 0.25,
            "random_state": 42,
        },
        "optimization": {
            "param_space": {
                "n_layers": {"low": 2, "high": 4},
                "units_first": {"low": 32, "high": 256, "step": 32},
                "units_factor": {"low": 0.5, "high": 0.8},
                "dropout": {"low": 0.1, "high": 0.5},
                "learning_rate": {"low": 1e-4, "high": 1e-2, "log": True},
                "batch_size": [32, 64, 128, 256],
                "activation": ["relu", "selu"],
                "optimizer": ["adam", "sgd"],
            }
        },
        "plots": {
            "style": "seaborn-v0_8-bright",
            "context": "paper",
            "font_scale": 1.2,
            "figure": {
                "figsize": (10, 6),
                "dpi": 300,
            },
            "colors": {
                "primary": "#2196F3",
                "secondary": "#FF9800",
                "error": "#F44336",
                "success": "#4CAF50",
            },
        },
    }
    train_neural_network(CONFIG)


if __name__ == "__main__":
    MODE = os.getenv("MODE", "quick").lower()

    if MODE == "quick":
        quick_run()
    elif MODE == "full":
        full_run()
    else:
        print("Invalid mode. Please choose 'quick' or 'full'.")
