import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from dvclive import Live
from dvclive.keras import DVCLiveCallback
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function implementing the complete neural network pipeline."""

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
        "splits": {
            "test_size": 0.2,
            "val_size": 0.25,  # Adjusted to ensure correct split sizes
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
            "style": "seaborn-darkgrid",
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
        "shap": {
            "sample_size": 100,
            "max_display": 20,
        },
    }

    # Clean up and recreate artifact directories
    if CONFIG["paths"]["artifacts"].exists():
        shutil.rmtree(CONFIG["paths"]["artifacts"])
    for subdir in CONFIG["paths"]["subdirs"]:
        (CONFIG["paths"]["artifacts"] / subdir).mkdir(parents=True, exist_ok=True)

    # Initialize DVC Live
    live = Live(dir=str(CONFIG["paths"]["artifacts"] / "metrics"), dvcyaml=False)

    try:
        # Step 1: Data Loading and Preprocessing
        logger.info("Loading and preparing data...")
        data_path = CONFIG["paths"]["data"]
        artifacts_path = CONFIG["paths"]["artifacts"]

        # Calculate data hash
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()

        # Load and prepare data
        df = pd.read_parquet(data_path)
        X = df.drop(columns=[CONFIG["model"]["target"]])
        y = df[CONFIG["model"]["target"]]

        # Log data info
        class_distribution = y.value_counts().to_dict()
        class_distribution_str = {str(k): int(v) for k, v in class_distribution.items()}
        imbalance_ratio = max(class_distribution.values()) / min(
            class_distribution.values()
        )
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

        live.log_params(
            {
                "data_hash": data_hash,
                "n_samples": len(df),
                "n_features": len(X.columns),
                "class_distribution": class_distribution_str,
                "imbalance_ratio": float(imbalance_ratio),
            }
        )

        # Create data splits
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=CONFIG["splits"]["test_size"],
            random_state=CONFIG["splits"]["random_state"],
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=CONFIG["splits"]["val_size"],
            random_state=CONFIG["splits"]["random_state"],
            stratify=y_train_val,
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Log split sizes
        live.log_params(
            {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            }
        )

        # Step 2: Hyperparameter Optimization
        logger.info("Starting hyperparameter optimization...")

        def create_model(trial, input_dim):
            # Define hyperparameters to optimize
            n_layers = trial.suggest_int(
                "n_layers",
                CONFIG["optimization"]["param_space"]["n_layers"]["low"],
                CONFIG["optimization"]["param_space"]["n_layers"]["high"],
            )
            units_first = trial.suggest_int(
                "units_first",
                CONFIG["optimization"]["param_space"]["units_first"]["low"],
                CONFIG["optimization"]["param_space"]["units_first"]["high"],
                CONFIG["optimization"]["param_space"]["units_first"]["step"],
            )
            units_factor = trial.suggest_float(
                "units_factor",
                CONFIG["optimization"]["param_space"]["units_factor"]["low"],
                CONFIG["optimization"]["param_space"]["units_factor"]["high"],
            )
            dropout = trial.suggest_float(
                "dropout",
                CONFIG["optimization"]["param_space"]["dropout"]["low"],
                CONFIG["optimization"]["param_space"]["dropout"]["high"],
            )
            learning_rate = trial.suggest_float(
                "learning_rate",
                CONFIG["optimization"]["param_space"]["learning_rate"]["low"],
                CONFIG["optimization"]["param_space"]["learning_rate"]["high"],
                log=CONFIG["optimization"]["param_space"]["learning_rate"]["log"],
            )
            batch_size = trial.suggest_categorical(
                "batch_size", CONFIG["optimization"]["param_space"]["batch_size"]
            )
            activation = trial.suggest_categorical(
                "activation", CONFIG["optimization"]["param_space"]["activation"]
            )
            optimizer_name = trial.suggest_categorical(
                "optimizer", CONFIG["optimization"]["param_space"]["optimizer"]
            )

            # Build model
            model = Sequential()
            model.add(Dense(units_first, activation=activation, input_dim=input_dim))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

            units = units_first
            for _ in range(n_layers - 1):
                units = max(int(units * units_factor), 1)
                model.add(Dense(units, activation=activation))
                model.add(BatchNormalization())
                model.add(Dropout(dropout))

            model.add(Dense(1, activation="sigmoid"))

            # Choose optimizer
            if optimizer_name == "adam":
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == "sgd":
                optimizer = SGD(learning_rate=learning_rate)
            else:
                optimizer = Adam(learning_rate=learning_rate)

            # Compile model
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
            )

            return model, batch_size

        def objective(trial):
            # Create model
            model, batch_size = create_model(trial, input_dim=X_train_scaled.shape[1])

            # Early stopping
            early_stopping = EarlyStopping(
                monitor="val_auc", patience=5, mode="max", restore_best_weights=True
            )

            # Train model
            history = model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=50,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0,
            )

            # Get best validation AUC
            val_auc = max(history.history["val_auc"])

            return val_auc

        # Create and run study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=CONFIG["model"]["random_state"]),
        )

        study.optimize(
            objective,
            n_trials=CONFIG["model"]["optimization_trials"],
            show_progress_bar=True,
        )

        # Log best parameters
        best_params = study.best_trial.params
        for param_name, param_value in best_params.items():
            live.log_param(f"best_{param_name}", str(param_value))

        # Step 3: Train Final Model
        logger.info("Training final model with best parameters...")

        # Build final model
        final_model, batch_size = create_model(
            trial=optuna.trial.FixedTrial(best_params),
            input_dim=X_train_scaled.shape[1],
        )

        # Save model architecture visualization
        plot_model(
            final_model,
            to_file=str(
                CONFIG["paths"]["artifacts"] / "plots" / "model_architecture.png"
            ),
            show_shapes=True,
            show_layer_names=True,
        )

        # Save text representation of model architecture
        with open(
            CONFIG["paths"]["artifacts"] / "model" / "model_summary.txt", "w"
        ) as f:
            final_model.summary(print_fn=lambda x: f.write(x + "\n"))

        # Train final model
        early_stopping = EarlyStopping(
            monitor="val_auc", patience=10, mode="max", restore_best_weights=True
        )

        dvc_callback = DVCLiveCallback(
            live=live,
            model_file=str(CONFIG["paths"]["artifacts"] / "model" / "model.h5"),
        )

        final_history = final_model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stopping, dvc_callback],
            verbose=1,
        )

        # Step 4: Evaluate Model
        logger.info("Evaluating model...")

        # Get predictions
        val_pred = final_model.predict(X_val_scaled).flatten()
        test_pred = final_model.predict(X_test_scaled).flatten()

        val_pred_classes = (val_pred > 0.5).astype(int)
        test_pred_classes = (test_pred > 0.5).astype(int)

        # Calculate metrics
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

        # Log metrics
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)

        # Set plotting style
        plt.style.use(CONFIG["plots"]["style"])
        sns.set_context(
            CONFIG["plots"]["context"], font_scale=CONFIG["plots"]["font_scale"]
        )

        # Generate plots
        logger.info("Generating visualization plots...")

        # 1. Confusion Matrix
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        cm = confusion_matrix(y_test, test_pred_classes)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Readmission", "Readmission"],
            yticklabels=["No Readmission", "Readmission"],
        )
        plt.title("Confusion Matrix", pad=20)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "confusion_matrix.png",
            dpi=CONFIG["plots"]["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

        # 2. ROC Curve
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])

        # Plot validation ROC
        fpr_val, tpr_val, _ = roc_curve(y_val, val_pred)
        plt.plot(
            fpr_val,
            tpr_val,
            color=CONFIG["plots"]["colors"]["primary"],
            label=f'Validation (AUC = {metrics["val_auc"]:.3f})',
        )

        # Plot test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
        plt.plot(
            fpr_test,
            tpr_test,
            color=CONFIG["plots"]["colors"]["secondary"],
            label=f'Test (AUC = {metrics["test_auc"]:.3f})',
        )

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve", pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "roc_curve.png",
            dpi=CONFIG["plots"]["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

        # Step 5: SHAP Analysis
        logger.info("Generating SHAP explanations...")

        try:
            # Use DeepExplainer for neural networks
            sample_size = min(CONFIG["shap"]["sample_size"], X_test_scaled.shape[0])
            X_shap = X_test_scaled[:sample_size]

            explainer = shap.DeepExplainer(final_model, X_train_scaled[:sample_size])
            shap_values = explainer.shap_values(X_shap)

            # SHAP Summary Plot
            plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
            shap.summary_plot(
                shap_values[0],
                X_shap,
                feature_names=X.columns.tolist(),
                max_display=CONFIG["shap"]["max_display"],
                show=False,
            )
            plt.title("SHAP Summary Plot", pad=20)
            plt.tight_layout()
            plt.savefig(
                CONFIG["paths"]["artifacts"] / "plots" / "shap_summary.png",
                dpi=CONFIG["plots"]["figure"]["dpi"],
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            logger.warning(f"SHAP analysis skipped: {e}")

        # Step 6: Save Artifacts
        logger.info("Saving artifacts...")

        # Save model and components
        final_model.save(CONFIG["paths"]["artifacts"] / "model" / "model.h5")
        joblib.dump(scaler, CONFIG["paths"]["artifacts"] / "model" / "scaler.joblib")

        # Save metrics
        with open(CONFIG["paths"]["artifacts"] / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save parameters
        with open(CONFIG["paths"]["artifacts"] / "model" / "params.json", "w") as f:
            json.dump(best_params, f, indent=4)

        # Save training history
        with open(CONFIG["paths"]["artifacts"] / "metrics" / "history.json", "w") as f:
            json.dump(final_history.history, f, indent=4)

        live.end()
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        live.end()
        raise


if __name__ == "__main__":
    main()
