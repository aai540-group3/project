import hashlib
import json
import logging
import os
import shutil
import warnings
from pathlib import Path

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
    """Trains a logistic regression model based on provided configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Set seeds for reproducibility
    np.random.seed(42)

    # Ignore warnings
    warnings.filterwarnings("ignore")

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16

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

        # Calculate data hash
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        df = pd.read_parquet(data_path)

        # Drop unnecessary columns
        columns_to_drop = ["event_timestamp", "created_timestamp"]
        X = df.drop(columns=[CONFIG["model"]["target"]] + columns_to_drop)
        y = df[CONFIG["model"]["target"]]

        # Verify data types before proceeding
        logger.info("Data types after preprocessing:")
        for col in X.columns:
            logger.info(f"{col}: {X[col].dtype}")

        # Log data info
        class_distribution = y.value_counts().to_dict()
        class_distribution_str = {str(k): int(v) for k, v in class_distribution.items()}
        imbalance_ratio = max(class_distribution.values()) / min(
            class_distribution.values()
        )
        logger.info(f"Class imbalance ratio before SMOTE: {imbalance_ratio:.2f}")

        live.log_params(
            {
                "data_hash": data_hash,
                "n_samples": len(df),
                "n_features": len(X.columns),
                "class_distribution_before_smote": class_distribution_str,
                "imbalance_ratio_before_smote": float(imbalance_ratio),
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

        # Apply SMOTE to training data only
        logger.info("Applying SMOTE to training data...")
        smote = SMOTE(random_state=CONFIG["model"]["random_state"])
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Update class distribution after SMOTE
        class_distribution_after_smote = (
            pd.Series(y_train_balanced).value_counts().to_dict()
        )
        class_distribution_after_smote_str = {
            str(k): int(v) for k, v in class_distribution_after_smote.items()
        }
        imbalance_ratio_after_smote = max(
            class_distribution_after_smote.values()
        ) / min(class_distribution_after_smote.values())
        logger.info(
            f"Class imbalance ratio after SMOTE: {imbalance_ratio_after_smote:.2f}"
        )

        live.log_params(
            {
                "class_distribution_after_smote": class_distribution_after_smote_str,
                "imbalance_ratio_after_smote": float(imbalance_ratio_after_smote),
            }
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Update y_train
        y_train = y_train_balanced

        # Log split sizes
        live.log_params(
            {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "train_size_after_smote": len(X_train_balanced),
            }
        )

        # Step 2: Hyperparameter Optimization
        logger.info("Starting hyperparameter optimization...")

        def objective(trial):
            # Define hyperparameters to optimize
            C = trial.suggest_float(
                "C",
                CONFIG["optimization"]["param_space"]["C"]["low"],
                CONFIG["optimization"]["param_space"]["C"]["high"],
                log=CONFIG["optimization"]["param_space"]["C"]["log"],
            )
            penalty = trial.suggest_categorical(
                "penalty",
                CONFIG["optimization"]["param_space"]["penalty"],
            )
            solver = trial.suggest_categorical(
                "solver",
                CONFIG["optimization"]["param_space"]["solver"],
            )
            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float(
                    "l1_ratio",
                    CONFIG["optimization"]["param_space"]["l1_ratio"]["low"],
                    CONFIG["optimization"]["param_space"]["l1_ratio"]["high"],
                )
            else:
                l1_ratio = None

            # Handle penalty=None correctly
            penalty_param = penalty  # This can be None or a string

            # Create model
            model = LogisticRegression(
                C=C,
                penalty=penalty_param,
                solver=solver,
                l1_ratio=l1_ratio,
                max_iter=1000,
                random_state=CONFIG["model"]["random_state"],
            )

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate on validation set
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)

            return val_auc

        # Create and run study
        study = optuna.create_study(
            study_name="logistic_regression_hyperparameter_tuning",
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
            # Convert None to string 'None' for logging
            live.log_param(f"best_{param_name}", str(param_value))

        # Step 3: Train Final Model
        logger.info("Training final model with best parameters...")

        # Handle l1_ratio if needed
        if best_params.get("penalty") == "elasticnet":
            l1_ratio = best_params.get("l1_ratio", None)
        else:
            l1_ratio = None

        # Handle penalty=None correctly
        penalty_param = best_params.get("penalty")
        if penalty_param == "None":
            penalty_param = None

        # Build final model
        final_model = LogisticRegression(
            C=best_params["C"],
            penalty=penalty_param,
            solver=best_params["solver"],
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=CONFIG["model"]["random_state"],
        )

        # Train final model
        final_model.fit(X_train_scaled, y_train)

        # Step 4: Evaluate Model
        logger.info("Evaluating model...")

        # Get predictions
        val_pred = final_model.predict_proba(X_val_scaled)[:, 1]
        test_pred = final_model.predict_proba(X_test_scaled)[:, 1]

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
            label=f"Validation (AUC = {metrics['val_auc']:.3f})",
        )

        # Plot test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, test_pred)
        plt.plot(
            fpr_test,
            tpr_test,
            color=CONFIG["plots"]["colors"]["secondary"],
            label=f"Test (AUC = {metrics['test_auc']:.3f})",
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

        # Step 5: Save Artifacts
        logger.info("Saving artifacts...")

        # Save model and scaler
        joblib.dump(
            final_model, CONFIG["paths"]["artifacts"] / "model" / "model.joblib"
        )
        joblib.dump(scaler, CONFIG["paths"]["artifacts"] / "model" / "scaler.joblib")

        # Save metrics
        with open(CONFIG["paths"]["artifacts"] / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save parameters
        # Convert None to string 'None' for JSON serialization
        best_params_serializable = {
            k: (str(v) if v is None else v) for k, v in best_params.items()
        }
        with open(CONFIG["paths"]["artifacts"] / "model" / "params.json", "w") as f:
            json.dump(best_params_serializable, f, indent=4)

        live.end()
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        live.end()
        raise


def quick_run():
    """Runs the logistic regression pipeline with quick configuration."""
    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/logistic_regression/artifacts"),
            "subdirs": ["model", "metrics", "plots"],
        },
        "model": {
            "target": "readmitted",
            "random_state": 42,
            "optimization_trials": 5,
            "cv_folds": 2,
        },
        "splits": {
            "test_size": 0.2,
            "val_size": 0.25,
            "random_state": 42,
        },
        "optimization": {
            "param_space": {
                "C": {"low": 1e-1, "high": 1e1, "log": True},
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
                "l1_ratio": {"low": 0.0, "high": 1.0},
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
    train_logistic_regression(CONFIG)


def full_run():
    """Runs the logistic regression pipeline with the full configuration."""
    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/logistic_regression/artifacts"),
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
            "val_size": 0.25,
            "random_state": 42,
        },
        "optimization": {
            "param_space": {
                "C": {"low": 1e-4, "high": 1e2, "log": True},
                "penalty": ["l1", "l2", "elasticnet", None],
                "solver": ["saga"],
                "l1_ratio": {"low": 0.0, "high": 1.0},
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
    train_logistic_regression(CONFIG)


if __name__ == "__main__":
    MODE = "quick"
    if MODE.lower() == "quick":
        quick_run()
    elif MODE.lower() == "full":
        full_run()
    else:
        print("Invalid mode. Please choose 'quick' or 'full'.")
