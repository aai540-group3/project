import hashlib
import json
import logging
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

warnings.filterwarnings("ignore")
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function implementing the logistic regression pipeline."""

    CONFIG = {
        "paths": {
            "data": "data/interim/data_featured.parquet",
            "artifacts": Path("models/logistic_regression/artifacts"),
            "subdirs": ["model", "metrics", "plots"],
        },
        "model": {
            "target": "readmitted",
            "params": {
                "penalty": "l1",
                "solver": "liblinear",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
            "optimization_trials": 50,
        },
        "splits": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "plots": {
            "style": "seaborn-darkgrid",
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

    # Clean up and recreate artifact directories
    if CONFIG["paths"]["artifacts"].exists():
        shutil.rmtree(CONFIG["paths"]["artifacts"])
    for subdir in CONFIG["paths"]["subdirs"]:
        (CONFIG["paths"]["artifacts"] / subdir).mkdir(parents=True, exist_ok=True)

    # Initialize DVC Live
    live = Live(dir=str(CONFIG["paths"]["artifacts"] / "metrics"), dvcyaml=False)

    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_parquet(CONFIG["paths"]["data"])
        logger.info(f"Data shape: {df.shape}")

        # Calculate data hash
        with open(CONFIG["paths"]["data"], "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()

        # Split features and target
        X = df.drop(columns=[CONFIG["model"]["target"]])
        y = df[CONFIG["model"]["target"]]
        feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=CONFIG["splits"]["test_size"],
            random_state=CONFIG["splits"]["random_state"],
            stratify=y,
        )

        # Log data splits info
        live.log_params(
            {
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": X.shape[1],
                "data_hash": data_hash,
            }
        )

        # Define the Optuna objective function
        def objective(trial):
            # Suggest hyperparameters
            C = trial.suggest_float("C", 1e-5, 1e5, log=True)  # Regularization strength

            # Create model with suggested hyperparameters
            model = LogisticRegression(
                **CONFIG["model"]["params"], C=C
            )
            model.fit(X_train, y_train)

            # Evaluate model on validation set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            return auc

        # Create Optuna study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=CONFIG["model"]["optimization_trials"])

        # Get best hyperparameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters found by Optuna: {best_params}")

        # Train final model with best hyperparameters
        logger.info("Training final logistic regression model with best params...")
        model = LogisticRegression(**CONFIG["model"]["params"], **best_params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_pred_proba)),
            "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
        }

        # Log metrics
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)
            logger.info(f"{metric_name}: {value:.4f}")

        # Generate plots
        plt.style.use(CONFIG["plots"]["style"])
        sns.set_context(
            CONFIG["plots"]["context"], font_scale=CONFIG["plots"]["font_scale"]
        )

        # Feature Importance Plot
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(model.coef_[0])}
        ).sort_values("importance", ascending=True)

        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        sns.barplot(
            x="importance",
            y="feature",
            data=feature_importance.tail(20),
            palette="viridis",
        )
        plt.title("Feature Importance (Absolute Coefficients)")
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "feature_importance.png",
            dpi=CONFIG["plots"]["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

        # Confusion Matrix
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Readmission", "Readmission"],
            yticklabels=["No Readmission", "Readmission"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "confusion_matrix.png",
            dpi=CONFIG["plots"]["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

        # ROC Curve
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(
            fpr,
            tpr,
            color=CONFIG["plots"]["colors"]["primary"],
            label=f'ROC curve (AUC = {metrics["auc"]:.3f})',
        )
        plt.plot(
            [0, 1], [0, 1], color=CONFIG["plots"]["colors"]["secondary"], linestyle="--"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "roc_curve.png",
            dpi=CONFIG["plots"]["figure"]["dpi"],
            bbox_inches="tight",
        )
        plt.close()

        # Save artifacts
        logger.info("Saving artifacts...")

        # Save model
        joblib.dump(model, CONFIG["paths"]["artifacts"] / "model" / "model.joblib")

        # Save metrics
        with open(CONFIG["paths"]["artifacts"] / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save feature importance
        feature_importance.to_csv(
            CONFIG["paths"]["artifacts"] / "metrics" / "feature_importance.csv",
            index=False,
        )

        live.end()
        logger.info("Pipeline completed successfully!")
        logger.info(f"Model artifacts saved to: {CONFIG['paths']['artifacts']}")
        logger.info(f"Test AUC-ROC: {metrics['auc']:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Full error:", exc_info=True)
        raise


if __name__ == "__main__":
    main()