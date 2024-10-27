import hashlib
import json
import logging
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
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

from dvclive import Live

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
np.random.seed(42)

# Set the style and font settings
plt.style.use("seaborn-v0_8")
sns.set_theme()

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16


def main():
    """Main function implementing the AutoGluon pipeline."""

    CONFIG = {
        "training": {
            "time_limit": 7200,  # 2 hours
            "bag_folds": 5,
            "stack_levels": 1,
            "use_bag_holdout": True,
            "splits": {"train_test": 0.2, "val_test": 0.5, "random_state": 42},
        },
        "model": {
            "label": "readmitted",
            "eval_metric": "roc_auc",
            "problem_type": "binary",
            "presets": "best_quality",
        },
        "hyperparameters": {
            "GBM": [
                {
                    "extra_trees": True,
                    "ag_args": {"name_suffix": "ExtraTrees"},
                    "learning_rate": 0.05,
                    "num_boost_round": 500,
                    "num_leaves": 128,
                },
                {
                    "learning_rate": 0.01,
                    "num_boost_round": 1000,
                    "num_leaves": 64,
                },
            ],
            "CAT": {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 8,
            },
            "XGB": {
                "learning_rate": 0.05,
                "n_estimators": 500,
                "max_depth": 8,
            },
            "RF": [
                {"criterion": "gini", "max_depth": 15},
                {"criterion": "entropy", "max_depth": 15},
            ],
            "XT": {"n_estimators": 500, "max_depth": 15},
            "NN_TORCH": {
                "num_epochs": 100,
                "learning_rate": 0.001,
                "dropout_prob": 0.1,
            },
        },
        "paths": {
            "artifacts": Path("models/autogluon/artifacts"),
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
        y = df[CONFIG["model"]["label"]]
        X = df.drop(columns=[CONFIG["model"]["label"]])

        # Split data
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

        # Log data splits info
        live.log_params(
            {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
                "n_features": X.shape[1],
                "data_hash": data_hash,
            }
        )

        # Initialize and train predictor
        logger.info("Training AutoGluon models...")
        predictor = TabularPredictor(
            label=CONFIG["model"]["label"],
            path=str(CONFIG["paths"]["artifacts"] / "model"),
            eval_metric=CONFIG["model"]["eval_metric"],
            problem_type=CONFIG["model"]["problem_type"],
        )

        predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            time_limit=CONFIG["training"]["time_limit"],
            hyperparameters=CONFIG["hyperparameters"],
            presets=CONFIG["model"]["presets"],
            num_bag_folds=CONFIG["training"]["bag_folds"],
            num_stack_levels=CONFIG["training"]["stack_levels"],
            use_bag_holdout=CONFIG["training"]["use_bag_holdout"],
            verbosity=2,
        )

        # Generate model_info.txt
        model_info = predictor.info()
        with open(CONFIG["paths"]["artifacts"] / "model" / "model_info.txt", "w") as f:
            f.write(json.dumps(model_info, indent=4))

        # Generate Ensemble Model Visualization
        try:
            ensemble_plot_path = predictor.plot_ensemble_model(
                filename=CONFIG["paths"]["artifacts"]
                / "plots"
                / "best_model_architecture.png"
            )
            logger.info(f"Saved ensemble model visualization to: {ensemble_plot_path}")
        except ImportError:
            logger.warning(
                "Could not generate ensemble model visualization. "
                "Ensure graphviz and pygraphviz are installed."
            )

        # Get predictions
        best_model_name = predictor.get_model_best()
        logger.info(f"Best Model: {best_model_name}")
        test_pred = predictor.predict(test_data)
        test_pred_proba = predictor.predict_proba(test_data)
        test_pred_proba_class1 = (
            test_pred_proba[1]
            if isinstance(test_pred_proba, pd.DataFrame)
            else test_pred_proba
        )

        # Calculate metrics
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_precision": float(precision_score(y_test, test_pred)),
            "test_recall": float(recall_score(y_test, test_pred)),
            "test_f1": float(f1_score(y_test, test_pred)),
            "test_auc": float(roc_auc_score(y_test, test_pred_proba_class1)),
            "test_pr_auc": float(
                average_precision_score(y_test, test_pred_proba_class1)
            ),
        }

        # Log metrics
        for metric_name, value in metrics.items():
            live.log_metric(metric_name, value)
            logger.info(f"{metric_name}: {value:.4f}")

        # Save metrics
        with open(CONFIG["paths"]["artifacts"] / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Feature importance
        feature_importance = predictor.feature_importance(test_data)
        feature_importance.to_csv(
            CONFIG["paths"]["artifacts"] / "model" / "feature_importance.csv"
        )
        feature_importance_dict = feature_importance["importance"].to_dict()

        # Generate plots
        plt.style.use(CONFIG["plots"]["style"])
        sns.set_context(
            CONFIG["plots"]["context"], font_scale=CONFIG["plots"]["font_scale"]
        )

        # Feature Importance Plot
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        importance_data = pd.Series(feature_importance_dict).sort_values(
            ascending=True
        )[-20:]
        sns.barplot(
            x=importance_data.values, y=importance_data.index, palette="viridis"
        )
        plt.title("Feature Importance", pad=20)
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "feature_importance.png",
            bbox_inches="tight",
            dpi=CONFIG["plots"]["figure"]["dpi"],
        )
        plt.close()

        # Confusion Matrix
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        cm = confusion_matrix(y_test, test_pred)
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
            bbox_inches="tight",
            dpi=CONFIG["plots"]["figure"]["dpi"],
        )
        plt.close()

        # ROC Curve
        plt.figure(figsize=CONFIG["plots"]["figure"]["figsize"])
        fpr, tpr, _ = roc_curve(y_test, test_pred_proba_class1)
        plt.plot(
            fpr,
            tpr,
            color=CONFIG["plots"]["colors"]["primary"],
            label=f'ROC curve (AUC = {metrics["test_auc"]:.3f})',
        )
        plt.plot(
            [0, 1], [0, 1], color=CONFIG["plots"]["colors"]["secondary"], linestyle="--"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic", pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            CONFIG["paths"]["artifacts"] / "plots" / "roc_curve.png",
            bbox_inches="tight",
            dpi=CONFIG["plots"]["figure"]["dpi"],
        )
        plt.close()

        # Save leaderboard
        predictor.leaderboard(test_data, silent=True).to_csv(
            CONFIG["paths"]["artifacts"] / "model" / "leaderboard.csv", index=False
        )

        live.end()
        logger.info(f"Pipeline completed successfully! Best model: {best_model_name}")
        logger.info(f"Model artifacts saved to: {CONFIG['paths']['artifacts']}")
        logger.info(f"Test AUC-ROC: {metrics['test_auc']:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Full error:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
