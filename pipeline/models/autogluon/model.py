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
plt.style.use("seaborn-v0_8-bright")
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


def train_autogluon(CONFIG):
    """Trains AutoGluon models based on provided configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    artifacts_path = CONFIG["paths"]["artifacts"]
    # Clean up and recreate artifact directories
    if os.path.exists(artifacts_path):
        shutil.rmtree(artifacts_path)
    for subdir in CONFIG["paths"]["subdirs"]:
        os.makedirs(os.path.join(artifacts_path, subdir), exist_ok=True)

    # Initialize DVC Live
    live = Live(dir=os.path.join(artifacts_path, "metrics"), dvcyaml=False)

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
        model_path = os.path.join(artifacts_path, "model")
        predictor = TabularPredictor(
            label=CONFIG["model"]["label"],
            path=model_path,
            eval_metric=CONFIG["model"]["eval_metric"],
            problem_type=CONFIG["model"]["problem_type"],
        )

        if CONFIG["training"]["use_bag_holdout"]:
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
                use_bag_holdout=CONFIG["training"]["use_bag_holdout"],
                verbosity=2,
                **CONFIG["training"]["extra_params"],
            )

        # Save model info with proper serialization
        model_info = predictor.info()
        serializable_info = {
            "type_of_learner": str(model_info.get("type_of_learner", "")),
            "model_types": str(model_info.get("model_types", [])),
            "model_performance": str(model_info.get("model_performance", {})),
            "model_best": str(model_info.get("model_best", "")),
            "model_paths": str(model_info.get("model_paths", {})),
            "model_fit_times": str(model_info.get("model_fit_times", {})),
            "model_pred_times": str(model_info.get("model_pred_times", {})),
            "num_bag_folds": str(model_info.get("num_bag_folds", "")),
            "max_stack_level": str(model_info.get("max_stack_level", "")),
        }

        with open(
            os.path.join(artifacts_path, "model", "model_info.txt"), "w"
        ) as f:
            json.dump(serializable_info, f, indent=4)

        # Generate Ensemble Model Visualization
        try:
            predictor.plot_ensemble_model(filename="best_model_architecture.png")
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
        with open(
            os.path.join(artifacts_path, "metrics", "metrics.json"), "w"
        ) as f:
            json.dump(metrics, f, indent=4)

        # Feature importance
        try:
            feature_importance = predictor.feature_importance(test_data)
            feature_importance.to_csv(
                os.path.join(artifacts_path, "model", "feature_importance.csv")
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
            )[-20:]  # Top 20 features
            sns.barplot(
                x=importance_data.values, y=importance_data.index, palette="viridis"
            )
            plt.title("Feature Importance", pad=20)
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.savefig(
                os.path.join(artifacts_path, "plots", "feature_importance.png"),
                bbox_inches="tight",
                dpi=CONFIG["plots"]["figure"]["dpi"],
            )
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate feature importance: {str(e)}")

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
            os.path.join(artifacts_path, "plots", "confusion_matrix.png"),
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
            os.path.join(artifacts_path, "plots", "roc_curve.png"),
            bbox_inches="tight",
            dpi=CONFIG["plots"]["figure"]["dpi"],
        )
        plt.close()

        # Save leaderboard
        predictor.leaderboard(test_data, silent=True).to_csv(
            os.path.join(artifacts_path, "model", "leaderboard.csv"), index=False
        )

        live.end()
        logger.info(f"Pipeline completed successfully! Best model: {best_model_name}")
        logger.info(f"Model artifacts saved to: {artifacts_path}")
        logger.info(f"Test AUC-ROC: {metrics['test_auc']:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Full error:", exc_info=True)
        live.end()
        raise


def quick_run():
    """Runs an ultra-minimal AutoGluon configuration for instant feedback.
    Perfect for code testing, debugging, and development iterations."""
    CONFIG = {
        "training": {
            "time_limit": 10,
            "bag_folds": 0,
            "stack_levels": 0,
            "use_bag_holdout": False,
            "splits": {"train_test": 0.2, "val_test": 0.5, "random_state": 42},
            "extra_params": {
                "dynamic_stacking": False,
                "ds_args": {
                    "enable_ray_logging": False
                },
                "num_gpus": 0,
                "feature_generator": None,
                "auto_stack": False,
                "save_space": True,
            },
        },
        "model": {
            "label": "readmitted",
            "eval_metric": "roc_auc",
            "problem_type": "binary",
            "presets": None,
        },
        "hyperparameters": {
            "GBM": [
                {
                    "extra_trees": False,
                    "ag_args": {"name_suffix": "Basic"},
                    "learning_rate": 0.1,
                    "num_boost_round": 10,
                    "num_leaves": 4,
                    "deterministic": True,
                    "early_stopping_rounds": 3,
                },
            ],
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


def full_run():
    """Runs a comprehensive AutoGluon configuration optimized for model performance.
    Includes extensive hyperparameter search and model stacking."""
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
            "GBM": [
                {
                    "extra_trees": True,
                    "ag_args": {"name_suffix": "ExtraTrees"},
                    "learning_rate": 0.05,
                    "num_boost_round": 500,
                    "num_leaves": 128,
                    "feature_fraction": 0.8,
                    "min_data_in_leaf": 20,
                },
                {
                    "learning_rate": 0.01,
                    "num_boost_round": 1000,
                    "num_leaves": 64,
                    "feature_fraction": 0.9,
                    "min_data_in_leaf": 10,
                },
            ],
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
            "RF": [
                {
                    "criterion": "gini",
                    "max_depth": 15,
                    "min_samples_split": 10,
                    "min_samples_leaf": 5,
                },
                {
                    "criterion": "entropy",
                    "max_depth": 15,
                    "min_samples_split": 15,
                    "min_samples_leaf": 8,
                },
            ],
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
