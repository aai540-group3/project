import hashlib
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor
from loguru import logger
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
from pipeline.stages.base import PipelineStage


class Autogluon(PipelineStage):
    """Pipeline stage for AutoGluon model training."""

    def run(self):
        """Execute AutoGluon training process."""

        warnings.filterwarnings("ignore")
        np.random.seed(self.cfg.seed)  # Use seed from configuration

        # Load configuration
        mode = os.getenv("TRAIN_MODE", "quick")
        config = self.cfg.autogluon[mode]
        model_params = config.get("model", {})
        label_column = model_params.get("label", "readmitted")

        # Load data
        data = self.load_data("features.parquet", subdir=str(self.cfg.paths.processed))
        X = data.drop(columns=[label_column])
        y = data[label_column]
        logger.info(f"Data shape: {X.shape}")

        # Calculate data hash for reproducibility
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()

        # Initialize DVC Live
        live = Live(dir=str(self.cfg.paths.metrics / "autogluon"), dvcyaml=False)

        try:
            # Split data based on configuration
            train_size = self.cfg.splits.train
            val_size = self.cfg.splits.val
            test_size = self.cfg.splits.test

            # First split into train and temp (val + test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X,
                y,
                test_size=(1 - train_size),
                random_state=self.cfg.seed,
                stratify=y,
            )

            # Then split temp into validation and test
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=(test_size / (val_size + test_size)),
                random_state=self.cfg.seed,
                stratify=y_temp,
            )

            # Log parameters
            live.log_params(
                {
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "test_size": len(X_test),
                    "features": len(X.columns),
                    "data_hash": data_hash,
                    "mode": mode,
                    **config.get("hyperparameters", {}),
                }
            )

            # Initialize predictor using configuration
            predictor = TabularPredictor(
                label=label_column,
                path=str(self.cfg.paths.models / "autogluon"),
                eval_metric=model_params.get("metric", "roc_auc"),
                problem_type=model_params.get("problem_type", "binary"),
            )

            # Prepare data for training
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)

            # Train model
            predictor.fit(
                train_data=train_data,
                tuning_data=val_data,
                time_limit=config.get("time_limit", None),
                hyperparameters=config.get("hyperparameters", {}),
                verbosity=2,
            )

            # Generate predictions
            y_pred = predictor.predict(X_test)
            y_pred_proba = predictor.predict_proba(X_test)[1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
                "pr_auc": average_precision_score(y_test, y_pred_proba),
            }

            # Log metrics
            for name, value in metrics.items():
                live.log_metric(name, value)
                logger.info(f"{name}: {value:.4f}")

            # Generate confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual No Readmission", "Actual Readmission"],
                columns=["Predicted No Readmission", "Predicted Readmission"],
            )

            self.save_plot(
                "confusion_matrix",
                lambda data: sns.heatmap(
                    data,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                ).set(title="Confusion Matrix"),
                data=cm_df,
            )

            # Generate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})

            self.save_plot(
                "roc_curve",
                lambda data: (
                    plt.plot(
                        data["FPR"],
                        data["TPR"],
                        label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})',
                        color=self.colors.primary,
                    ),
                    plt.plot([0, 1], [0, 1], "--", color=self.colors.neutral),
                    plt.xlabel("False Positive Rate"),
                    plt.ylabel("True Positive Rate"),
                    plt.title("ROC Curve"),
                    plt.legend(),
                ),
                data=roc_df,
            )

            # Generate feature importance plot
            importance_df = predictor.feature_importance(train_data)
            self.save_plot(
                "feature_importance",
                lambda data: sns.barplot(
                    data=data.head(20),
                    y="feature",
                    x="importance",
                    palette="viridis",
                ).set(title="Top 20 Features by Importance"),
                data=importance_df.reset_index(),
            )

            # Save feature importance using base class method
            self.save_output(
                importance_df.reset_index(),
                "feature_importance.csv",
                subdir=str(self.cfg.paths.metrics / self.name),
            )

            # Save model info
            model_info = {
                "best_model": predictor.get_model_best(),
                "leaderboard": predictor.leaderboard().to_dict(),
                "feature_importance": importance_df.to_dict(),
                "hyperparameters": config.get("hyperparameters", {}),
                "training_time": predictor.fit_time,
                "metrics": metrics,
            }

            self.save_metrics("model_info", model_info)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error("Full error:", exc_info=True)
            raise
        finally:
            live.end()
            logger.info("Training completed")
