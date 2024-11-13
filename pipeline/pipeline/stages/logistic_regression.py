"""
Logistic Regression Stage
=========================

.. module:: pipeline.stages.logistic
   :synopsis: Pipeline stage for Logistic Regression model training with DVC Live tracking

.. moduleauthor:: aai540-group3
"""

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from pipeline.models.logistic_regression import LogisticRegressionModel
from pipeline.models.metrics import Metrics
from pipeline.stages.stage import Stage


class LogisticRegressionStage(Stage):
    """Pipeline stage for Logistic Regression model training with DVC Live tracking."""

    def __init__(self, cfg: DictConfig):
        """Initialize the Logistic Regression pipeline stage with base configurations."""
        super().__init__(cfg)
        self.model: Optional[LogisticRegressionModel] = None
        self.metrics: Optional[Metrics] = None
        self.data: Optional[pd.DataFrame] = None
        self.label_column: str = self.cfg.get("label_column", "readmitted")

    def run(self):
        """Entry point for running model training."""
        self.setup()
        model_path, metrics = self.model.train()
        self.evaluate_and_log(metrics)
        self.create_plots(model_path)
        self.save_outputs(model_path, metrics)

    def setup(self):
        """Set up the model and necessary configurations."""
        logger.info("Setting up the LogisticRegressionStage.")
        self.model = LogisticRegressionModel(cfg=self.cfg)

        # Load data if not already loaded
        if self.data is None:
            self.data = self.load_data("features.parquet", subdir=self.cfg.paths.processed)
            X = self.data.drop(columns=[self.label_column])
            y = self.data[self.label_column]
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data(X, y)
            data_hash = hashlib.md5(
                pd.util.hash_pandas_object(pd.concat([X, y], axis=1), index=True).values
            ).hexdigest()
            self.log_params(self.cfg, data_hash, self.X_train, self.X_val, self.X_test)

        # Update model's configuration with training data
        self.model.cfg.data = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
        }

    def evaluate_and_log(self, metrics: Metrics):
        """Log metrics and perform evaluations."""
        self.metrics = metrics
        self.log_metrics(metrics.to_dict())
        # Optionally, log to DVC Live or other tracking systems
        if self.live:
            self.live.log_metrics(metrics.to_dict())

    def create_plots(self, model_path: Path):
        """Create and save evaluation plots."""
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test).iloc[:, 1]
            # Initialize Metrics for plotting
            plotting_metrics = Metrics(y_true=self.y_test.tolist(), y_pred=y_pred.tolist(), y_proba=y_proba.tolist())
            # Define save paths
            cm_save_path = Path(self.cfg.paths.plots) / self.name / "confusion_matrix.png"
            roc_save_path = Path(self.cfg.paths.plots) / self.name / "roc_curve.png"
            pr_save_path = Path(self.cfg.paths.plots) / self.name / "precision_recall_curve.png"
            fi_save_path = Path(self.cfg.paths.plots) / self.name / "feature_importance.png"

            # Plot and save confusion matrix
            plotting_metrics.plot_confusion_matrix(
                y_true=self.y_test.tolist(), y_pred=y_pred.tolist(), save_path=cm_save_path, title="Confusion Matrix"
            )

            # Plot and save ROC curve
            plotting_metrics.plot_roc_curve(
                y_true=self.y_test.tolist(), y_proba=y_proba.tolist(), save_path=roc_save_path, title="ROC Curve"
            )

            # Plot and save Precision-Recall curve
            plotting_metrics.plot_precision_recall_curve(
                y_true=self.y_test.tolist(),
                y_proba=y_proba.tolist(),
                save_path=pr_save_path,
                title="Precision-Recall Curve",
            )

            # Plot and save feature importance
            feature_importance = self.calculate_feature_importance()
            plotting_metrics.plot_feature_importance(
                feature_importance=feature_importance, save_path=fi_save_path, title="Top 20 Features by Importance"
            )

        except Exception as e:
            logger.error(f"Failed to generate evaluation plots: {e}")

    def calculate_feature_importance(self) -> pd.DataFrame:
        """Calculate feature importance based on model coefficients."""
        try:
            if not self.model.predictor:
                raise ValueError("Model has not been trained or loaded.")

            # Assuming the model is a Pipeline with 'classifier' as the final step
            classifier = self.model.predictor
            if not isinstance(classifier, LogisticRegressionModel):
                raise TypeError("Classifier is not an instance of LogisticRegression.")

            # Extract feature names from preprocessor if available
            if hasattr(self.model, "preprocessor") and self.model.preprocessor:
                preprocessor = self.model.preprocessor
                numeric_features = preprocessor.transformers_[0][2]
                categorical_transformers = preprocessor.transformers_[1][1]
                if isinstance(categorical_transformers, Stage):
                    ohe = categorical_transformers.named_steps.get("onehot", None)
                    if ohe:
                        ohe_features = ohe.get_feature_names_out(categorical_transformers.transformers_[0][2])
                    else:
                        ohe_features = []
                else:
                    ohe_features = []
                all_features = list(numeric_features) + list(ohe_features)
            else:
                all_features = [f"feature_{i}" for i in range(len(classifier.coef_[0]))]

            coefficients = classifier.coef_[0]
            feature_importance = pd.DataFrame({"feature": all_features, "importance": coefficients})
            feature_importance["abs_importance"] = feature_importance["importance"].abs()
            feature_importance = feature_importance.sort_values(by="abs_importance", ascending=False).drop(
                "abs_importance", axis=1
            )

            logger.debug(f"Feature Importance DataFrame:\n{feature_importance.head(20)}")
            return feature_importance.head(20)

        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return pd.DataFrame()

    def save_outputs(self, model_path: Path, metrics: Metrics):
        """Save model outputs and metrics."""
        self.save_metrics("model_metrics", metrics.to_dict())
        # Additional outputs like feature importance can be saved here
