"""
AutoGluon Stage Implementation
==============================

.. module:: pipeline.stages.autogluon
   :synopsis: Pipeline stage for AutoGluon model training with DVC Live tracking

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

import pipeline.models as models

from .stage import Stage


class Autogluon(Stage):
    """Pipeline stage for AutoGluon model training with DVC Live tracking."""

    def __init__(self):
        """Initialize the AutoGluon pipeline stage with base configurations."""
        super().__init__()
        self.model: Optional[models.Autogluon] = models.Autogluon()
        self.metrics: Optional[models.Metrics] = None

    def run(self):
        """Execute the full training, evaluation, and logging pipeline."""
        try:
            logger.info("Starting AutoGluon stage.")

            # Ensure data is prepared
            if not self.model.X_train or not self.model.y_train:
                logger.info("Preparing data for training.")
                self.model.prepare_data()

            # Train the model
            model_path, metrics = self.model.train()
            self.metrics = metrics
            logger.info("Training complete.")

            # Log metrics
            logger.info("Logging metrics.")
            self.model.metrics = models.Metrics.from_dict(metrics.to_dict())
            self.model.save_metrics("autogluon_metrics", metrics.to_dict())
            if self.model.live:
                self.model.live.log_metrics(metrics.to_dict())
                logger.info("Metrics logged to DVC Live.")

            # Generate predictions and plots for the test set
            y_pred = self.model.predict(self.model.X_test)
            y_proba = self.model.predict_proba(self.model.X_test).iloc[:, 1]

            # Initialize Metrics for plotting
            plotting_metrics = models.Metrics(
                y_true=self.model.y_test.tolist(),
                y_pred=y_pred.tolist(),
                y_proba=y_proba.tolist(),
            )

            # Define paths for saving plots
            cm_save_path = Path(self.model.plots_dir) / "confusion_matrix.png"
            roc_save_path = Path(self.model.plots_dir) / "roc_curve.png"
            fi_save_path = Path(self.model.plots_dir) / "feature_importance.png"

            # Plot and save confusion matrix
            plotting_metrics.plot_confusion_matrix(
                y_true=self.model.y_test.tolist(),
                y_pred=y_pred.tolist(),
                save_path=cm_save_path,
                title=self.model.cfg.plots.confusion_matrix.title,
            )
            logger.info(f"Confusion matrix saved at {cm_save_path}")

            # Plot and save ROC curve
            plotting_metrics.plot_roc_curve(
                y_true=self.model.y_test.tolist(),
                y_proba=y_proba.tolist(),
                save_path=roc_save_path,
                title=self.model.cfg.plots.roc_curve.title,
            )
            logger.info(f"ROC curve saved at {roc_save_path}")

            # Plot and save feature importance
            feature_importance = self.model.predictor.feature_importance(
                data=pd.concat([self.model.X_test, self.model.y_test], axis=1)
            )
            feature_importance = feature_importance.reset_index().rename(
                columns={"index": "feature", "importance": "importance"}
            )
            plotting_metrics.plot_feature_importance(
                feature_importance=feature_importance,
                save_path=fi_save_path,
                title=self.model.cfg.plots.feature_importance.title,
            )
            logger.info(f"Feature importance plot saved at {fi_save_path}")

            # Save final metrics to disk
            self.save_metrics("model_metrics", metrics.to_dict())
            logger.info("AutoGluon stage completed successfully.")

        except Exception as e:
            logger.error(f"Error during the AutoGluon stage: {e}")
            raise
