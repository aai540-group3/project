"""
Logistic Regression Stage Implementation
========================================

.. module:: pipeline.stages.logisticregression
   :synopsis: Pipeline stage for training and evaluating a Logistic Regression model

.. moduleauthor:: aai540-group3
"""

from loguru import logger

import pipeline.models as models

from .stage import Stage


class LogisticRegression(Stage):
    """Pipeline stage for Logistic Regression model training and evaluation."""

    def __init__(self):
        """Initialize the LogisticRegression pipeline stage."""
        super().__init__()

    def run(self):
        """Execute the Logistic Regression model stage.

        This method performs the following:
            1. Logs the start of the Logistic Regression stage.
            2. Executes the `LogisticRegression` model's `execute` method to train and evaluate the model.

        :raises Exception: If an error occurs during model execution.
        """
        try:
            logger.info("Starting LogisticRegression stage.")
            models.LogisticRegression().execute()

        except Exception as e:
            logger.error(f"Error during the LogisticRegression stage: {e}")
            raise
