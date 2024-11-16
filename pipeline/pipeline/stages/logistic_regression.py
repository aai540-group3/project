"""
Logistic Regression Stage Implementation
==============================

.. module:: pipeline.stages.logistic_regression
   :synopsis: Pipeline stage for LogisticRegression

.. moduleauthor:: aai540-group3
"""

from loguru import logger

import pipeline.models as models

from .stage import Stage


class LogisticRegression(Stage):
    """Pipeline stage for LogisticRegression model."""

    def __init__(self):
        """Initialize the LogisticRegression pipeline stage."""
        super().__init__()

    def run(self):
        try:
            logger.info("Starting LogisticRegression stage.")
            models.LogisticRegression().execute()

        except Exception as e:
            logger.error(f"Error during the LogisticRegression stage: {e}")
            raise
