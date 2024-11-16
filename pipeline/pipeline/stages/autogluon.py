"""
AutoGluon Stage Implementation
==============================

.. module:: pipeline.stages.autogluon
   :synopsis: Pipeline stage for AutoGluon model training with DVC Live tracking

.. moduleauthor:: aai540-group3
"""

from loguru import logger

import pipeline.models as models

from .stage import Stage


class Autogluon(Stage):
    """Pipeline stage for AutoGluon model training with DVC Live tracking."""

    def __init__(self):
        """Initialize the AutoGluon pipeline stage with base configurations."""
        super().__init__()

    def run(self):
        """Execute the full training, evaluation, and logging pipeline."""
        try:
            logger.info("Starting AutoGluon stage.")
            models.Autogluon().execute()

        except Exception as e:
            logger.error(f"Error during the AutoGluon stage: {e}")
            raise
