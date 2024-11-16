"""
Neural Network Stage
====================

.. module:: pipeline.stages.neural_network
   :synopsis: Pipeline stage for NeuralNetwork

.. moduleauthor:: aai540-group3
"""

from loguru import logger

import pipeline.models as models

from .stage import Stage


class NeuralNetwork(Stage):
    """Pipeline stage for NeuralNetwork model."""

    def __init__(self):
        """Initialize the NeuralNetwork pipeline stage."""
        super().__init__()

    def run(self):
        try:
            logger.info("Starting NeuralNetwork stage.")
            models.NeuralNetwork().execute()

        except Exception as e:
            logger.error(f"Error during the NeuralNetwork stage: {e}")
            raise
