"""
Neural Network Stage
====================

.. module:: pipeline.stages.neuralnetwork
   :synopsis: Pipeline stage for training and evaluating a Neural Network model

.. moduleauthor:: aai540-group3
"""

from loguru import logger

from .models.neuralnetwork import NeuralNetwork

from .stage import Stage


class NeuralNetwork(Stage):
    """Pipeline stage for Neural Network model training and evaluation."""

    def __init__(self):
        """Initialize the Neural Network pipeline stage."""
        super().__init__()

    def run(self):
        """Execute the Neural Network model stage.

        This method performs the following:
            1. Logs the start of the Neural Network stage.
            2. Executes the `NeuralNetwork` model's `execute` method to train and evaluate the model.

        :raises Exception: If an error occurs during model execution.
        """
        try:
            logger.info("Starting NeuralNetwork stage.")
            NeuralNetwork().execute()

        except Exception as e:
            logger.error(f"Error during the Neural Network stage: {e}")
            raise
