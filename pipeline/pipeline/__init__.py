"""
Diabetic Readmission Risk Prediction Pipeline
==========================================

A complete MLOps pipeline for predicting hospital readmission risk.

.. module:: pipeline
   :synopsis: MLOps pipeline for diabetic readmission prediction

.. moduleauthor:: aai540-group3
"""

from importlib.metadata import version

__version__ = version("pipeline")

from .models import AutoGluonModel, LogisticRegressionModel, NeuralNetworkModel
from .stages import (
    DataIngestionStage,
    PreprocessingStage,
    FeatureEngineeringStage,
    TrainingStage,
    EvaluationStage,
    DeploymentStage
)

__all__ = [
    'AutoGluonModel',
    'LogisticRegressionModel',
    'NeuralNetworkModel',
    'DataIngestionStage',
    'PreprocessingStage',
    'FeatureEngineeringStage',
    'TrainingStage',
    'EvaluationStage',
    'DeploymentStage',
]
