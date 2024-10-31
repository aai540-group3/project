"""
Model Implementations
===================

This module provides implementations of various machine learning models
for diabetic readmission prediction.

Available Models
--------------
- Logistic Regression
- Neural Network
- AutoGluon

Each model supports both quick validation and full training modes.
"""

from .autogluon import AutoGluonModel
from .base import BaseModel
from .logistic import LogisticRegressionModel
from .neural import NeuralNetworkModel

__all__ = [
    "AutoGluonModel",
    "LogisticRegressionModel",
    "NeuralNetworkModel",
    "BaseModel",
]
