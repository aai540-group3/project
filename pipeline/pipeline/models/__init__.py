"""
Model Implementations
===================

This module provides implementations of various machine learning models
for diabetic readmission prediction.

Each model supports both quick validation and full training modes.
"""

import importlib

__all__ = [
    "Autogluon",
    "LogisticRegression",
    "Metrics",
    "Model",
    "NeuralNetwork",
]

_MODULES = {
    "Autogluon": "pipeline.models.autogluon",
    "LogisticRegression": "pipeline.models.logisticregression",
    "Metrics": "pipeline.models.metrics",
    "Model": "pipeline.models.model",
    "NeuralNetwork": "pipeline.models.neuralnetwork",
}


def __getattr__(name):
    if name in _MODULES:
        try:
            module = importlib.import_module(_MODULES[name])
            attribute = getattr(module, name)
            globals()[name] = attribute  # Cache the imported attribute
            return attribute
        except ImportError as e:
            raise ImportError(f"Cannot import name '{name}' from '{_MODULES[name]}'") from e
        except AttributeError as e:
            raise AttributeError(f"Module '{_MODULES[name]}' does not have attribute '{name}'") from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
