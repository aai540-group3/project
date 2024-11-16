"""
Pipeline Stages
==============

This module provides the implementation of various stages in the ML pipeline
for diabetic readmission prediction.

.. module:: pipeline.stages
   :synopsis: ML pipeline stages for data processing and model training

.. moduleauthor:: aai540-group3
"""

import importlib

__all__ = [
    "autogluon",
    "explore",
    "feast",
    "featurize",
    "infrastruct",
    "ingest",
    "logisticregression",
    "preprocess",
    "stage",
]

_MODULES = {
    "autogluon": "pipeline.stages.autogluon",
    "explore": "pipeline.stages.explore",
    "feast": "pipeline.stages.feast",
    "featurize": "pipeline.stages.featurize",
    "infrastruct": "pipeline.stages.infrastruct",
    "ingest": "pipeline.stages.ingest",
    "logisticregression": "pipeline.stages.logisticregression",
    "preprocess": "pipeline.stages.preprocess",
    "stage": "pipeline.stages.stage",
}


def __getattr__(name):
    """Lazily import module attributes.

    :param name: Name of attribute to import
    :type name: str
    :return: Imported module attribute
    :raises: AttributeError if attribute not found
    """
    if name in _MODULES:
        try:
            module = importlib.import_module(_MODULES[name])
            attribute = getattr(module, name)
            globals()[name] = attribute
            return attribute
        except ImportError as e:
            raise ImportError(f"Cannot import name '{name}' from '{_MODULES[name]}'") from e
        except AttributeError as e:
            raise AttributeError(f"Module '{_MODULES[name]}' does not have attribute '{name}'") from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
