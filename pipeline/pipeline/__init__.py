"""
MLOps Pipeline
===========

.. module:: pipeline
   :synopsis: MLOps pipeline for diabetic readmission prediction
"""

from importlib.metadata import version

__version__ = version("pipeline")

from .stages.base import PipelineStage

__all__ = [
    "PipelineStage",
]
