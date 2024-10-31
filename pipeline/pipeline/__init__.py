"""
MLOps Pipeline
===========

.. module:: pipeline
   :synopsis: MLOps pipeline for diabetic readmission prediction
"""

from importlib.metadata import version

__version__ = version("pipeline")

# Import only what's needed for infrastructure stage
from .stages.base import PipelineStage
from .stages.infrastruct import InfrastructStage

__all__ = [
    "PipelineStage",
    "InfrastructStage",
]
