"""
Pipeline Package
================

This package serves as the entry point for all pipeline functionalities, including initializing
pipeline stages and setting up the package environment.

.. module:: pipeline
   :synopsis: Main package for pipeline stages and versioning.

.. moduleauthor:: aai540-group3
"""

from importlib.metadata import version

__version__ = version("pipeline")

import pipeline

__all__ = [
    "__version__",
    "pipeline",
]
