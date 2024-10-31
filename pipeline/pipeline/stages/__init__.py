"""
Pipeline Stages
===============

.. module:: pipeline.stages
   :synopsis: Modules defining individual pipeline stages

.. moduleauthor:: aai540-group3

This package contains modules for each stage of the pipeline, from data ingestion to model training and deployment.
"""

from .base import PipelineStage

__all__ = ["PipelineStage"]
