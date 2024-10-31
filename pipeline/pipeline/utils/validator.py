"""
Data and Model Validation
=====================

.. module:: pipeline.utils.validation
   :synopsis: Validation utilities for data and models

.. moduleauthor:: aai540-group3
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import cross_val_score


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_schema(data: pd.DataFrame, schema: Dict[str, str]) -> List[str]:
        """Validate DataFrame schema.

        :param data: Input DataFrame
        :type data: pd.DataFrame
        :param schema: Expected schema
        :type schema: Dict[str, str]
        :return: List of validation errors
        :rtype: List[str]
        """
        errors = []

        # Check columns
        missing_cols = set(schema.keys()) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # Check types
        for col, expected_type in schema.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if actual_type != expected_type:
                    errors.append(f"Column {col} has type {actual_type}, " f"expected {expected_type}")

        return errors

    @staticmethod
    def validate_values(data: pd.DataFrame, constraints: Dict[str, Dict]) -> List[str]:
        """Validate value constraints.

        :param data: Input DataFrame
        :type data: pd.DataFrame
        :param constraints: Value constraints
        :type constraints: Dict[str, Dict]
        :return: List of validation errors
        :rtype: List[str]
        """
        errors = []

        for col, rules in constraints.items():
            if col not in data.columns:
                continue

            # Check range
            if "range" in rules:
                min_val, max_val = rules["range"]
                if data[col].min() < min_val or data[col].max() > max_val:
                    errors.append(f"Column {col} values outside range " f"[{min_val}, {max_val}]")

            # Check categories
            if "categories" in rules:
                invalid = set(data[col].unique()) - set(rules["categories"])
                if invalid:
                    errors.append(f"Column {col} contains invalid categories: {invalid}")

            # Check missing values
            if "allow_missing" in rules:
                if not rules["allow_missing"] and data[col].isnull().any():
                    errors.append(f"Column {col} contains missing values")

        return errors


class ModelValidator:
    """Model validation utilities."""

    @staticmethod
    def validate_predictions(
        y_pred: np.ndarray,
        expected_shape: Optional[tuple] = None,
        value_range: Optional[tuple] = None,
    ) -> List[str]:
        """Validate model predictions.

        :param y_pred: Model predictions
        :type y_pred: np.ndarray
        :param expected_shape: Expected shape
        :type expected_shape: Optional[tuple]
        :param value_range: Expected value range
        :type value_range: Optional[tuple]
        :return: List of validation errors
        :rtype: List[str]
        """
        errors = []

        # Check shape
        if expected_shape and y_pred.shape != expected_shape:
            errors.append(f"Prediction shape {y_pred.shape} != " f"expected shape {expected_shape}")

        # Check range
        if value_range:
            min_val, max_val = value_range
            if y_pred.min() < min_val or y_pred.max() > max_val:
                errors.append(f"Predictions outside range [{min_val}, {max_val}]")

        return errors

    @staticmethod
    def validate_performance(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metric: str,
        threshold: float,
        cv: int = 5,
    ) -> List[str]:
        """Validate model performance.

        :param model: Model to validate
        :type model: Any
        :param X: Features
        :type X: np.ndarray
        :param y: Labels
        :type y: np.ndarray
        :param metric: Performance metric
        :type metric: str
        :param threshold: Performance threshold
        :type threshold: float
        :param cv: Cross-validation folds
        :type cv: int
        :return: List of validation errors
        :rtype: List[str]
        """
        errors = []

        # Cross-validation
        scores = cross_val_score(model, X, y, scoring=metric, cv=cv)
        mean_score = scores.mean()

        if mean_score < threshold:
            errors.append(f"Cross-validation {metric} ({mean_score:.3f}) " f"below threshold ({threshold})")

        return errors
