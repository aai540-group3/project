"""
Data Cleaning Utilities
===================

.. module:: pipeline.utils.data_cleaning
   :synopsis: Data cleaning and preprocessing utilities

.. moduleauthor:: aai540-group3
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .logging import get_logger


class DataCleaner:
    """Data cleaning and preprocessing utility."""

    def __init__(self, cfg: DictConfig):
        """Initialize data cleaner.

        :param cfg: Cleaning configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.imputers: Dict[str, SimpleImputer] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize cleaning components."""
        # Initialize imputers
        if self.cfg.missing_values.strategy == "impute":
            self.imputers["numeric"] = SimpleImputer(strategy=self.cfg.missing_values.numeric.method)
            self.imputers["categorical"] = SimpleImputer(strategy=self.cfg.missing_values.categorical.method)

    def clean_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Clean and preprocess data.

        :param data: Input data
        :type data: pd.DataFrame
        :param fit: Whether to fit transformers
        :type fit: bool
        :return: Cleaned data
        :rtype: pd.DataFrame
        """
        df = data.copy()

        # Handle missing values
        df = self._handle_missing_values(df, fit)

        # Handle invalid values
        df = self._handle_invalid_values(df)

        # Handle outliers
        df = self._handle_outliers(df)

        # Handle duplicates
        df = self._handle_duplicates(df)

        # Convert types
        df = self._convert_types(df)

        # Standardize values
        df = self._standardize_values(df)

        return df

    def _handle_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values.

        :param data: Input data
        :type data: pd.DataFrame
        :param fit: Whether to fit imputers
        :type fit: bool
        :return: Data with handled missing values
        :rtype: pd.DataFrame
        """
        df = data.copy()

        if self.cfg.missing_values.strategy == "remove":
            return df.dropna()

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.any():
            if fit:
                df[numeric_cols] = self.imputers["numeric"].fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.imputers["numeric"].transform(df[numeric_cols])

        # Handle categorical columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if cat_cols.any():
            if fit:
                df[cat_cols] = self.imputers["categorical"].fit_transform(df[cat_cols])
            else:
                df[cat_cols] = self.imputers["categorical"].transform(df[cat_cols])

        # Handle custom imputations
        for col, config in self.cfg.missing_values.custom.items():
            if col in df.columns:
                if config.method == "constant":
                    df[col].fillna(config.value, inplace=True)
                elif config.method == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif config.method == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def _handle_invalid_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle invalid values.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Data with handled invalid values
        :rtype: pd.DataFrame
        """
        df = data.copy()

        # Replace invalid values with NaN
        for value in self.cfg.invalid_values.replace_with_nan:
            df.replace(value, np.nan, inplace=True)

        # Apply custom mappings
        for col, mappings in self.cfg.invalid_values.custom_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mappings).fillna(df[col])

        return df

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Data with handled outliers
        :rtype: pd.DataFrame
        """
        df = data.copy()

        for col, config in self.cfg.outliers.custom.items():
            if col not in df.columns:
                continue

            if config.method == "clip":
                df[col] = df[col].clip(*config.range)
            elif config.method == "remove":
                mask = (df[col] >= config.range[0]) & (df[col] <= config.range[1])
                df = df[mask]
            elif config.method == "winsorize":
                df[col] = stats.mstats.winsorize(df[col], limits=config.limits)

        # General outlier handling
        if self.cfg.outliers.detection.method == "iqr":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.cfg.outliers.detection.threshold * IQR
                upper = Q3 + self.cfg.outliers.detection.threshold * IQR
                df[col] = df[col].clip(lower, upper)

        return df

    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate rows.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Data with handled duplicates
        :rtype: pd.DataFrame
        """
        if not self.cfg.duplicates.remove:
            return data

        return data.drop_duplicates(
            subset=self.cfg.duplicates.subset,
            keep=self.cfg.duplicates.keep,
            ignore_index=self.cfg.duplicates.ignore_index,
        )

    def _convert_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert column types.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Data with converted types
        :rtype: pd.DataFrame
        """
        df = data.copy()

        for col, dtype in self.cfg.type_conversion.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {e}")

        return df

    def _standardize_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize values.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Data with standardized values
        :rtype: pd.DataFrame
        """
        df = data.copy()

        # Apply case standardization
        if not self.cfg.standardization.case_sensitive:
            object_cols = df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                df[col] = df[col].str.lower()

        # Strip whitespace
        if self.cfg.standardization.strip_whitespace:
            object_cols = df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                df[col] = df[col].str.strip()

        # Remove special characters
        if self.cfg.standardization.remove_special_chars:
            object_cols = df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                df[col] = df[col].str.replace(r"[^\w\s]", "", regex=True)

        # Apply value mappings
        for col in df.columns:
            if col in self.cfg.standardization.mappings:
                df[col] = df[col].map(self.cfg.standardization.mappings[col])

        return df

    def validate_data(self, data: pd.DataFrame) -> List[str]:
        """Validate data against rules.

        :param data: Input data
        :type data: pd.DataFrame
        :return: List of validation errors
        :rtype: List[str]
        """
        errors = []

        for rule in self.cfg.validation.rules:
            if rule.type == "range":
                if data[rule.column].min() < rule.min or data[rule.column].max() > rule.max:
                    errors.append(f"Column {rule.column} contains values outside " f"range [{rule.min}, {rule.max}]")

            elif rule.type == "categorical":
                invalid = set(data[rule.column].unique()) - set(rule.categories)
                if invalid:
                    errors.append(f"Column {rule.column} contains invalid " f"categories: {invalid}")

            elif rule.type == "unique":
                if data[rule.column].duplicated().any():
                    errors.append(f"Column {rule.column} contains duplicate values")

        if errors and self.cfg.validation.raise_on_fail:
            raise ValueError("\n".join(errors))

        return errors
