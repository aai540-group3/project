import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler

from ..utils.experiment import ExperimentTracker

logger = logging.getLogger(__name__)


class DataPreprocessingStage:
    """Data preprocessing stage."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tracker = ExperimentTracker(cfg, cfg.experiment.name)

    def run(self) -> None:
        """Execute preprocessing pipeline."""
        self.tracker.start_run(run_name="preprocessing")

        try:
            # Load raw data
            df = pd.read_parquet(self.cfg.paths.raw / "data.parquet")
            logger.info(f"Initial shape: {df.shape}")

            # Initial data quality check
            self._log_data_quality(df, "initial")

            # Clean data
            df = self._clean_data(df)

            # Handle missing values
            df = self._handle_missing_values(df)

            # Handle categorical variables
            df = self._handle_categorical(df)

            # Process target variable
            df = self._process_target(df)

            # Handle outliers
            df = self._handle_outliers(df)

            # Final data quality check
            self._log_data_quality(df, "final")

            # Save preprocessed data
            self._save_data(df)

            logger.info("Preprocessing completed successfully")

        finally:
            self.tracker.end_run()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on configuration."""
        # Drop unnecessary columns
        drop_cols = self.cfg.preprocessing.drop_columns
        df = df.drop(columns=drop_cols, errors="ignore")

        # Clean column names
        df.columns = df.columns.str.strip().str.lower()

        # Remove duplicates
        df = df.drop_duplicates(
            subset=self.cfg.preprocessing.duplicate_subset,
            keep=self.cfg.preprocessing.duplicate_keep,
        )

        # Remove invalid entries based on conditions
        for condition in self.cfg.preprocessing.invalid_conditions:
            df = df.query(condition)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        # Replace '?' with NaN
        df = df.replace("?", np.nan)

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            strategy = self.cfg.preprocessing.missing_values.numeric
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())

        # Handle categorical columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            strategy = self.cfg.preprocessing.missing_values.categorical
            if strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif strategy == "constant":
                df[col] = df[col].fillna(
                    self.cfg.preprocessing.missing_values.fill_value
                )

        return df

    def _handle_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical variables."""
        # Binary mappings
        for col, mapping in self.cfg.preprocessing.binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Ordinal mappings
        for col, mapping in self.cfg.preprocessing.ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # One-hot encoding
        if self.cfg.preprocessing.one_hot_encoding.enabled:
            for col in self.cfg.preprocessing.one_hot_encoding.columns:
                if col in df.columns:
                    dummies = pd.get_dummies(
                        df[col],
                        prefix=col,
                        drop_first=self.cfg.preprocessing.one_hot_encoding.drop_first,
                    )
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(columns=[col], inplace=True)

        return df

    def _process_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process target variable."""
        target_col = self.cfg.preprocessing.target_column
        if target_col in df.columns:
            mapping = self.cfg.preprocessing.target_mapping
            df[target_col] = df[target_col].map(mapping)

            # Verify binary classification
            unique_values = df[target_col].unique()
            if len(unique_values) != 2:
                raise ValueError(
                    f"Target variable should be binary, got {unique_values}"
                )

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        if self.cfg.preprocessing.outlier_handling.enabled:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = (
                    Q1 - self.cfg.preprocessing.outlier_handling.iqr_multiplier * IQR
                )
                upper_bound = (
                    Q3 + self.cfg.preprocessing.outlier_handling.iqr_multiplier * IQR
                )

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _log_data_quality(self, df: pd.DataFrame, stage: str) -> None:
        """Log data quality metrics."""
        metrics = {
            f"{stage}_shape": df.shape,
            f"{stage}_missing_values": df.isnull().sum().to_dict(),
            f"{stage}_unique_values": {col: df[col].nunique() for col in df.columns},
        }

        if self.cfg.preprocessing.target_column in df.columns:
            metrics[f"{stage}_target_distribution"] = (
                df[self.cfg.preprocessing.target_column].value_counts().to_dict()
            )

        self.tracker.log_metrics(metrics)

    def _save_data(self, df: pd.DataFrame) -> None:
        """Save preprocessed data."""
        output_path = Path(self.cfg.paths.interim) / "data_cleaned.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path)
        logger.info(f"Saved preprocessed data to {output_path}")
