"""
Data Management
=============

.. module:: pipeline.utils.data.manager
   :synopsis: Data loading, validation, and transformation utilities

.. moduleauthor:: aai540-group3
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataManager:
    """Manager for data operations."""

    def __init__(self, cfg: DictConfig):
        """Initialize data manager.

        :param cfg: Data configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.schema = self._load_schema()
        self.transformers = {}
        self.metadata = {
            "quality_metrics": {},
            "transformations": {},
            "splits": {},
        }

    def _load_schema(self) -> Dict[str, Any]:
        """Load data schema from configuration.

        :return: Data schema
        :rtype: Dict[str, Any]
        """
        return self.cfg.schema

    def load_data(self, path: Optional[Union[str, Path]] = None, **kwargs) -> pd.DataFrame:
        """Load data with validation.

        :param path: Optional path override
        :type path: Optional[Union[str, Path]]
        :param kwargs: Additional loading arguments
        :return: Loaded DataFrame
        :rtype: pd.DataFrame
        """
        path = path or self.cfg.path
        if not path:
            raise ValueError("No data path specified")

        try:
            df = self._read_data(path, **kwargs)
            self._validate_schema(df)
            self._collect_quality_metrics(df, "initial")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def save_data(self, df: pd.DataFrame, path: Optional[Union[str, Path]] = None, **kwargs) -> None:
        """Save data with metadata.

        :param df: DataFrame to save
        :type df: pd.DataFrame
        :param path: Optional path override
        :type path: Optional[Union[str, Path]]
        :param kwargs: Additional saving arguments
        """
        path = path or self.cfg.output_path
        if not path:
            raise ValueError("No output path specified")

        try:
            self._write_data(df, path, **kwargs)
            self._save_metadata(path)
            logger.info(f"Saved data to {path}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

    def _read_data(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read data based on file format.

        :param path: Data path
        :type path: Union[str, Path]
        :param kwargs: Additional reading arguments
        :return: Loaded DataFrame
        :rtype: pd.DataFrame
        """
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)
        elif path.suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif path.suffix == ".json":
            return pd.read_json(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _write_data(self, df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
        """Write data based on file format.

        :param df: DataFrame to write
        :type df: pd.DataFrame
        :param path: Output path
        :type path: Union[str, Path]
        :param kwargs: Additional writing arguments
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".parquet":
            df.to_parquet(path, **kwargs)
        elif path.suffix == ".csv":
            df.to_csv(path, index=False, **kwargs)
        elif path.suffix == ".json":
            df.to_json(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame against schema.

        :param df: DataFrame to validate
        :type df: pd.DataFrame
        :raises ValueError: If validation fails
        """
        for column, schema in self.schema.items():
            if schema.get("required", False) and column not in df.columns:
                raise ValueError(f"Required column missing: {column}")

            if column in df.columns:
                col_type = schema.get("type")
                if col_type == "category":
                    df[column] = df[column].astype("category")
                elif col_type == "int":
                    df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
                elif col_type == "float":
                    df[column] = pd.to_numeric(df[column], errors="coerce")

    def _collect_quality_metrics(self, df: pd.DataFrame, stage: str) -> None:
        """Collect data quality metrics.

        :param df: DataFrame to analyze
        :type df: pd.DataFrame
        :param stage: Analysis stage name
        :type stage: str
        """
        metrics = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": {
                "total": df.isnull().sum().sum(),
                "by_column": df.isnull().sum().to_dict(),
            },
            "data_types": df.dtypes.astype(str).to_dict(),
        }

        # Add numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            metrics["numerical_stats"] = {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
                for col in numerical_cols
            }

        self.metadata["quality_metrics"][stage] = metrics

    def split_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        :param target_column: Target column name
        :type target_column: str
        :return: Split DataFrames and Series
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # First split: train + validation, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.cfg.splits.test_size,
            random_state=self.cfg.splits.random_state,
            stratify=y if self.cfg.splits.stratify else None,
        )

        # Second split: train, validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=self.cfg.splits.val_size,
            random_state=self.cfg.splits.random_state,
            stratify=y_temp if self.cfg.splits.stratify else None,
        )

        # Record split information
        self.metadata["splits"] = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "stratified": self.cfg.splits.stratify,
            "random_state": self.cfg.splits.random_state,
        }

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _save_metadata(self, data_path: Union[str, Path]) -> None:
        """Save data processing metadata.

        :param data_path: Path to saved data
        :type data_path: Union[str, Path]
        """
        metadata_path = Path(data_path).parent / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
