import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_data(
    path: Union[str, Path],
    target_column: str,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from file.

    Args:
        path: Path to data file
        target_column: Name of target column
        feature_columns: List of feature columns to use

    Returns:
        Features and target
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load data based on file extension
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    # Select features
    if feature_columns is not None:
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        X = df[feature_columns]
    else:
        X = df.drop(target_column, axis=1)

    y = df[target_column]

    return X, y

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split data into train, validation, and test sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        random_state: Random state for reproducibility

    Returns:
        Train, validation, and test sets
    """
    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(
    data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
    path: Union[str, Path],
    format: str = "parquet"
) -> None:
    """Save data to file.

    Args:
        data: Data to save
        path: Path to save file
        format: File format (parquet or csv)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle tuple input
    if isinstance(data, tuple):
        X, y = data
        df = pd.concat([X, y], axis=1)
    else:
        df = data

    # Save based on format
    if format.lower() == "parquet":
        df.to_parquet(path)
    elif format.lower() == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved data to {path}")
