"""
.. module:: src.data.cleaning
   :synopsis: Clean the raw dataset.

This script loads the raw dataset, performs various data cleaning operations, and saves the cleaned
dataset as a CSV file in the interim data directory.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning operations.

    :param df: The raw Pandas DataFrame.
    :type df: pd.DataFrame
    :return: The cleaned Pandas DataFrame.
    :rtype: pd.DataFrame
    """

    logger.info(f"Columns in the dataset: {', '.join(df.columns)}")

    # Remove duplicates
    original_shape = df.shape
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")

    # Handle missing values in numerical columns
    numerical_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]

    for col in numerical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            logger.info(f"Column '{col}' has {missing_count} missing values")
            df[col] = df[col].fillna(df[col].median())

    # Convert numerical columns to appropriate data types
    df[numerical_cols] = df[numerical_cols].astype("float32")

    # Handle binary categorical variables
    binary_cols = [
        col
        for col in df.columns
        if df[col].nunique() == 2 and col not in numerical_cols
    ]
    df[binary_cols] = df[binary_cols].astype("bool")

    # Handle 'readmitted' column
    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].astype("int32")
    else:
        logger.warning("'readmitted' column not found in the dataset")

    # Remove features with more than 50% missing values
    columns_to_drop = df.columns[df.isnull().mean() > 0.5]
    df = df.drop(columns=columns_to_drop)
    if len(columns_to_drop) > 0:
        logger.info(
            f"Dropped columns with >50% missing values: {', '.join(columns_to_drop)}"
        )

    return df


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Clean the raw dataset.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises Exception: If any error occurs during data cleaning.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Get data paths
        raw_data_path = Path(to_absolute_path(cfg.dataset.path.raw)) / "data.csv"
        interim_data_path = Path(to_absolute_path(cfg.dataset.path.interim)) / "data.csv"

        # Load raw data
        logger.info("Loading raw data...")
        df = pd.read_csv(raw_data_path)

        # Perform data cleaning
        logger.info("Performing data cleaning...")
        df_cleaned = clean_data(df)

        # Save cleaned data
        logger.info("Saving cleaned data...")
        interim_data_path.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(interim_data_path, index=False)
        logger.info(f"Interim data saved to {interim_data_path}")

        # Log data quality metrics
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Cleaned data shape: {df_cleaned.shape}")
        logger.info(f"Columns in cleaned data: {', '.join(df_cleaned.columns)}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
