#!/usr/bin/env python3
"""
pipeline.src.featurize

This module performs feature engineering on a dataset for a diabetes prediction model.
It includes data preprocessing, feature generation, and integration with the Feast feature store.

Attributes:
    logger (Logger): Configured logger for the module.

Example:
    To execute the pipeline, run:

        $ python featurize.py

"""

import contextlib
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from feast import Entity, Feature, FeatureStore, FeatureView, FileSource, ValueType
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def ignore_astimezone_error():
    """Temporarily suppress errors from astimezone() calls in Feast materialization.

    This context manager is useful for handling timezone errors that may arise
    during incremental feature materialization with Feast.

    Yields:
        None

    Raises:
        Exception: If an unexpected error occurs, it logs the error.
    """
    try:
        with contextlib.suppress(TypeError):
            yield
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def main():
    """Main function implementing the feature engineering pipeline and Feast integration.

    This function executes the complete feature engineering workflow:
    1. Loads and preprocesses the dataset.
    2. Performs feature engineering, including scaling and transformations.
    3. Sets up and materializes features with the Feast feature store.

    Raises:
        ValueError: If the target variable 'readmitted' is missing or has insufficient classes.
        Exception: For errors during Feast setup, materialization, or feature engineering.
    """
    TARGET_FEATURES = [
        "age", "time_in_hospital", "num_procedures", "num_medications",
        "number_outpatient_log1p", "number_emergency_log1p", "number_inpatient_log1p",
        "number_diagnoses", "metformin", "repaglinide", "nateglinide",
        "chlorpropamide", "glimepiride", "glipizide", "glyburide",
        "pioglitazone", "rosiglitazone", "acarbose", "tolazamide", "insulin",
        "glyburide-metformin", "AfricanAmerican", "Asian", "Caucasian",
        "Hispanic", "Other", "gender_1", "admission_type_id_3",
        "admission_type_id_5", "discharge_disposition_id_2",
        "discharge_disposition_id_7", "discharge_disposition_id_10",
        "discharge_disposition_id_18", "admission_source_id_4",
        "admission_source_id_7", "admission_source_id_9", "max_glu_serum_0",
        "max_glu_serum_1", "A1Cresult_0", "A1Cresult_1", "level1_diag1_1.0",
        "level1_diag1_2.0", "level1_diag1_3.0", "level1_diag1_4.0",
        "level1_diag1_5.0", "level1_diag1_6.0", "level1_diag1_7.0",
        "level1_diag1_8.0"
    ]

    interim_data_path = "data/interim/data_cleaned.parquet"
    featured_data_path = "data/interim/data_featured.parquet"
    feature_store_path = "."

    try:
        logger.info("Loading preprocessed data...")
        df = pd.read_parquet(interim_data_path)
        logger.info(f"Initial dataset shape: {df.shape}")

        # Ensure correct types and handle categorical data
        logger.info("Cleaning and converting data types...")
        categorical_cols = [
            "race", "gender", "admission_type_id",
            "discharge_disposition_id", "admission_source_id", "level1_diag1",
        ]

        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    try:
                        df[col] = (
                            df[col].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
                        )
                    except Exception:
                        if col not in categorical_cols:
                            logger.warning(f"Dropping column {col} due to conversion failure")
                            df = df.drop(columns=[col])

        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            logger.info(f"Dropping columns with all null values: {null_cols}")
            df = df.drop(columns=null_cols)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Preprocess the target variable
        if "readmitted" not in df.columns:
            raise ValueError("Target variable 'readmitted' not found in dataset")

        original_readmitted_values = df["readmitted"].value_counts().to_dict()
        logger.info(f"Original 'readmitted' value distribution: {original_readmitted_values}")

        readmission_map = {">30": 0, "<30": 1, "NO": 0, 0: 0, 1: 1}
        y = df["readmitted"].map(readmission_map).fillna(0).astype(int)
        if len(y.unique()) < 2:
            raise ValueError(f"Target variable has insufficient classes: {y.unique()}")

        df = df.drop("readmitted", axis=1)

        # Feature generation steps
        logger.info("Creating medication-related features...")
        medication_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "glipizide", "glyburide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "insulin",
            "glyburide-metformin", "tolazamide", "metformin-pioglitazone",
            "metformin-rosiglitazone", "glimepiride-pioglitazone",
            "glipizide-metformin", "troglitazone", "tolbutamide", "acetohexamide"
        ]
        df["total_medications"] = df[medication_cols].sum(axis=1)
        df["medication_density"] = df["total_medications"] / df["time_in_hospital"]

        logger.info("Creating service utilization features...")
        df["total_encounters"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )

        logger.info("Creating procedure-related features...")
        df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"]

        logger.info("Applying log transformations for skewed features...")
        skewed_features = ["number_outpatient", "number_emergency", "number_inpatient"]
        for col in skewed_features:
            if col in df.columns:
                df[f"{col}_log1p"] = np.log1p(df[col])
                logger.info(f"Created log transform for {col}")

        # Drop original skewed columns
        df = df.drop(columns=[col for col in skewed_features if col in df.columns])

        logger.info("One-hot encoding categorical variables...")
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies.iloc[:, 1:]], axis=1)
                df.drop(columns=[col], inplace=True)
                logger.info(f"Created dummy variables for {col}")

        # Standardize numeric features
        logger.info("Standardizing numeric features...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        # Verify data quality
        logger.info("Verifying data quality before saving...")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())

        # Add target variable back to the dataframe
        df["readmitted"] = y

        # Add timestamps for Feast
        logger.info("Adding timestamp fields...")
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        df["event_timestamp"] = pd.date_range(
            start=start_date, periods=len(df), freq="min", tz="UTC"
        )
        df["created_timestamp"] = pd.Timestamp.now(tz="UTC")

        # Save featured data
        logger.info(f"Saving featured data to {featured_data_path}")
        os.makedirs(os.path.dirname(featured_data_path), exist_ok=True)
        df.to_parquet(featured_data_path, index=False)

        # Set up Feast feature store
        logger.info("Setting up Feast feature store...")
        patient = Entity(
            name="id", value_type=ValueType.INT64, description="Patient ID"
        )

        data_source = FileSource(
            path=os.path.abspath(featured_data_path),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )

        features = [
            Feature(name=col, dtype=ValueType.DOUBLE if pd.api.types.is_numeric_dtype(df[col]) else ValueType.STRING)
            for col in df.columns if col not in ["readmitted", "event_timestamp", "created_timestamp", "id"]
        ]

        feature_view = FeatureView(
            name="diabetes_features",
            entities=["id"],
            ttl=timedelta(days=365 * 10),
            features=features,
            batch_source=data_source,
            online=True,
        )

        store = FeatureStore(repo_path=".")
        store.apply([patient, feature_view])

        # Materialize features
        logger.info("Materializing features...")
        try:
            end_date = df["event_timestamp"].max()
            with ignore_astimezone_error():
                store.materialize_incremental(end_date)
                store.apply([patient, feature_view])
            logger.info(f"Features materialized up to {end_date} successfully")
        except Exception as e:
            logger.error(f"Materialization error: {str(e)}")
            raise

        logger.info("Feature engineering pipeline completed successfully!")
        logger.info(f"Final feature set shape: {df.shape}")
        logger.info("Final feature set:")
        for feature in sorted(df.columns):
            if feature not in ["readmitted", "event_timestamp", "created_timestamp", "id"]:
                logger.info(f"  - {feature}: {df[feature].dtype}")

    except Exception as e:
        logger.error(f"An error occurred during feature engineering: {str(e)}")
        logger.error("Full error:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
