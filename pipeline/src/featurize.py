import contextlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy as sp
from feast import Entity, Feature, FeatureStore, FeatureView, FileSource, ValueType
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def ignore_astimezone_error():
    """Temporarily suppresses errors from astimezone() calls."""
    try:
        with contextlib.suppress(TypeError):
            yield
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def main():
    """Main function implementing the complete feature engineering pipeline."""

    # Define target feature set
    TARGET_FEATURES = [
        "age",
        "time_in_hospital",
        "num_procedures",
        "num_medications",
        "number_outpatient_log1p",
        "number_emergency_log1p",
        "number_inpatient_log1p",
        "number_diagnoses",
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "AfricanAmerican",
        "Asian",
        "Caucasian",
        "Hispanic",
        "Other",
        "gender_1",
        "admission_type_id_3",
        "admission_type_id_5",
        "discharge_disposition_id_2",
        "discharge_disposition_id_7",
        "discharge_disposition_id_10",
        "discharge_disposition_id_18",
        "admission_source_id_4",
        "admission_source_id_7",
        "admission_source_id_9",
        "max_glu_serum_0",
        "max_glu_serum_1",
        "A1Cresult_0",
        "A1Cresult_1",
        "level1_diag1_1.0",
        "level1_diag1_2.0",
        "level1_diag1_3.0",
        "level1_diag1_4.0",
        "level1_diag1_5.0",
        "level1_diag1_6.0",
        "level1_diag1_7.0",
        "level1_diag1_8.0",
    ]

    # Define paths
    interim_data_path = "data/interim/data_cleaned.parquet"
    featured_data_path = "data/interim/data_featured.parquet"

    try:
        # Load preprocessed data
        logger.info("Loading preprocessed data...")
        df = pd.read_parquet(interim_data_path)
        logger.info(f"Initial shape: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")

        logger.info("Inspecting readmitted column...")
        logger.info(f"Readmitted column unique values: {df['readmitted'].unique()}")
        logger.info(
            f"Readmitted value counts: {df['readmitted'].value_counts().to_dict()}"
        )

        # Clean and convert data types
        logger.info("Cleaning and converting data types...")

        # First, handle any string columns that should be numeric
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    # Try to convert directly to numeric
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    # If direct conversion fails, try to extract numbers
                    try:
                        df[col] = (
                            df[col]
                            .astype(str)
                            .str.extract("(\d+)", expand=False)
                            .astype(float)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not convert column {col} to numeric: {e}"
                        )
                        # If it's a categorical column, keep as is
                        if col not in categorical_cols:
                            logger.warning(
                                f"Dropping column {col} due to conversion failure"
                            )
                            df = df.drop(columns=[col])

        # Remove any columns that are all NaN
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            logger.info(f"Dropping columns with all null values: {null_cols}")
            df = df.drop(columns=null_cols)

        # Fill remaining NaN values with mean for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # Fill remaining NaN values with mode for categorical columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else 0
            )

        logger.info(f"Shape after cleaning: {df.shape}")
        logger.info("Data types after cleaning:")
        for col in df.columns:
            logger.info(f"  {col}: {df[col].dtype}")

        # Store and verify target variable
        logger.info("Processing target variable...")
        if "readmitted" not in df.columns:
            raise ValueError("Target variable 'readmitted' not found in dataset")

        # Store original values for logging
        original_readmitted_values = df["readmitted"].value_counts().to_dict()
        logger.info(f"Original readmitted values: {original_readmitted_values}")

        # Convert readmitted to binary (0 for no readmission, 1 for readmission)
        logger.info("Converting readmitted variable to binary...")
        readmission_map = {
            ">30": 0,  # No readmission (more than 30 days)
            "<30": 1,  # Readmission (less than 30 days)
            "NO": 0,  # No readmission
            0: 0,  # Already binary
            1: 1,  # Already binary
        }

        # Create target variable before dropping from dataframe
        y = df["readmitted"].map(readmission_map)
        if y.isnull().any():
            logger.warning(f"Found {y.isnull().sum()} null values in target variable")
            y = y.fillna(0)  # Fill any remaining NaN with 0 (no readmission)

        y = y.astype(int)

        # Verify target variable distribution
        logger.info(
            f"Target variable distribution after conversion: {y.value_counts().to_dict()}"
        )

        # Check for sufficient classes
        if len(y.unique()) < 2:
            logger.error("Insufficient classes in target variable")
            logger.error(f"Original values: {original_readmitted_values}")
            logger.error(f"Converted values: {y.unique()}")
            raise ValueError(f"Target variable has insufficient classes: {y.unique()}")

        # Now drop the column from the dataframe
        df = df.drop("readmitted", axis=1)

        # 1. Create medication-related features
        logger.info("Creating medication-related features...")
        medication_cols = [
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "glipizide",
            "glyburide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "insulin",
            "glyburide-metformin",
            "tolazamide",
            "metformin-pioglitazone",
            "metformin-rosiglitazone",
            "glimepiride-pioglitazone",
            "glipizide-metformin",
            "troglitazone",
            "tolbutamide",
            "acetohexamide",
        ]

        df["total_medications"] = df[medication_cols].sum(axis=1)
        df["medication_density"] = df["total_medications"] / df["time_in_hospital"]
        df["insulin_with_oral"] = (
            (df["insulin"] == 1)
            & (df[medication_cols].drop("insulin", axis=1).sum(axis=1) > 0)
        ).astype(int)
        df["numchange"] = df[medication_cols].sum(axis=1)
        df["nummed"] = df[medication_cols].sum(axis=1)

        # 2. Create service utilization features
        logger.info("Creating service utilization features...")
        df["total_encounters"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )
        df["encounter_per_time"] = df["total_encounters"] / df["time_in_hospital"]

        # 3. Create procedure-related features
        logger.info("Creating procedure-related features...")
        df["procedures_per_day"] = df["num_procedures"] / df["time_in_hospital"]
        df["lab_procedures_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"]
        df["procedures_to_medications"] = df["num_procedures"] / (
            df["num_medications"] + 1
        )

        # 4. Create diagnostic-related features
        logger.info("Creating diagnostic-related features...")
        df["diagnoses_per_encounter"] = df["number_diagnoses"] / (
            df["total_encounters"] + 1
        )

        # 5. Log transform features with high skewness
        logger.info("Applying log transformations...")
        skewed_features = ["number_outpatient", "number_emergency", "number_inpatient"]
        for col in skewed_features:
            if col in df.columns:
                df[f"{col}_log1p"] = np.log1p(df[col])
                logger.info(f"Created log transform for {col}")

        # Drop original skewed columns
        df = df.drop(
            columns=[
                col
                for col in skewed_features + ["service_utilization"]
                if col in df.columns
            ]
        )

        # 6. Create interaction features
        logger.info("Creating interaction features...")
        interactions = [
            ("num_medications", "time_in_hospital"),
            ("num_procedures", "time_in_hospital"),
            ("num_lab_procedures", "time_in_hospital"),
            ("number_diagnoses", "time_in_hospital"),
            ("age", "number_diagnoses"),
            ("age", "num_medications"),
            ("total_medications", "number_diagnoses"),
            ("num_medications", "num_procedures"),
            ("time_in_hospital", "num_lab_procedures"),
            ("num_medications", "num_lab_procedures"),
            ("change", "num_medications"),
            ("num_medications", "numchange"),
        ]

        for feat1, feat2 in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                feature_name = f"{feat1}_x_{feat2}"
                df[feature_name] = df[feat1] * df[feat2]
                logger.info(f"Created interaction feature: {feature_name}")

        # 7. Create ratio features
        logger.info("Creating ratio features...")
        df["procedure_medication_ratio"] = df["num_procedures"] / (
            df["num_medications"] + 1
        )
        df["lab_procedure_ratio"] = df["num_lab_procedures"] / (
            df["num_procedures"] + 1
        )
        df["diagnosis_procedure_ratio"] = df["number_diagnoses"] / (
            df["num_procedures"] + 1
        )

        # 8. One-hot encode categorical variables
        logger.info("One-hot encoding categorical variables...")
        categorical_cols = [
            "race",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "level1_diag1",
        ]

        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies.iloc[:, 1:]], axis=1)
                df.drop(columns=[col], inplace=True)
                logger.info(f"Created dummy variables for {col}")

        # 9. Ensure all required features exist
        for feature in TARGET_FEATURES:
            if feature not in df.columns:
                logger.warning(f"Missing expected feature: {feature}")
                if "_" in feature:
                    base_col, val = feature.rsplit("_", 1)
                    logger.info(f"Creating dummy variable {feature}")
                    df[feature] = 0

        # 10. Standardize numeric features
        logger.info("Standardizing numeric features...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        # 11. Handle outliers
        logger.info("Handling outliers...")
        original_len = len(df)

        # Define key features for outlier detection
        key_features = [
            "time_in_hospital",
            "num_procedures",
            "num_medications",
            "num_lab_procedures",
            "number_diagnoses",
        ]

        # Calculate z-scores only for key features
        z_scores = pd.DataFrame()
        for col in key_features:
            if col in df.columns:
                z_scores[col] = np.abs(sp.stats.zscore(df[col], nan_policy="omit"))

        # Instead of removing, cap the outliers at 5 standard deviations
        for col in key_features:
            if col in df.columns:
                lower_bound = df[col].mean() - 5 * df[col].std()
                upper_bound = df[col].mean() + 5 * df[col].std()
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info(f"Capped outliers in {len(key_features)} features")

        # Verify data quality before SMOTE
        logger.info("Verifying data quality before SMOTE...")
        logger.info(f"Number of samples: {len(df)}")
        logger.info(f"Number of features: {df.shape[1]}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(
            f"Infinite values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}"
        )

        # Handle any remaining missing or infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())

        # 12. Apply SMOTE
        logger.info("Applying SMOTE to balance classes...")
        logger.info(f"Class distribution before SMOTE: {y.value_counts().to_dict()}")

        try:
            # Check if we have enough samples
            if len(df) < 2:
                raise ValueError("Not enough samples for SMOTE after preprocessing")

            # Check class distribution
            if len(y.unique()) < 2:
                raise ValueError(
                    f"Need at least 2 classes for SMOTE, found {len(y.unique())}"
                )

            # Apply SMOTE with adjusted parameters
            smote = SMOTE(random_state=42, n_jobs=-1)
            X_resampled, y_resampled = smote.fit_resample(df, y)

            logger.info(
                f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}"
            )

        except Exception as e:
            logger.warning(f"SMOTE failed: {str(e)}")
            logger.warning("Proceeding without SMOTE balancing")
            X_resampled, y_resampled = df, y

        # Convert back to DataFrame
        df_balanced = pd.DataFrame(X_resampled, columns=df.columns)
        df_balanced["readmitted"] = y_resampled

        # 13. Add timestamps for Feast
        logger.info("Adding timestamp fields...")
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
        df_balanced["event_timestamp"] = pd.date_range(
            start=start_date, periods=len(df_balanced), freq="T", tz="UTC"
        )
        df_balanced["created_timestamp"] = pd.Timestamp.now(tz="UTC")

        # 14. Save featured data
        logger.info(f"Saving featured data to {featured_data_path}")
        os.makedirs(os.path.dirname(featured_data_path), exist_ok=True)
        df_balanced.to_parquet(featured_data_path, index=False)

        # 15. Set up Feast feature store
        logger.info("Setting up Feast feature store...")
        patient = Entity(
            name="id", value_type=ValueType.INT64, description="Patient ID"
        )

        data_source = FileSource(
            path=os.path.abspath(featured_data_path),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )

        features = []
        for col in df_balanced.columns:
            if col not in ["readmitted", "event_timestamp", "created_timestamp", "id"]:
                dtype = (
                    ValueType.DOUBLE
                    if pd.api.types.is_numeric_dtype(df_balanced[col])
                    else ValueType.STRING
                )
                features.append(Feature(name=col, dtype=dtype))

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

        # 16. Materialize features
        logger.info("Materializing features...")
        try:
            end_date = df_balanced["event_timestamp"].max()
            with ignore_astimezone_error():
                store.materialize_incremental(end_date)
                store.apply([patient, feature_view])
            logger.info(f"Features materialized up to {end_date} successfully")
        except Exception as e:
            logger.error(f"Materialization error: {str(e)}")
            raise

        # Log final statistics
        logger.info("Feature engineering pipeline completed successfully!")
        logger.info(f"Final feature set shape: {df_balanced.shape}")
        logger.info("Final feature set:")
        for feature in sorted(df_balanced.columns):
            if feature not in [
                "readmitted",
                "event_timestamp",
                "created_timestamp",
                "id",
            ]:
                logger.info(f"  - {feature}: {df_balanced[feature].dtype}")

    except Exception as e:
        logger.error(f"An error occurred during feature engineering: {str(e)}")
        logger.error("Full error:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
