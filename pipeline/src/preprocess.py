#!/usr/bin/env python3
"""
pipeline.src.preprocess

This script preprocesses a raw dataset by cleaning, imputing missing values, encoding categorical variables,
handling outliers, and saving the processed data for feature engineering.

Attributes:
    logger (Logger): Configured logger for logging messages.

Example:
    To run the preprocessing, use:

        $ python preprocess.py
"""

import logging
import os

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main function implementing the complete preprocessing pipeline.

    This function performs the following steps:
    1. Loads the raw dataset.
    2. Cleans data by handling missing values, outliers, and categorical encoding.
    3. Saves the processed data and feature names for further processing.

    Raises:
        ValueError: If the 'readmitted' column loses all positive cases during preprocessing.
        Exception: Logs any unexpected error during the preprocessing pipeline and re-raises it.
    """
    try:
        # Define paths
        raw_data_path = "data/raw/data.csv"
        interim_data_path = "data/interim/data_cleaned.parquet"
        feature_names_path = "data/interim/feature_names.joblib"

        # Load raw data
        logger.info(f"Reading raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, low_memory=False)
        logger.info(f"Initial shape: {df.shape}")

        # Log initial distribution of 'readmitted' variable
        logger.info(f"Initial readmitted distribution: {df['readmitted'].value_counts().to_dict()}")

        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        logger.info(f"Columns after cleaning: {df.columns.tolist()}")

        # Drop columns with high missing values
        columns_to_drop = ["weight", "payer_code", "medical_specialty", "citoglipton", "examide"]
        df = df.drop(columns=columns_to_drop, axis=1)
        logger.info(f"Dropped columns with high missing values: {columns_to_drop}")

        # Replace '?' with NaN
        df.replace("?", np.nan, inplace=True)

        # Handle missing values
        logger.info("Handling missing values...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        logger.info(f"Numeric columns: {numeric_columns.tolist()}")
        logger.info(f"Categorical columns: {categorical_columns.tolist()}")

        # Impute missing values
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

        # Remove invalid entries based on specific conditions
        logger.info("Removing invalid entries...")
        df = df[df["gender"].isin(["Male", "Female"])]
        df = df.dropna(subset=["race"])
        diagnosis_columns = ["diag_1", "diag_2", "diag_3"]
        df = df.dropna(subset=diagnosis_columns)
        df = df[df["discharge_disposition_id"] != 11]
        logger.info(f"Shape after removing invalid entries: {df.shape}")

        # Generate unique ID and reset index
        df.reset_index(drop=True, inplace=True)
        df["id"] = df.index

        # Encode categorical variables
        logger.info("Encoding categorical variables...")
        age_map = {
            "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
            "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
            "[80-90)": 85, "[90-100)": 95,
        }
        df["age"] = df["age"].map(age_map)

        # Encode binary variables
        binary_mappings = {
            "gender": {"Male": 1, "Female": 0},
            "change": {"Ch": 1, "No": 0},
            "diabetesmed": {"Yes": 1, "No": 0},
        }
        for col, mapping in binary_mappings.items():
            df[col] = df[col].map(mapping)
            logger.info(f"Encoded binary variable: {col}")

        # Encode medication columns
        medication_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "glipizide", "glyburide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "insulin",
            "glyburide-metformin", "tolazamide", "metformin-pioglitazone",
            "metformin-rosiglitazone", "glimepiride-pioglitazone",
            "glipizide-metformin", "troglitazone", "tolbutamide", "acetohexamide",
        ]
        for col in medication_cols:
            df[col] = df[col].replace({"No": 0, "Steady": 1, "Up": 1, "Down": 1})

        # Encode lab results
        df["a1cresult"] = df["a1cresult"].replace({">7": 1, ">8": 1, "Norm": 0, "None": -99})
        df["max_glu_serum"] = df["max_glu_serum"].replace({">200": 1, ">300": 1, "Norm": 0, "None": -99})

        # Encode admission type and discharge mappings
        df["admission_type_id"] = df["admission_type_id"].replace({2: 1, 7: 1, 6: 5, 8: 5})
        discharge_mapping = {6: 1, 8: 1, 9: 1, 13: 1, 3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2, 12: 10, 15: 10, 16: 10, 17: 10, 25: 18, 26: 18}
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(discharge_mapping)

        # Diagnosis level features
        logger.info("Creating diagnosis level features...")
        for i in range(1, 4):
            col = f"diag_{i}"
            level_col = f"level1_diag{i}"
            df[level_col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
            df[level_col] = np.select(
                [
                    ((df[level_col] >= 390) & (df[level_col] < 460)) | (df[level_col] == 785),
                    ((df[level_col] >= 460) & (df[level_col] < 520)) | (df[level_col] == 786),
                    ((df[level_col] >= 520) & (df[level_col] < 580)) | (df[level_col] == 787),
                    (df[level_col] == 250),
                    ((df[level_col] >= 800) & (df[level_col] < 1000)),
                    ((df[level_col] >= 710) & (df[level_col] < 740)),
                    ((df[level_col] >= 580) & (df[level_col] < 630)) | (df[level_col] == 788),
                    ((df[level_col] >= 140) & (df[level_col] < 240)),
                ],
                range(1, 9),
                default=0
            )

        # Remove duplicates
        logger.info("Handling duplicates...")
        patient_id_col = "patient_nbr" if "patient_nbr" in df.columns else "encounter_id"
        original_len = len(df)
        df = df.drop_duplicates(subset=[patient_id_col], keep="first")
        logger.info(f"Removed {original_len - len(df)} duplicate records")

        # Outlier handling for numeric columns
        logger.info("Handling outliers...")
        columns_to_exclude = ["readmitted", "id", "patient_nbr", "encounter_id"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(columns_to_exclude)

        logger.info("Capping outliers...")
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Process and verify target variable 'readmitted'
        logger.info("Processing readmitted variable...")
        df["readmitted"] = df["readmitted"].replace({">30": 0, "<30": 1, "NO": 0})
        if len(df["readmitted"].unique()) < 2:
            raise ValueError("Lost all positive cases during preprocessing!")

        # Final verification
        logger.info("Performing final verification...")
        if df.isnull().sum().any():
            logger.warning("Remaining missing values detected.")

        # Save processed data and feature names
        os.makedirs(os.path.dirname(interim_data_path), exist_ok=True)
        df.to_parquet(interim_data_path, index=False)
        pd.DataFrame(df.columns.tolist(), columns=["features"]).to_csv(feature_names_path, index=False)

        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Final shape: {df.shape}")

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}")
        logger.error("Full error:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
