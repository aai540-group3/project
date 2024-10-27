import logging
import os
import sys
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function implementing the complete preprocessing pipeline."""
    try:
        # Define paths
        raw_data_path = "data/raw/data.csv"
        interim_data_path = "data/interim/data_cleaned.parquet"

        # Load raw data
        logger.info(f"Reading raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path, low_memory=False)
        logger.info(f"Initial shape: {df.shape}")

        # Log initial readmitted distribution
        logger.info(
            f"Initial readmitted distribution: {df['readmitted'].value_counts().to_dict()}"
        )

        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        logger.info(f"Columns after cleaning: {df.columns.tolist()}")

        # Drop columns with too many missing values
        columns_to_drop = [
            "weight",
            "payer_code",
            "medical_specialty",
            "citoglipton",
            "examide",
        ]
        df = df.drop(columns_to_drop, axis=1)
        logger.info(f"Dropped columns with high missing values: {columns_to_drop}")

        # Replace '?' with NaN
        df.replace("?", np.nan, inplace=True)

        # Handle missing values
        logger.info("Handling missing values...")

        # Identify numeric and categorical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        logger.info(f"Numeric columns: {numeric_columns.tolist()}")
        logger.info(f"Categorical columns: {categorical_columns.tolist()}")

        # For numeric columns, impute with median
        if len(numeric_columns) > 0:
            logger.info("Imputing numeric columns with median")
            df[numeric_columns] = df[numeric_columns].fillna(
                df[numeric_columns].median()
            )

        # For categorical columns, impute with mode
        if len(categorical_columns) > 0:
            logger.info("Imputing categorical columns with mode")
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        # Remove invalid entries
        logger.info("Removing invalid entries...")

        # Gender validation
        df = df[df["gender"].isin(["Male", "Female"])]

        # Race validation
        df = df.dropna(subset=["race"])
        df = df[df["race"] != "?"]

        # Diagnosis validation
        diagnosis_columns = ["diag_1", "diag_2", "diag_3"]
        df = df.dropna(subset=diagnosis_columns)
        for col in diagnosis_columns:
            df = df[df[col] != "?"]

        # Remove expired patients
        df = df[df["discharge_disposition_id"] != 11]

        logger.info(f"Shape after removing invalid entries: {df.shape}")

        # Generate unique ID
        df.reset_index(drop=True, inplace=True)
        df["id"] = df.index

        # Handle categorical variables
        logger.info("Encoding categorical variables...")

        # Map age to numerical values
        age_map = {
            "[0-10)": 5,
            "[10-20)": 15,
            "[20-30)": 25,
            "[30-40)": 35,
            "[40-50)": 45,
            "[50-60)": 55,
            "[60-70)": 65,
            "[70-80)": 75,
            "[80-90)": 85,
            "[90-100)": 95,
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

        for col in medication_cols:
            df[col] = df[col].replace({"No": 0, "Steady": 1, "Up": 1, "Down": 1})

        # Encode lab results
        df["a1cresult"] = df["a1cresult"].replace(
            {">7": 1, ">8": 1, "Norm": 0, "None": -99}
        )

        df["max_glu_serum"] = df["max_glu_serum"].replace(
            {">200": 1, ">300": 1, "Norm": 0, "None": -99}
        )

        # Encode admission type
        df["admission_type_id"] = df["admission_type_id"].replace(
            {2: 1, 7: 1, 6: 5, 8: 5}
        )

        # Encode discharge disposition
        discharge_mapping = {
            6: 1,
            8: 1,
            9: 1,
            13: 1,  # Home
            3: 2,
            4: 2,
            5: 2,
            14: 2,
            22: 2,
            23: 2,
            24: 2,  # Healthcare Facility
            12: 10,
            15: 10,
            16: 10,
            17: 10,  # Outpatient
            25: 18,
            26: 18,  # Psychiatric
        }
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(
            discharge_mapping
        )

        # Encode admission source
        admission_mapping = {
            2: 1,
            3: 1,  # Physician Referral
            5: 4,
            6: 4,
            10: 4,
            22: 4,
            25: 4,  # Transfer
            15: 9,
            17: 9,
            20: 9,
            21: 9,  # Emergency
            13: 11,
            14: 11,  # Other
        }
        df["admission_source_id"] = df["admission_source_id"].replace(admission_mapping)

        # Create diagnosis level features
        logger.info("Creating diagnosis level features...")
        for i in range(1, 4):
            col = f"diag_{i}"
            level_col = f"level1_diag{i}"
            df[level_col] = df[col].astype(str)

            # Handle V and E codes
            df.loc[df[level_col].str.contains("V", na=False), level_col] = "0"
            df.loc[df[level_col].str.contains("E", na=False), level_col] = "0"

            # Convert to numeric
            df[level_col] = pd.to_numeric(df[level_col], errors="coerce")

            # Map to categories
            conditions = [
                ((df[level_col] >= 390) & (df[level_col] < 460))
                | (df[level_col] == 785),
                ((df[level_col] >= 460) & (df[level_col] < 520))
                | (df[level_col] == 786),
                ((df[level_col] >= 520) & (df[level_col] < 580))
                | (df[level_col] == 787),
                (df[level_col] == 250),
                ((df[level_col] >= 800) & (df[level_col] < 1000)),
                ((df[level_col] >= 710) & (df[level_col] < 740)),
                ((df[level_col] >= 580) & (df[level_col] < 630))
                | (df[level_col] == 788),
                ((df[level_col] >= 140) & (df[level_col] < 240)),
            ]
            choices = range(1, 9)
            df[level_col] = np.select(conditions, choices, default=0)

        # Handle duplicates
        logger.info("Handling duplicates...")
        patient_id_col = "patient_nbr"
        if patient_id_col not in df.columns:
            patient_id_col = "patient_number"
            if patient_id_col not in df.columns:
                logger.warning(
                    "No patient number column found. Using encounter_id instead."
                )
                patient_id_col = "encounter_id"

        if patient_id_col in df.columns:
            original_len = len(df)
            df = df.drop_duplicates(subset=[patient_id_col], keep="first")
            logger.info(f"Removed {original_len - len(df)} duplicate records")

        # Handle outliers using IQR method for numeric columns
        logger.info("Handling outliers...")
        # Exclude certain columns from outlier detection
        columns_to_exclude = ["readmitted", "id", "patient_nbr", "encounter_id"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(
            columns_to_exclude
        )

        logger.info("Capping outliers...")
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Instead of removing rows, cap the values
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info(f"Capped outliers in {len(numeric_cols)} columns")

        # Process readmitted variable
        logger.info("Processing readmitted variable...")
        logger.info(
            f"Original readmitted distribution: {df['readmitted'].value_counts().to_dict()}"
        )

        # Convert readmitted to binary
        df["readmitted"] = df["readmitted"].replace({">30": 0, "<30": 1, "NO": 0})

        logger.info(
            f"Readmitted distribution after conversion: {df['readmitted'].value_counts().to_dict()}"
        )

        # Verify we have both classes
        if len(df["readmitted"].unique()) < 2:
            raise ValueError("Lost all positive cases during preprocessing!")

        # Final verification
        logger.info("Performing final verification...")

        # Check for remaining missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning("Remaining missing values:")
            logger.warning(missing_values[missing_values > 0])

        # Check data types
        logger.info("Final data types:")
        logger.info(df.dtypes)

        # Final verification of target variable
        logger.info("Final verification of target variable...")
        readmitted_dist = df["readmitted"].value_counts().to_dict()
        logger.info(f"Final readmitted distribution: {readmitted_dist}")
        if len(readmitted_dist) < 2:
            raise ValueError(f"Invalid target variable distribution: {readmitted_dist}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(interim_data_path), exist_ok=True)

        # Save processed data
        logger.info(f"Saving processed data to {interim_data_path}")
        df.to_parquet(interim_data_path, index=False)

        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Final shape: {df.shape}")

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}")
        logger.error("Full error:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
