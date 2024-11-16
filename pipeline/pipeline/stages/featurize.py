"""
Featurize Stage
===============

.. module:: pipeline.stages.featurize
   :synopsis: This module handles feature engineering

.. moduleauthor:: aai540-group3
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from .stage import Stage

MEDICATIONS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

AGE_MAP = {
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

BINARY_MAP = {
    "gender": {"Male": 1, "Female": 0},
    "change": {"Ch": 1, "No": 0},
    "diabetesMed": {"Yes": 1, "No": 0},
}

LAB_MAP = {
    "A1Cresult": {">8": 1, ">7": 1, "Norm": 0},
    "max_glu_serum": {">300": 1, ">200": 1, "Norm": 0},
}

DISCHARGE_MAP = {
    6: 1, 8: 1, 9: 1, 13: 1,                       # HOME 
    3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2,  # HEALTHCARE FACILITY
    12: 10, 15: 10, 16: 10, 17: 10,                # OUTPATIENT
    25: 18, 26: 18,                                # PSYCHIATRIC 
}  # fmt: off

ADMISSION_MAP = {
    2: 1, 3: 1,                       # PHYSICIAN REFERRAL
    5: 4, 6: 4, 10: 4, 22: 4, 25: 4,  # TRANSFER
    15: 9, 17: 9, 20: 9, 21: 9,       # EMERGENCY
    13: 11, 14: 11,                   # OTHER
}  # fmt: off

ADMISSION_TYPE_MAP = {
    2: 1, 7: 1,  # EMERGENCY/URGENT
    6: 5, 8: 5,  # OTHER
}  # fmt: off

MEDICATION_STATUS_MAP = {"No": 0, "Steady": 1, "Up": 1, "Down": 1}

DIAGNOSIS_CATEGORIES = {
    "circulatory": lambda x: ((x >= 390) & (x < 460)) | (x == 785),
    "respiratory": lambda x: ((x >= 460) & (x < 520)) | (x == 786),
    "digestive": lambda x: ((x >= 520) & (x < 580)) | (x == 787),
    "diabetes": lambda x: (x == 250),
    "injury": lambda x: (x >= 800) & (x < 1000),
    "musculoskeletal": lambda x: (x >= 710) & (x < 740),
    "genitourinary": lambda x: ((x >= 580) & (x < 630)) | (x == 788),
    "neoplasms": lambda x: (x >= 140) & (x < 240),
}

SKEWED_FEATURES = ["number_outpatient", "number_inpatient", "number_emergency"]

INTERACTION_TERMS = [
    ("age", "number_diagnoses"),
    ("change", "num_medications"),
    ("num_medications", "num_lab_procedures"),
    ("num_medications", "num_procedures"),
    ("num_medications", "number_diagnoses"),
    ("num_medications", "numchange"),
    ("num_medications", "time_in_hospital"),
    ("number_diagnoses", "time_in_hospital"),
    ("time_in_hospital", "num_lab_procedures"),
]

NUMERIC_FEATURES = [
    "age",
    "num_medications",
    "num_procedures",
    "numchange",
    "service_utilization",
    "time_in_hospital",
]

CATEGORICAL_FEATURES = [
    "A1Cresult",
    "admission_source_id",
    "admission_type_id",
    "discharge_disposition_id",
    "level1_diag1",
    "max_glu_serum",
    "race",
]

TARGET_FEATURE_SET = [
    "A1Cresult_0",
    "A1Cresult_1",
    "acarbose",
    "admission_source_id_4",
    "admission_source_id_7",
    "admission_source_id_9",
    "admission_type_id_3",
    "admission_type_id_5",
    "AfricanAmerican",
    "age",
    "age|number_diagnoses",
    "Asian",
    "Caucasian",
    "change|num_medications",
    "chlorpropamide",
    "discharge_disposition_id_10",
    "discharge_disposition_id_18",
    "discharge_disposition_id_2",
    "discharge_disposition_id_7",
    "gender_1",
    "glimepiride",
    "glipizide",
    "glyburide-metformin",
    "glyburide",
    "Hispanic",
    "insulin",
    "level1_diag1_1.0",
    "level1_diag1_2.0",
    "level1_diag1_3.0",
    "level1_diag1_4.0",
    "level1_diag1_5.0",
    "level1_diag1_6.0",
    "level1_diag1_7.0",
    "level1_diag1_8.0",
    "max_glu_serum_0",
    "max_glu_serum_1",
    "metformin",
    "nateglinide",
    "num_medications",
    "num_medications|num_lab_procedures",
    "num_medications|num_procedures",
    "num_medications|number_diagnoses",
    "num_medications|numchange",
    "num_medications|time_in_hospital",
    "num_procedures",
    "number_diagnoses",
    "number_diagnoses|time_in_hospital",
    "number_emergency_log1p",
    "number_inpatient_log1p",
    "number_outpatient_log1p",
    "Other",
    "pioglitazone",
    "repaglinide",
    "rosiglitazone",
    "time_in_hospital",
    "time_in_hospital|num_lab_procedures",
    "tolazamide",
]

HIGH_MISSING_VALUES = [
    "weight",
    "payer_code",
    "medical_specialty",
]


class Featurize(Stage):
    """Pipeline stage for feature engineering with medical domain knowledge.

    This stage performs comprehensive feature transformations, including handling missing values,
    encoding categorical variables, creating interaction terms, and generating visualizations
    to prepare the data for machine learning models.
    """

    def run(self):
        """Generate and transform features with medical domain expertise.

        This method performs the following steps:
            1. Loads the preprocessed dataset.
            2. Defines output directories for processed data and plots.
            3. Drops columns with extensive missing values.
            4. Maps and transforms various features based on medical domain knowledge.
            5. Handles binary and multi-categorical feature encodings.
            6. Creates interaction terms between selected features.
            7. Validates the presence of target features.
            8. Saves the processed features and generates relevant visualizations.
            9. Logs metrics related to feature engineering.

        :raises RuntimeError: If essential columns are missing from the dataset.
        :raises ValueError: If target features are missing after processing.
        """
        # Load preprocessed data
        df = self.load_data("cleaned.parquet", "data/interim")
        logger.info(f"Initial shape: {df.shape}")
        logger.debug(f"Columns:\n{df.columns}")

        # Define paths for saving outputs and plots
        output_dir = pathlib.Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = pathlib.Path(self.cfg.paths.plots) / self.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Plots will be saved to: {plots_dir}")

        # Drop columns with extensive missing values
        df = df.drop(columns=[col for col in HIGH_MISSING_VALUES if col in df.columns])

        # Age processing
        df["age"] = df["age"].map(AGE_MAP)
        assert df["age"].between(5, 95).all(), "Age values are not correctly transformed to midpoints"
        logger.info("Age categories converted to midpoints")

        # Medication processing
        for med in MEDICATIONS:
            if med not in df.columns:
                raise RuntimeError(f"Medication column not found: {med}")
            df[med] = df[med].map(MEDICATION_STATUS_MAP)

        # Calculate medication change count and service utilization
        df["numchange"] = df[MEDICATIONS].apply(lambda row: sum(row == 1), axis=1)
        df["service_utilization"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]

        # Binary encodings
        for col, mapping in BINARY_MAP.items():
            if col not in df.columns:
                raise RuntimeError(f"Binary mapping column not found: {col}")
            df[col] = df[col].map(mapping)

        # Lab results
        for col, mapping in LAB_MAP.items():
            if col not in df.columns:
                raise RuntimeError(f"Lab mapping column not found: {col}")
            df[col] = df[col].map(mapping)

        # Admission type mapping
        if "admission_type_id" not in df.columns:
            raise RuntimeError("Column 'admission_type_id' not found")
        df["admission_type_id"] = df["admission_type_id"].replace(ADMISSION_TYPE_MAP)

        # Discharge and admission mappings
        if "discharge_disposition_id" not in df.columns:
            raise RuntimeError("Column 'discharge_disposition_id' not found")
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(DISCHARGE_MAP)

        if "admission_source_id" not in df.columns:
            raise RuntimeError("Column 'admission_source_id' not found")
        df["admission_source_id"] = df["admission_source_id"].replace(ADMISSION_MAP)

        # Diagnosis categorization
        for i in range(1, 4):
            diag_col = f"diag_{i}"
            level_col = f"level1_diag{i}"

            if diag_col not in df.columns:
                logger.warning(f"Diagnosis column '{diag_col}' not found")
                df[level_col] = 0
                continue

            # Convert diagnosis code and categorize
            df[diag_col] = df[diag_col].astype(str).apply(lambda x: "0" if any(c in x for c in ["V", "E"]) else x)
            df[diag_col] = pd.to_numeric(df[diag_col], errors="coerce")

            # Apply categorization functions
            df[level_col] = 0  # Default value
            for idx, (category, func) in enumerate(DIAGNOSIS_CATEGORIES.items(), start=1):
                df.loc[func(df[diag_col]), level_col] = idx

        # Log transformations for skewed features
        for col in SKEWED_FEATURES:
            if col in df.columns:
                if df[col].skew() > 2:
                    df[f"{col}_log1p"] = np.log1p(df[col])
                    logger.debug(f"Applied log1p transformation to '{col}'")
                else:
                    df[f"{col}_log1p"] = df[col]  # Keep original if not skewed
            else:
                logger.warning(f"Skewed feature '{col}' not found in DataFrame.")
                df[f"{col}_log1p"] = 0  # Default if missing

        # Binary encodings for race categories
        race_categories = ["Asian", "Hispanic", "Caucasian", "AfricanAmerican", "Other"]
        for race in race_categories:
            if race not in df.columns:
                if "race" in df.columns:
                    df[race] = (df["race"] == race).astype(int)
                else:
                    df[race] = 0  # Default if 'race' column is missing
                logger.debug(f"Created binary column for race: '{race}'")

        # Define expected categories for binary features
        expected_binary_categories = {
            "A1Cresult": ["0", "1"],
            "max_glu_serum": ["0", "1"],
            "gender": ["0", "1"],
        }

        # Define expected categories for multi-categorical features
        expected_multi_categories = {
            "admission_source_id": ["1", "2", "3", "4", "7", "9"],
            "admission_type_id": ["3", "5"],
            "discharge_disposition_id": ["2", "7", "10", "18"],
            "level1_diag1": [f"{i}.0" for i in range(1, 9)],
        }

        # One-hot encode binary features without dropping any categories
        for col, categories in expected_binary_categories.items():
            if col in df.columns:
                # Convert to string to ensure consistent naming
                df[col] = df[col].astype(str)
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                # Ensure all expected dummy columns are present
                for cat in categories:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in dummies.columns:
                        dummies[dummy_col] = 0  # Add missing dummy column with 0
                df = pd.concat([df, dummies], axis=1)
                logger.debug(f"One-hot encoded binary feature '{col}' with {len(dummies.columns)} dummy variables.")

        # One-hot encode multi-categorical features without dropping the first category
        for col, categories in expected_multi_categories.items():
            if col in df.columns:
                # Convert to string to ensure consistent naming
                df[col] = df[col].astype(str)
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                # Ensure all expected dummy columns are present
                for cat in categories:
                    dummy_col = f"{col}_{cat}"
                    if dummy_col not in dummies.columns:
                        dummies[dummy_col] = 0  # Add missing dummy column with 0
                df = pd.concat([df, dummies], axis=1)
                logger.debug(
                    f"One-hot encoded multi-categorical feature '{col}' with {len(dummies.columns)} dummy variables."
                )

        # Drop original categorical columns after encoding
        df = df.drop(columns=[col for col in CATEGORICAL_FEATURES if col in df.columns])

        # Create Interaction Terms
        for term1, term2 in INTERACTION_TERMS:
            if term1 in df.columns and term2 in df.columns:
                interaction_col = f"{term1}|{term2}"
                df[interaction_col] = df[term1] * df[term2]
                logger.debug(f"Created interaction term '{interaction_col}'")
            else:
                logger.warning(
                    f"Cannot create interaction term '{term1}|{term2}' because one or both features are missing."
                )

        # Final selection of columns - ensure 'readmitted' is present
        if "readmitted" not in df.columns:
            raise RuntimeError("Target column 'readmitted' not found in DataFrame.")

        # Select target features if they exist
        existing_target_features = [col for col in TARGET_FEATURE_SET if col in df.columns]
        df_final = df.loc[:, existing_target_features + ["readmitted"]].copy()

        logger.info("Checking for missing features in final dataset")
        missing_features = set(TARGET_FEATURE_SET) - set(df_final.columns)
        if missing_features:
            logger.warning(f"Missing features in final dataset: {missing_features}")
            # Investigate why these features are missing
            for feature in missing_features:
                if feature.startswith("A1Cresult_"):
                    # Possible issue with binary encoding
                    logger.error(f"Binary encoding for 'A1Cresult' might be incorrect. Missing feature: {feature}")
                elif feature.startswith("max_glu_serum_"):
                    # Possible issue with binary encoding
                    logger.error(f"Binary encoding for 'max_glu_serum' might be incorrect. Missing feature: {feature}")
                elif feature.startswith("level1_diag1_"):
                    # Possible issue with diagnosis categorization
                    logger.error(f"Diagnosis categorization might be incorrect. Missing feature: {feature}")
                elif feature.startswith("gender_"):
                    # Possible issue with binary encoding of gender
                    logger.error(f"Binary encoding for 'gender' might be incorrect. Missing feature: {feature}")
                else:
                    # Other features
                    logger.error(f"Feature '{feature}' is missing due to unexpected data or encoding issues.")
            # Raise error to halt pipeline and prompt investigation
            raise ValueError(f"Missing features in final dataset: {missing_features}")
        else:
            logger.info("All target features are present in the final dataset.")

        # Save processed features
        df_final.to_parquet(output_dir / "features.parquet", index=False)
        logger.info(f"Feature engineering completed. Final shape: {df_final.shape}")

        # Generate and save metrics
        metrics = {
            "total_features": len(df_final.columns),
            "feature_groups": {
                "medications": len(MEDICATIONS),
                "lab_results": len(LAB_MAP),
                "diagnoses": len(DIAGNOSIS_CATEGORIES),
                "interactions": len(INTERACTION_TERMS),
                "binary": len(BINARY_MAP),
                "categorical": len(CATEGORICAL_FEATURES),
            },
            "feature_names": list(df_final.columns),
        }
        self.save_metrics("metrics", metrics)

        # Filter NUMERIC_FEATURES and INTERACTION_TERMS based on availability in df_final
        numeric_features = [feature for feature in NUMERIC_FEATURES if feature in df_final.columns]
        interaction_terms_present = [
            (term1, term2) for term1, term2 in INTERACTION_TERMS if f"{term1}|{term2}" in df_final.columns
        ]

        # Generate statistics
        stats = {
            "readmission_rate": float(df_final["readmitted"].mean()),
            "total_patients": len(df_final),
            "readmitted_patients": int(df_final["readmitted"].sum()),
            "feature_stats": df_final[numeric_features].describe().to_dict(),
            "missing_values": df_final.isnull().sum().to_dict(),
        }
        self.save_metrics("statistics", stats)

        # Generate and save visualizations
        logger.info("Generating visualizations")

        # Correlation plot for numeric features in df_final
        if numeric_features:  # Check if there are any numeric features
            plt.figure(figsize=(12, 10))
            corr_matrix = df_final[numeric_features].corr()

            mask = np.triu(np.ones_like(corr_matrix), k=1)
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Correlation Coefficient"},
            )
            plt.title("Numeric Feature Correlations")
            plt.tight_layout()
            figure_filename = plots_dir / "feature_correlations.png"
            plt.savefig(str(figure_filename), bbox_inches="tight")
            plt.close()
            logger.debug(f"Saved correlation heatmap to: {figure_filename}")

        # Feature importance plot
        X = df_final.drop("readmitted", axis=1)
        y = df_final["readmitted"]
        X = X.fillna(0)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        importance_df = (
            pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
            .sort_values(by="importance", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(12, 10))
        sns.barplot(data=importance_df, y="feature", x="importance")
        plt.title("Top 20 Features by Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        figure_filename = plots_dir / "feature_importance.png"
        plt.savefig(str(figure_filename), bbox_inches="tight")
        logger.debug(f"Saved feature importance plot to: {figure_filename}")
        plt.close()
        csv_filename = plots_dir / "feature_importance.csv"
        importance_df.to_csv(str(csv_filename), index=False)
        logger.debug(f"Saved feature importance data to: {csv_filename}")

        # Distribution plots for numeric features
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_final, x=feature, kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution of {feature.replace('_', ' ').title()}")
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{feature}_distribution.png", bbox_inches="tight")
            plt.close()
            logger.debug(f"Saved distribution plot for {feature} to: {plots_dir / f'{feature}_distribution.png'}")

        # Distribution plots for interaction terms
        for term1, term2 in interaction_terms_present:
            feature = f"{term1}|{term2}"
            plt.figure(figsize=(10, 6))
            sns.histplot(df_final[feature], kde=True, bins=30, color="salmon")
            plt.title(f"Distribution of {term1.replace('_', ' ').title()} and {term2.replace('_', ' ').title()}")
            plt.xlabel(f"{term1.replace('_', ' ').title()} * {term2.replace('_', ' ').title()}")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plot_filepath = plots_dir / f"{feature.replace('|', '_')}_distribution.png"
            plt.savefig(str(plot_filepath), bbox_inches="tight")
            plt.close()
            logger.debug(f"Saved distribution plot for interaction term {feature} to: {plot_filepath}")

        # Readmission distribution plot
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_final, x="readmitted")
        plt.title("Distribution of Readmission")
        plt.xlabel("Readmitted")
        plt.ylabel("Count")
        plt.tight_layout()
        figure_filename = plots_dir / "readmission_distribution.png"
        plt.savefig(str(figure_filename), bbox_inches="tight")
        plt.close()
        logger.debug(f"Saved readmission distribution plot to: {figure_filename}")
        logger.info("All visualizations generated and saved successfully.")
