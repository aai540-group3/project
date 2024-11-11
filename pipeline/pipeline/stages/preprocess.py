import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from pipeline.stages.base import PipelineStage


class Preprocess(PipelineStage):
    """Pipeline stage for data preprocessing."""

    @logger.catch()
    def run(self):
        """Execute preprocessing pipeline."""

        DEMOGRAPHIC_FEATURES = [
            "race",
            "gender",
            "age",
        ]

        HIGH_MISSING_FEATURES = [
            "weight",
            "payer_code",
            "medical_specialty",
            "examide",
            "citoglipton",
        ]

        DIAGNOSIS_FEATURES = [
            "diag_1",
            "diag_2",
            "diag_3",
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
            "gender": {
                "Male": 1,
                "Female": 0,
            },
            "change": {
                "Ch": 1,
                "No": 0,
            },
            "diabetesMed": {
                "Yes": 1,
                "No": 0,
            },
        }

        LAB_MAP = {
            "A1Cresult": {
                ">8": 2,
                ">7": 1,
                "Norm": 0,
                "None": -99,
            },
            "max_glu_serum": {
                ">300": 2,
                ">200": 1,
                "Norm": 0,
                "None": -99,
            },
        }

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

        DISCHARGE_MAP = {
            6: 1, 8: 1, 9: 1, 13: 1,                       # HOME
            3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2,  # HEALTHCARE FACILITY
            12: 10, 15: 10, 16: 10, 17: 10,                # OUTPATIENT
            25: 18, 26: 18,                                # PSYCHIATRIC
        }  # fmt: off

        ADMISSION_MAP = {
            2: 1,  3: 1,                                   # PHYSICIAN REFERRAL
            5: 4,  6: 4,  10: 4, 22: 4, 25: 4,             # TRANSFER
            15: 9, 17: 9, 20: 9, 21: 9,                    # EMERGENCY
            13: 11, 14: 11,                                # OTHER
        }  # fmt: off

        # Define paths for saving plots
        plots_dir = pathlib.Path(self.cfg.paths.plots) / self.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Plots will be saved to: {plots_dir}")

        # Load data
        logger.info("Loading raw data...")
        df = pd.read_csv("data/raw/data.csv", low_memory=False)

        logger.info("Cleaning data...")
        logger.info("Dropping missing features with too many missing values")
        df = df.drop(columns=HIGH_MISSING_FEATURES, errors="ignore")

        # Replace missing values
        logger.info("Replacing missing values")
        df = df.replace("?", np.nan)

        # Apply filters
        logger.info("Applying filters")
        mask = (
            (df["gender"].isin(["Male", "Female"]))
            & (df["race"].notna() & (df["race"] != "?"))
            & (df["diag_1"].notna() & (df["diag_1"] != "?"))
            & (df["diag_2"].notna() & (df["diag_2"] != "?"))
            & (df["diag_3"].notna() & (df["diag_3"] != "?"))
            & (df["discharge_disposition_id"] != 11)
        )
        df = df[mask].copy()

        # Reset index and add ID
        logger.info("Resetting index and adding ID column")
        df = df.reset_index(drop=True)
        df["id"] = df.index

        # Apply mappings
        logger.info("Applying feature mappings")
        df["age"] = df["age"].map(AGE_MAP)

        for col, mapping in BINARY_MAP.items():
            if col not in df.columns:
                raise KeyError(f"Binary mapping column not found: {col}")
            df[col] = df[col].map(mapping)

        for col, mapping in LAB_MAP.items():
            if col not in df.columns:
                raise KeyError(f"Lab mapping column not found: {col}")
            df[col] = df[col].map(mapping)

        # Process medications
        logger.info("Processing medications")
        logger.trace(f"Columns found:\n {df.columns}")
        for med in MEDICATIONS:
            if med not in df.columns:
                raise KeyError(f"Medication column '{med}' not found")
            df[med] = df[med].map(lambda x: 2 if x == "Up" else (1 if x in ["Down", "Steady"] else 0))

        # Process admission and discharge codes
        logger.info("Processing admission and discharge codes")
        if "admission_type_id" not in df.columns:
            raise KeyError("Column 'admission_type_id' not found for mapping")
        df["admission_type_id"] = df["admission_type_id"].map(lambda x: 1 if x in [2, 7] else (5 if x in [6, 8] else x))

        if "discharge_disposition_id" not in df.columns:
            raise KeyError("Column 'discharge_disposition_id' not found for mapping")
        df["discharge_disposition_id"] = df["discharge_disposition_id"].map(DISCHARGE_MAP, na_action="ignore")

        if "admission_source_id" not in df.columns:
            raise KeyError("Column 'admission_source_id' not found for mapping")
        df["admission_source_id"] = df["admission_source_id"].map(ADMISSION_MAP, na_action="ignore")

        # PROCESS DIAGNOSES
        logger.info("Processing diagnoses")
        for i in range(1, 4):
            diag_col = f"diag_{i}"
            level_col = f"level1_diag{i}"

            if diag_col not in df.columns:
                raise KeyError(f"Diagnosis column '{diag_col}' not found")

            # CONVERT DIAGNOSIS CODES
            df[diag_col] = df[diag_col].astype(str)
            df[diag_col] = df[diag_col].apply(lambda x: "0" if any(code in str(x) for code in ["V", "E"]) else x)
            df[diag_col] = pd.to_numeric(df[diag_col], errors="coerce")

            # MAP TO DIAGNOSIS CATEGORIES
            conditions = [
                (((df[diag_col] >= 390) & (df[diag_col] < 460)) | (df[diag_col] == 785)),  # CIRCULATORY
                (((df[diag_col] >= 460) & (df[diag_col] < 520)) | (df[diag_col] == 786)),  # RESPIRATORY
                (((df[diag_col] >= 520) & (df[diag_col] < 580)) | (df[diag_col] == 787)),  # DIGESTIVE
                (df[diag_col] == 250),                                                     # DIABETES
                ((df[diag_col] >= 800) & (df[diag_col] < 1000)),                           # INJURY
                ((df[diag_col] >= 710) & (df[diag_col] < 740)),                            # MUSCULOSKELETAL
                (((df[diag_col] >= 580) & (df[diag_col] < 630)) | (df[diag_col] == 788)),  # GENITOURINARY
                ((df[diag_col] >= 140) & (df[diag_col] < 240)),                            # NEOPLASMS
            ]  # fmt: off
            choices = range(1, 9)
            df[level_col] = np.select(conditions, choices, default=0)

        # HANDLE OUTLIERS
        logger.info("Handling outliers")
        exclude_cols = ["readmitted", "id"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_min, initial_max = df[col].min(), df[col].max()
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            final_min, final_max = df[col].min(), df[col].max()
            logger.debug(f"Clipped '{col}' from ({initial_min}, {initial_max}) to ({final_min}, {final_max})")

        # PROCESS TARGET VARIABLE
        logger.info("Processing target variable")
        if "readmitted" not in df.columns:
            raise KeyError("Target column 'readmitted' not found")
        df["readmitted"] = df["readmitted"].map(lambda x: 1 if x == "<30" else 0)

        # VALIDATE TARGET VARIABLE
        logger.info("Validating target variable")
        if df["readmitted"].nunique() < 2:
            logger.error("Lost all positive cases during preprocessing!")
            raise ValueError("Lost all positive cases during preprocessing!")

        # SAVE PROCESSED DATA
        logger.info("Saving processed data")
        self.save_output(df, "data.parquet", "data/interim")

        # Calculate and save metrics
        logger.info("Calculating and saving metrics")
        metrics = {
            "dataset": {
                "initial_shape": df.shape,
                "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "missing_values": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict(),
            },
            "features": {
                "demographic": DEMOGRAPHIC_FEATURES,
                "medications": MEDICATIONS,
                "diagnosis": DIAGNOSIS_FEATURES,
            },
            "target": {"distribution": df["readmitted"].value_counts(normalize=True).to_dict()},
        }
        self.save_metrics("metrics", metrics)

        # GENERATE AND SAVE PLOTS
        logger.info("Generating and saving plots")

        # MISSING VALUES HEATMAP
        logger.info("Generating Missing Values Heatmap")
        plt.figure(figsize=(12, 8))
        missing_data = df.isnull().mean().to_frame("missing_rate")
        sns.heatmap(
            missing_data,
            cmap="YlOrRd",
            cbar=True,
            yticklabels=True,
        )
        plt.title("Distribution - Missing Values")
        plt.xlabel("Features")
        plt.ylabel("Missing Rate")
        plt.tight_layout()
        missing_plot_path = plots_dir / "missing_values.png"
        plt.savefig(missing_plot_path, bbox_inches="tight")
        logger.debug(f"Saved Missing Values Heatmap to: {missing_plot_path}")
        plt.close()

        # FEATURE DISTRIBUTIONS
        logger.info("Generating Feature Distributions Plots")
        NUMERICAL_FEATURES = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
        ]
        for feature in NUMERICAL_FEATURES:
            if feature not in df.columns:
                raise KeyError(f"Feature for distribution plot not found: {feature}")
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df, x=feature, kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution - {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plot_path = plots_dir / f"{feature}_distribution.png"
            plt.savefig(plot_path, bbox_inches="tight")
            logger.debug(f"Saved {feature} distribution plot to: {plot_path}")
            plt.close()

        # TARGET DISTRIBUTION
        logger.info("Generating Target Distribution Plot")
        if "readmitted" not in df.columns:
            raise KeyError("Target column 'readmitted' not found")
        plt.figure(figsize=(6, 6))
        sns.countplot(data=df, x="readmitted", hue="readmitted", legend=False)
        plt.title("Distribution - Readmission")
        plt.xlabel("Readmitted")
        plt.ylabel("Count")
        plt.xticks([0, 1], ["No", "Yes"])
        plt.tight_layout()
        target_plot_path = plots_dir / "readmission_distribution.png"
        plt.savefig(target_plot_path, bbox_inches="tight")
        logger.debug(f"Saved Readmission Distribution plot to: {target_plot_path}")
        plt.close()

        logger.info("Preprocessing completed successfully")
