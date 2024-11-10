import numpy as np
import pandas as pd
from loguru import logger

from pipeline.stages.base import PipelineStage


class Preprocess(PipelineStage):
    """Pipeline stage for data preprocessing."""

    def run(self):
        """Execute preprocessing pipeline."""

        # Define feature groups
        DEMOGRAPHIC_FEATURES = ["race", "gender", "age"]
        ID_COLUMNS = ["encounter_id", "patient_nbr"]
        HIGH_MISSING_FEATURES = [
            "weight",
            "payer_code",
            "medical_specialty",
            "examide",
            "citoglipton",
        ]
        DIAGNOSIS_FEATURES = ["diag_1", "diag_2", "diag_3"]

        # Define mappings
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

        binary_map = {
            "gender": {"Male": 1, "Female": 0},
            "change": {"Ch": 1, "No": 0},
            "diabetesmed": {"Yes": 1, "No": 0},
        }

        lab_map = {
            "a1cresult": {">8": 2, ">7": 1, "Norm": 0, "None": -99},
            "max_glu_serum": {">300": 2, ">200": 1, "Norm": 0, "None": -99},
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

        discharge_map = {
            6: 1,
            8: 1,
            9: 1,
            13: 1,  # HOME
            3: 2,
            4: 2,
            5: 2,
            14: 2,
            22: 2,
            23: 2,
            24: 2,  # HEALTHCARE FACILITY
            12: 10,
            15: 10,
            16: 10,
            17: 10,  # OUTPATIENT
            25: 18,
            26: 18,  # PSYCHIATRIC
        }

        admission_map = {
            2: 1,
            3: 1,  # PHYSICIAN REFERRAL
            5: 4,
            6: 4,
            10: 4,
            22: 4,
            25: 4,  # TRANSFER
            15: 9,
            17: 9,
            20: 9,
            21: 9,  # EMERGENCY
            13: 11,
            14: 11,  # OTHER
        }

        # Load data
        logger.info("Loading and cleaning data")
        df = pd.read_csv("data/raw/data.csv", low_memory=False)

        # Clean column names
        df.columns = df.columns.str.lower().str.replace("[^a-zA-Z0-9_]", "_").str.strip("_")

        # Initial cleaning and filtering
        logger.info("Performing initial cleaning")
        # Drop high missing features
        df = df.drop(columns=HIGH_MISSING_FEATURES)

        # Replace missing values
        df = df.replace("?", np.nan)

        # Apply filters
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
        df = df.reset_index(drop=True)
        df["id"] = df.index

        # Apply mappings
        logger.info("Applying feature mappings")
        df["age"] = df["age"].map(age_map)

        for col, mapping in binary_map.items():
            df[col] = df[col].map(mapping)

        for col, mapping in lab_map.items():
            df[col] = df[col].map(mapping)

        # Process medications
        logger.info("Processing medications")
        for med in MEDICATIONS:
            if med in df.columns:
                df[med] = df[med].map(lambda x: 2 if x == "Up" else (1 if x in ["Down", "Steady"] else 0))
            else:
                logger.warning(f"Medication column not found: {med}")

        # Process admission and discharge codes
        logger.info("Processing admission and discharge codes")
        df["admission_type_id"] = df["admission_type_id"].map(lambda x: 1 if x in [2, 7] else (5 if x in [6, 8] else x))
        df["discharge_disposition_id"] = df["discharge_disposition_id"].map(discharge_map, na_action="ignore")
        df["admission_source_id"] = df["admission_source_id"].map(admission_map, na_action="ignore")

        # Process diagnoses
        logger.info("Processing diagnoses")
        for i in range(1, 4):
            diag_col = f"diag_{i}"
            level_col = f"level1_diag{i}"

            # Convert diagnosis codes
            df[diag_col] = df[diag_col].astype(str)
            df[diag_col] = df[diag_col].apply(lambda x: "0" if any(code in str(x) for code in ["V", "E"]) else x)
            df[diag_col] = pd.to_numeric(df[diag_col], errors="coerce")

            # Map to diagnosis categories
            conditions = [
                ((df[diag_col] >= 390) & (df[diag_col] < 460)) | (df[diag_col] == 785),  # CIRCULATORY
                ((df[diag_col] >= 460) & (df[diag_col] < 520)) | (df[diag_col] == 786),  # RESPIRATORY
                ((df[diag_col] >= 520) & (df[diag_col] < 580)) | (df[diag_col] == 787),  # DIGESTIVE
                (df[diag_col] == 250),                                                   # DIABETES
                ((df[diag_col] >= 800) & (df[diag_col] < 1000)),                         # INJURY
                ((df[diag_col] >= 710) & (df[diag_col] < 740)),                          # MUSCULOSKELETAL
                ((df[diag_col] >= 580) & (df[diag_col] < 630)) | (df[diag_col] == 788),  # GENITOURINARY
                ((df[diag_col] >= 140) & (df[diag_col] < 240)),                          # NEOPLASMS
            ]  # fmt: skip
            choices = range(1, 9)
            df[level_col] = np.select(conditions, choices, default=0)

        # Handle duplicates
        logger.info("Handling duplicates")
        for id_col in ID_COLUMNS:
            if id_col in df.columns:
                df = df.drop_duplicates(subset=[id_col], keep="first")
                break

        # Handle outliers
        logger.info("Handling outliers")
        exclude_cols = ["readmitted", "id"] + ID_COLUMNS
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

        # Process target variable
        logger.info("Processing target variable")
        df["readmitted"] = df["readmitted"].map(lambda x: 1 if x == "<30" else 0)

        if len(df["readmitted"].unique()) < 2:
            raise ValueError("Lost all positive cases during preprocessing!")

        # Save processed data
        logger.info("Saving processed data")
        self.save_output(df, "data_cleaned.parquet", "data/interim")

        # Calculate and save metrics
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
                "ids": ID_COLUMNS,
            },
            "target": {"distribution": df["readmitted"].value_counts(normalize=True).to_dict()},
            "processing_steps": [
                "Removed high-missing-value features",
                "Standardized demographic data",
                "Processed medication changes",
                "Mapped admission/discharge codes",
                "Categorized diagnoses by ICD9 ranges",
                "Handled outliers",
                "Converted readmission to binary outcome",
            ],
        }
        self.save_metrics("metrics", metrics)

        # Generate plots
        self._generate_plots(
            df,
            key_features=[
                "time_in_hospital",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_diagnoses",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
            ],
        )

    def _generate_plots(self, df: pd.DataFrame, key_features: list):
        """Generate and save visualization plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Missing values heatmap
        missing_data = df.isnull().mean().to_frame("missing_rate")
        self.save_plot(
            "missing_values",
            lambda data: sns.heatmap(data=data, cmap="YlOrRd", yticklabels=data.index),
            data=missing_data,
        )

        # Feature distributions
        def plot_distributions(data):
            fig, axes = plt.subplots(4, 2, figsize=(15, 20))
            axes = axes.flatten()

            for ax, feature in zip(axes, key_features):
                sns.histplot(data=data, x=feature, ax=ax)
                ax.set_title(feature)

            plt.tight_layout()
            return fig

        self.save_plot("feature_distributions", plot_distributions, data=df)

        # Target distribution
        self.save_plot("target_distribution", lambda data: sns.countplot(data=data, x="readmitted"), data=df)
