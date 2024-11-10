import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from pipeline.stages.base import PipelineStage


class Featurize(PipelineStage):
    """Pipeline stage for feature engineering with medical domain knowledge."""

    def run(self):
        """Generate and transform features with medical domain expertise."""

        # Load preprocessed data
        df = self.load_data("data_cleaned.parquet", "data/interim")
        logger.info(f"Initial shape: {df.shape}")

        # Convert age categories to midpoints
        age_dict = {
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
        df["age"] = df["age"].map(age_dict)

        # Service Utilization Features
        df["service_utilization"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]

        # Medication Processing
        medications = [
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

        # Convert medication changes to numeric
        for med in medications:
            df[med] = df[med].map({"No": 0, "Steady": 1, "Up": 1, "Down": 1})

        # Count total medication changes
        df["numchange"] = df[medications].sum(axis=1)
        df["nummed"] = df[medications].sum(axis=1)

        # Binary Feature Encoding
        binary_mappings = {
            "change": {"Ch": 1, "No": 0},
            "diabetesmed": {"Yes": 1, "No": 0},
            "gender": {"Male": 1, "Female": 0},
        }

        for col, mapping in binary_mappings.items():
            df[col] = df[col].map(mapping)

        # Lab Results Processing
        lab_mappings = {
            "a1cresult": {">8": 1, ">7": 1, "Norm": 0, "None": -99},
            "max_glu_serum": {">300": 1, ">200": 1, "Norm": 0, "None": -99},
        }

        for col, mapping in lab_mappings.items():
            df[col] = df[col].map(mapping)

        # Admission Type
        admission_type_map = {
            2: 1, 7: 1,  # EMERGENCY/URGENT
            6: 5, 8: 5   # OTHER
        }  # fmt: skip
        df["admission_type_id"] = df["admission_type_id"].replace(admission_type_map)

        # Discharge Disposition
        discharge_map = {
            6: 1, 8: 1, 9: 1, 13: 1,                       # HOME
            3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2,  # HEALTHCARE
            12: 10, 15: 10, 16: 10, 17: 10,                # OUTPATIENT
            25: 18, 26: 18                                 # PSYCHIATRIC
        }  # fmt: skip
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(discharge_map)

        # Admission Source
        admission_source_map = {
            2: 1, 3: 1,                       # Physician Referral
            5: 4, 6: 4, 10: 4, 22: 4, 25: 4,  # Transfer
            15: 9, 17: 9, 20: 9, 21: 9,       # Emergency
            13: 11, 14: 11                    # Other
        }  # fmt: skip
        df["admission_source_id"] = df["admission_source_id"].replace(admission_source_map)

        # Diagnosis Processing
        for i in range(1, 4):
            col = f"diag_{i}"
            level_col = f"level1_diag{i}"

            # Convert diagnosis codes
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: "0" if any(c in str(x) for c in ["V", "E"]) else x)
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Map to diagnosis categories
            conditions = [
                ((df[col] >= 390) & (df[col] < 460)) | (df[col] == 785),  # CIRCULATORY
                ((df[col] >= 460) & (df[col] < 520)) | (df[col] == 786),  # RESPIRATORY
                ((df[col] >= 520) & (df[col] < 580)) | (df[col] == 787),  # DIGESTIVE
                (df[col] == 250),                                         # DIABETES
                ((df[col] >= 800) & (df[col] < 1000)),                    # INJURY
                ((df[col] >= 710) & (df[col] < 740)),                     # MUSCULOSKELETAL
                ((df[col] >= 580) & (df[col] < 630)) | (df[col] == 788),  # GENITOURINARY
                ((df[col] >= 140) & (df[col] < 240)),                     # NEOPLASMS
            ]  # fmt: skip
            choices = range(1, 9)
            df[level_col] = np.select(conditions, choices, default=0)

        # Create Interaction Terms
        interactions = [
            ("num_medications", "time_in_hospital"),
            ("num_medications", "num_procedures"),
            ("time_in_hospital", "num_lab_procedures"),
            ("num_medications", "num_lab_procedures"),
            ("num_medications", "number_diagnoses"),
            ("age", "number_diagnoses"),
            ("change", "num_medications"),
            ("number_diagnoses", "time_in_hospital"),
            ("num_medications", "numchange"),
        ]

        for term1, term2 in interactions:
            df[f"{term1}|{term2}"] = df[term1] * df[term2]

        # Handle Skewed Features
        skewed_features = ["number_outpatient", "number_inpatient", "number_emergency"]

        for col in skewed_features:
            if (df[col] >= 0).all():
                df[f"{col}_log1p"] = np.log1p(df[col])
            elif (df[col] > 0).all():
                df[f"{col}_log"] = np.log(df[col])

        # Create Dummy Variables
        categorical_features = [
            "race",
            "gender",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "max_glu_serum",
            "a1cresult",
            "level1_diag1",
        ]

        # Save data before one-hot encoding for Feast
        df.to_parquet("data/processed/features_not_onehot.parquet", index=False)

        # Proceed with one-hot encoding
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # Feature Selection
        feature_set = [
            "age",
            "time_in_hospital",
            "num_procedures",
            "num_medications",
            "number_outpatient_log1p",
            "number_emergency_log1p",
            "number_inpatient_log1p",
            "number_diagnoses",
            *medications,
            *[c for c in df.columns if c.startswith("race_")],
            *[c for c in df.columns if c.startswith("gender_")],
            *[c for c in df.columns if c.startswith("admission_type_id_")],
            *[c for c in df.columns if c.startswith("discharge_disposition_id_")],
            *[c for c in df.columns if c.startswith("admission_source_id_")],
            *[c for c in df.columns if c.startswith("max_glu_serum_")],
            *[c for c in df.columns if c.startswith("a1cresult_")],
            *[c for c in df.columns if c.startswith("level1_diag1_")],
            *[c for c in df.columns if "|" in c],
        ]

        # Select final feature set
        df_final = df[feature_set + ["readmitted"]]

        # SAVE OUTPUTS
        self.save_output(df_final, "features.parquet", "data/processed")

        # SAVE FEATURE METRICS
        metrics = {
            "total_features": len(df_final.columns),
            "feature_groups": {
                "medications": len(medications),
                "lab_results": len(lab_mappings),
                "diagnoses": len(conditions),
                "interactions": len(interactions),
                "binary": len(binary_mappings),
                "categorical": len(categorical_features),
            },
            "feature_names": list(df_final.columns),
        }
        self.save_metrics("metrics", metrics)

        # Generate visualizations
        self.save_plot(
            "feature_correlations",
            sns.heatmap,
            data=df_final.select_dtypes(include=[np.number]).corr(),
            cmap="RdBu",
            center=0,
            annot=False,
            mask=np.triu(np.ones_like(df_final.select_dtypes(include=[np.number]).corr()), k=1),
            figsize=(12, 8),
            title="Feature Correlations",
        )

        logger.info(f"Feature engineering completed. Final shape: {df_final.shape}")
