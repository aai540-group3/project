import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from .stage import Stage


class Featurize(Stage):
    """Pipeline stage for feature engineering with medical domain knowledge."""

    def run(self):
        """Generate and transform features with medical domain expertise."""

        # Load preprocessed data
        df = self.load_data("data.parquet", "data/interim")
        logger.info(f"Initial shape: {df.shape}")

        # Define paths for saving plots
        plots_dir = pathlib.Path(self.cfg.paths.plots) / self.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Plots will be saved to: {plots_dir}")

        # MEDICATIONS LIST
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

        # AGE PROCESSING
        # Convert age categories to meaningful numeric values
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
        logger.info("Age categories converted to midpoints")

        # MEDICATION PROCESSING
        med_mapping = {"No": 0, "Steady": 1, "Up": 1, "Down": 1}
        change_mapping = {"No": 0, "Steady": 0, "Up": 1, "Down": 1}

        for med in MEDICATIONS:
            if med not in df.columns:
                raise RuntimeError(f"Medication column not found: {med}")
            df[f"{med}_med"] = df[med].map(med_mapping)
            df[f"{med}_change"] = df[med].map(change_mapping)
            df[med] = df[med].map(med_mapping)

        # CALCULATE MEDICATION COUNTS
        df["num_med"] = df[[f"{med}_med" for med in MEDICATIONS]].sum(axis=1)
        df["num_change"] = df[[f"{med}_change" for med in MEDICATIONS]].sum(axis=1)

        # CLEANUP TEMPORARY COLUMNS
        for med in MEDICATIONS:
            df = df.drop([f"{med}_med", f"{med}_change"], axis=1)

        logger.info("Medication features processed")

        # SERVICE UTILIZATION
        df["service_utilization"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]

        # BINARY ENCODINGS
        binary_mappings = {
            "change": {"Ch": 1, "No": 0},
            "diabetesMed": {"Yes": 1, "No": 0},
            "gender": {"Male": 1, "Female": 0},
        }

        for col, mapping in binary_mappings.items():
            if col not in df.columns:
                raise RuntimeError(f"Binary mapping column not found: {col}")
            df[col] = df[col].map(mapping)

        # LAB RESULTS
        lab_mappings = {
            "A1Cresult": {">8": 1, ">7": 1, "Norm": 0, "None": -99},
            "max_glu_serum": {">300": 1, ">200": 1, "Norm": 0, "None": -99},
        }

        for col, mapping in lab_mappings.items():
            if col not in df.columns:
                raise RuntimeError(f"Lab mapping column not found: {col}")
            df[col] = df[col].map(mapping)

        # ADMISSION TYPE MAPPING
        admission_type_map = {
            2: 1, 7: 1,  # EMERGENCY/URGENT
            6: 5, 8: 5,  # OTHER
        }  # fmt: off
        if "admission_type_id" not in df.columns:
            raise RuntimeError("Column 'admission_type_id' not found")
        df["admission_type_id"] = df["admission_type_id"].replace(admission_type_map)

        # DISCHARGE DISPOSITION MAPPING
        discharge_map = {
            6: 1, 8: 1, 9: 1, 13: 1,                       # HOME
            3: 2, 4: 2, 5: 2, 14: 2, 22: 2, 23: 2, 24: 2,  # HEALTHCARE
            12: 10, 15: 10, 16: 10, 17: 10,                # OUTPATIENT
            25: 18, 26: 18,                                # PSYCHIATRIC
        }  # fmt: off
        if "discharge_disposition_id" not in df.columns:
            raise RuntimeError("Column 'discharge_disposition_id' not found")
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(discharge_map)

        # ADMISSION SOURCE MAPPING
        admission_source_map = {
            2: 1, 3: 1,                       # PHYSICIAN REFERRAL
            5: 4, 6: 4, 10: 4, 22: 4, 25: 4,  # TRANSFER
            15: 9, 17: 9, 20: 9, 21: 9,       # EMERGENCY
            13: 11, 14: 11,                   # OTHER
        }  # fmt: off
        if "admission_source_id" not in df.columns:
            raise RuntimeError("Column 'admission_source_id' not found")
        df["admission_source_id"] = df["admission_source_id"].replace(admission_source_map)

        # DIAGNOSIS CATEGORIZATION
        diagnosis_categories = {
            "CIRCULATORY": lambda x: ((x >= 390) & (x < 460)) | (x == 785),
            "RESPIRATORY": lambda x: ((x >= 460) & (x < 520)) | (x == 786),
            "DIGESTIVE": lambda x: ((x >= 520) & (x < 580)) | (x == 787),
            "DIABETES": lambda x: (x == 250),
            "INJURY": lambda x: (x >= 800) & (x < 1000),
            "MUSCULOSKELETAL": lambda x: (x >= 710) & (x < 740),
            "GENITOURINARY": lambda x: ((x >= 580) & (x < 630)) | (x == 788),
            "NEOPLASMS": lambda x: (x >= 140) & (x < 240),
        }

        for i in range(1, 4):
            diag_col = f"diag_{i}"
            level_col = f"level1_diag{i}"

            if diag_col not in df.columns:
                logger.warning(f"Diagnosis column '{diag_col}' not found")
                df[level_col] = 0
                continue

            # Convert diagnosis codes
            df[diag_col] = df[diag_col].astype(str)
            df[diag_col] = df[diag_col].apply(lambda x: "0" if any(c in str(x) for c in ["V", "E"]) else x)
            df[diag_col] = pd.to_numeric(df[diag_col], errors="coerce")

            # Create diagnosis categories
            conditions = [v(df[diag_col]) for v in diagnosis_categories.values()]
            choices = range(1, len(diagnosis_categories) + 1)
            df[level_col] = np.select(conditions, choices, default=0)

        logger.info("Diagnosis categorization completed")

        # HANDLE SKEWED FEATURES
        skewed_features = ["number_outpatient", "number_inpatient", "number_emergency"]

        for col in skewed_features:
            if col not in df.columns:
                raise RuntimeError(f"Skewed feature column not found: {col}")
            if (df[col] >= 0).all():
                df[f"{col}_log1p"] = np.log1p(df[col])
            elif (df[col] > 0).all():
                df[f"{col}_log"] = np.log(df[col])
            else:
                raise RuntimeError(f"Skewed feature contains invalid values: {col}")

        # INTERACTION TERMS
        interactions = [
            ("num_med", "time_in_hospital"),
            ("num_med", "num_procedures"),
            ("time_in_hospital", "num_lab_procedures"),
            ("num_med", "num_lab_procedures"),
            ("num_med", "number_diagnoses"),
            ("age", "number_diagnoses"),
            ("change", "num_med"),
            ("number_diagnoses", "time_in_hospital"),
            ("num_med", "num_change"),
        ]

        for term1, term2 in interactions:
            if term1 not in df.columns or term2 not in df.columns:
                raise RuntimeError(f"Interaction terms '{term1}' and/or '{term2}' not found")
            df[f"{term1}|{term2}"] = df[term1] * df[term2]

        logger.info("Interaction terms created")

        # ONE-HOT ENCODING
        categorical_features = [
            "race",
            "gender",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "max_glu_serum",
            "A1Cresult",
            "level1_diag1",
        ]

        # Save pre-encoding data
        output_pre_onehot = pathlib.Path("data/processed/features_not_onehot.parquet")
        df.to_parquet(output_pre_onehot, index=False)
        logger.debug(f"Saved pre-encoding features to: {output_pre_onehot}")

        # PERFORM ONE-HOT ENCODING
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # SELECT FINAL FEATURES
        feature_set = [
            # NUMERIC FEATURES
            "age",
            "num_change",
            "num_med",
            "num_procedures",
            "number_diagnoses",
            "number_emergency_log1p",
            "number_inpatient_log1p",
            "number_outpatient_log1p",
            "service_utilization",
            "time_in_hospital",
            # MEDICATIONS
            *MEDICATIONS,
            # ENCODED CATEGORICAL FEATURES
            *[c for c in df.columns if c.startswith("race_")],
            *[c for c in df.columns if c.startswith("gender_")],
            *[c for c in df.columns if c.startswith("admission_type_id_")],
            *[c for c in df.columns if c.startswith("discharge_disposition_id_")],
            *[c for c in df.columns if c.startswith("admission_source_id_")],
            *[c for c in df.columns if c.startswith("max_glu_serum_")],
            *[c for c in df.columns if c.startswith("A1Cresult_")],
            *[c for c in df.columns if c.startswith("level1_diag1_")],
            # INTERACTION TERMS
            *[c for c in df.columns if "|" in c],
        ]

        # VERIFY FEATURES EXIST
        missing_features = [feat for feat in feature_set if feat not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")

        # CREATE FINAL DATASET
        df_final = df[feature_set + ["readmitted"]].copy()

        # SAVE PROCESSED FEATURES
        output_features = pathlib.Path("data/processed/features.parquet")
        df_final.to_parquet(output_features, index=False)
        logger.debug(f"Saved final features to: {output_features}")

        # SAVE METRICS
        metrics = {
            "total_features": len(df_final.columns),
            "feature_groups": {
                "medications": len(MEDICATIONS),
                "lab_results": len(lab_mappings),
                "diagnoses": len(diagnosis_categories),
                "interactions": len(interactions),
                "binary": len(binary_mappings),
                "categorical": len(categorical_features),
            },
            "feature_names": list(df_final.columns),
        }
        self.save_metrics("metrics", metrics)

        # GENERATE VISUALIZATIONS
        logger.info("Generating visualizations")

        # FEATURE CORRELATION HEATMAP
        plt.figure(figsize=(12, 10))
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="RdBu",
            center=0,
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            square=True,
            cbar_kws={"shrink": 0.5},
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_correlations.png", bbox_inches="tight")
        plt.close()

        # FEATURE IMPORTANCE
        X = df.drop("readmitted", axis=1)
        y = df["readmitted"]
        X = X.fillna(0)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        importance_df = (
            pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
            .sort_values(by="importance", ascending=False)
            .head(20)
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y="feature", x="importance")
        plt.title("Top 20 Features by Importance")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", bbox_inches="tight")
        plt.close()

        # SAVE FEATURE IMPORTANCE DATA
        importance_df.to_csv(plots_dir / "feature_importance.csv", index=False)

        # DISTRIBUTION PLOTS
        numeric_features = [
            "age",
            "time_in_hospital",
            "num_med",
            "service_utilization",
            "num_procedures",
            "num_change",
        ]

        for feature in numeric_features:
            if feature not in df_final.columns:
                raise RuntimeError(f"Feature for distribution plot not found: {feature}")

            logger.info(f"Generating distribution plot for: {feature}")
            plt.figure(figsize=(10, 6))
            sns.histplot(df_final, x=feature, kde=True, bins=30, color="skyblue")
            plt.title(f"Distribution of {feature.replace('_', ' ').title()}")
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{feature}_distribution.png", bbox_inches="tight")
            plt.close()

        # DISTRIBUTION PLOTS FOR INTERACTION TERMS
        for term1, term2 in interactions:
            feature = f"{term1}|{term2}"
            filename_feature = feature.replace("|", "_")
            if feature not in df_final.columns:
                logger.warning(f"Interaction feature not found: {feature}")
                continue

            logger.info(f"Generating distribution plot for interaction: {feature}")
            plt.figure(figsize=(10, 6))
            sns.histplot(df_final[feature], kde=True, bins=30, color="salmon")
            plt.title(f"Distribution of {term1.title()} and {term2.title()}")
            plt.xlabel(f"{term1.replace('_', ' ').title()} and {term2.replace('_', ' ').title()}")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{filename_feature}_distribution.png", bbox_inches="tight")
            plt.close()

        # READMISSION DISTRIBUTION
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_final, x="readmitted")
        plt.title("Distribution of Readmission")
        plt.xlabel("Readmitted")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "readmission_distribution.png", bbox_inches="tight")
        plt.close()

        # SAVE ADDITIONAL STATISTICS
        stats = {
            "readmission_rate": float(df_final["readmitted"].mean()),
            "total_patients": len(df_final),
            "readmitted_patients": int(df_final["readmitted"].sum()),
            "feature_stats": df_final[numeric_features].describe().to_dict(),
            "missing_values": df_final.isnull().sum().to_dict(),
        }

        # SAVE TO METRICS
        self.save_metrics("statistics", stats)

        logger.info(f"Feature engineering completed. Final shape: {df_final.shape}")
        logger.info(f"Total features generated: {len(feature_set)}")
