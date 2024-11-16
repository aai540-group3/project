import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from .stage import Stage


class Preprocess(Stage):
    """Pipeline stage for data preprocessing."""

    def run(self):
        """Execute preprocessing pipeline to clean data without creating new features."""

        HIGH_MISSING_FEATURES = [
            "weight",       # 96.86% missing (98,569/101,766), only 9 unique values when present
            "examide",      # No variation: single value for all records (zero-variance predictor)
            "citoglipton",  # No variation: single value for all records (zero-variance predictor)
        ]  # fmt: off

        # Define paths for saving plots
        plots_dir = pathlib.Path(self.cfg.paths.plots) / self.name
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Plots will be saved to: {plots_dir}")

        # Load data
        logger.info("Loading raw data...")
        df = pd.read_csv("data/raw/data.csv", low_memory=False)

        # Capture initial metrics
        logger.info("Capturing initial data metrics")
        metrics = {
            "initial_shape": list(df.shape),
            "initial_memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": df.isnull().sum().to_dict(),
            "missing_values_rate": (df.isnull().mean() * 100).to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "column_names": list(df.columns),
            "five_number_summary": df.describe().to_dict(),
        }

        # Capture number of missing values for each high-missing column before dropping
        metrics["high_missing_features_dropped"] = {
            col: df[col].isnull().sum() for col in HIGH_MISSING_FEATURES if col in df.columns
        }

        # Handle medical specialty and payer code before other preprocessing
        logger.info("Processing medical specialty and payer code")

        # Analyze and visualize medical specialty distribution
        plt.figure(figsize=(12, 6))
        specialty_counts = df["medical_specialty"].value_counts().head(15)
        sns.barplot(x=specialty_counts.values, y=specialty_counts.index)
        plt.title("Top 15 Medical Specialties")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "medical_specialty_distribution.png", bbox_inches="tight")
        plt.close()

        # Fill missing values with meaningful categories
        df["medical_specialty"] = df["medical_specialty"].fillna("Unknown Specialty")
        df["payer_code"] = df["payer_code"].fillna("Unknown Payer")

        # Capture specialty and payer code distributions in metrics
        metrics.update(
            {
                "medical_specialty_distribution": df["medical_specialty"].value_counts().to_dict(),
                "payer_code_distribution": df["payer_code"].value_counts().to_dict(),
            }
        )

        # Drop columns with high missing values
        logger.info("Dropping columns with excessive missing values")
        df = df.drop(columns=HIGH_MISSING_FEATURES, errors="ignore")

        # Replace missing placeholders
        logger.info("Replacing missing placeholders")
        df = df.replace("?", np.nan)

        # Before filtering, capture diagnosis code distributions
        logger.info("Capturing diagnosis code distributions")

        for diag_col in ["diag_1", "diag_2", "diag_3"]:
            if diag_col in df.columns:
                # Get top 20 diagnoses and their stats
                diagnosis_stats = (
                    pd.DataFrame(
                        {
                            "diagnosis": df[diag_col],
                            "readmitted": df["readmitted"].map({"<30": 1, ">30": 0, "NO": 0}),
                        }
                    )
                    .groupby("diagnosis")
                    .agg({"readmitted": ["count", "mean"]})
                    .reset_index()
                )

                diagnosis_stats.columns = ["diagnosis", "count", "readmission_rate"]
                diagnosis_stats = diagnosis_stats.nlargest(20, "count").sort_values("count", ascending=True)

                # Create the visualization
                fig, ax = plt.subplots(figsize=(15, 8))

                # Create color map based on readmission rates
                norm = plt.Normalize(
                    diagnosis_stats["readmission_rate"].min(), diagnosis_stats["readmission_rate"].max()
                )
                cmap = plt.cm.RdYlBu
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])  # Only needed for older versions of Matplotlib

                # Apply the color map to the bars
                ax.barh(
                    diagnosis_stats["diagnosis"],
                    diagnosis_stats["count"],
                    color=cmap(norm(diagnosis_stats["readmission_rate"])),
                )

                # Configure axes
                ax.set_title(f"Top 20 {diag_col} Codes - Count and Readmission Rate")
                ax.set_xlabel("Number of Patients")
                ax.set_ylabel("Diagnosis Code")

                # Add colorbar associated with the current Axes
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label("Readmission Rate")

                # Adjust layout and save
                plt.tight_layout()
                plot_path = plots_dir / f"{diag_col}_distribution.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=300)
                plt.close()

        # Filter out unwanted data
        logger.info("Applying data filters")
        mask = (
            (df["gender"].isin(["Male", "Female"]))
            & (df["race"].notna() & (df["race"] != "?"))
            & (df["diag_1"].notna() & (df["diag_1"] != "?"))
            & (df["diag_2"].notna() & (df["diag_2"] != "?"))
            & (df["diag_3"].notna() & (df["diag_3"] != "?"))
            & (df["discharge_disposition_id"] != 11)  # Exclude expired patients
        )
        df = df[mask].copy()

        # Reset index and add unique ID if not present
        logger.info("Resetting index and adding unique 'id' column")
        df = df.reset_index(drop=True)
        df["id"] = df.index

        # Handle outliers by clipping
        logger.info("Clipping outliers for numeric features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["id", "readmitted"])
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Process target variable
        logger.info("Processing target variable")
        if "readmitted" not in df.columns:
            raise KeyError("Target column 'readmitted' not found")
        df["readmitted"] = df["readmitted"].map(lambda x: 1 if x == "<30" else 0)

        # Validate target variable to ensure no data loss
        if df["readmitted"].nunique() < 2:
            logger.error("All positive cases lost during preprocessing!")
            raise ValueError("All positive cases lost during preprocessing.")

        # Capture target balance for metrics
        target_balance = df["readmitted"].value_counts(normalize=True).to_dict()
        metrics["target_balance"] = target_balance

        # Update metrics after processing
        logger.info("Capturing post-cleaning metrics")
        metrics.update(
            {
                "final_shape": list(df.shape),
                "final_memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "post_cleaning_missing_values": df.isnull().sum().to_dict(),
                "post_cleaning_missing_rate": (df.isnull().mean() * 100).to_dict(),
                "final_dtypes": df.dtypes.astype(str).to_dict(),
                "target_distribution": df["readmitted"].value_counts(normalize=True).to_dict(),
            }
        )

        # Save consolidated metrics
        self.save_metrics("metrics", metrics)

        # Save cleaned data
        logger.info("Saving cleaned data")
        self.save_output(df, "cleaned.parquet", "data/interim")

        # Generate and save missing values plot
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

        # Generate feature distribution plots
        logger.info("Generating Feature Distribution Plots")
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

        # Generate target distribution plot
        logger.info("Generating Target Distribution Plot")
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
