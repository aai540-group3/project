import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier

from .stage import Stage


class Explore(Stage):
    """Pipeline stage for data exploration and analysis."""

    def run(self):
        """Execute exploration analysis."""

        try:
            # Load data
            df = self.load_data("cleaned.parquet", "data/interim")
            logger.info(f"Dataset loaded with shape: {df.shape}")

            # Define feature groups
            FEATURE_GROUPS = {
                "demographic": [
                    "race",
                    "gender",
                    "age",
                ],
                "clinical": [
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_diagnoses",
                ],
                "service": [
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                ],
                "labs": ["max_glu_serum", "A1Cresult"],
                "medications": [
                    col for col in df.columns if any(med in col for med in ["metformin", "insulin", "glyburide"])
                ],
                "diagnosis": [col for col in df.columns if "diag" in col],
            }  # fmt: off

            # Define paths for saving plots
            plots_dir = pathlib.Path(self.cfg.paths.plots) / self.name
            plots_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Plots will be saved to: {plots_dir}")

            # Calculate statistics
            analysis = {
                "dataset": {
                    "shape": df.shape,
                    "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
                    "missing": df.isnull().sum().to_dict(),
                }
            }

            # Group-specific analysis
            group_analysis = {}
            for group, cols in FEATURE_GROUPS.items():
                logger.info(f"Analyzing feature group: {group}")
                # Split features and type
                numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df[cols].select_dtypes(exclude=[np.number]).columns.tolist()

                group_stats = {}

                # Handle numeric features
                if numeric_cols:
                    group_stats["numeric_stats"] = df[numeric_cols].describe().to_dict()
                    group_stats["numeric_correlations"] = df[numeric_cols].corr().to_dict()
                    group_stats["readmission_means"] = df.groupby("readmitted")[numeric_cols].mean().to_dict()

                    # Generate Numeric Statistics Plot
                    logger.info(f"Generating numeric statistics plots for group: {group}")
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
                    plt.title(f"{group.capitalize()} - Correlation Heatmap")
                    plot_path = plots_dir / f"{group}_numeric_correlation_heatmap.png"
                    plt.savefig(plot_path, bbox_inches="tight")
                    logger.debug(f"Saved plot: {plot_path}")
                    plt.close()

                # Handle categorical features
                if categorical_cols:
                    group_stats["categorical_stats"] = {
                        col: df[col].value_counts(normalize=True).to_dict() for col in categorical_cols
                    }
                    group_stats["categorical_counts"] = {
                        col: df.groupby("readmitted")[col].value_counts().reset_index(name="count").to_dict("records")
                        for col in categorical_cols
                    }

                    # Generate Categorical Distribution Plots
                    for col in categorical_cols:
                        logger.info(f"Generating categorical distribution plot for column: {col}")
                        plt.figure(figsize=(8, 6))
                        sns.countplot(data=df, x=col, hue="readmitted")
                        plt.title(f"{col} Distribution and Readmission Status")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plot_path = plots_dir / f"{col}_distribution.png"
                        plt.savefig(plot_path, bbox_inches="tight")
                        logger.debug(f"Saved plot: {plot_path}")
                        plt.close()

                # Add missing rates for all columns
                group_stats["missing_rates"] = df[cols].isnull().mean().to_dict()
                group_analysis[group] = group_stats

            analysis["groups"] = group_analysis

            # Categorical associations (chi-square)
            logger.info("Performing chi-square tests on categorical variables")
            analysis["categorical"] = {}
            for col in df.select_dtypes(exclude=[np.number]).columns:
                logger.debug(f"Analyzing categorical column: {col}")
                # Ensure that there are at least two unique values to perform chi-square test
                if df[col].nunique() > 1:
                    contingency_table = pd.crosstab(df[col], df["readmitted"])
                    chi2, p, dof, ex = chi2_contingency(contingency_table)
                    analysis["categorical"][col] = {
                        "distribution": df[col].value_counts(normalize=True).to_dict(),
                        "readmission_rate": df.groupby(col)["readmitted"].mean().to_dict(),
                        "chi_square": {"chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof},
                    }

                    # Generate Chi-Square Test Plot
                    plt.figure(figsize=(8, 6))
                    sns.countplot(data=df, x=col, hue="readmitted")
                    plt.title(f"Chi-Square Test for {col}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plot_path = plots_dir / f"{col}_chi_square.png"
                    plt.savefig(plot_path, bbox_inches="tight")
                    logger.debug(f"Saved plot: {plot_path}")
                    plt.close()
                else:
                    analysis["categorical"][col] = {
                        "distribution": df[col].value_counts(normalize=True).to_dict(),
                        "readmission_rate": df.groupby(col)["readmitted"].mean().to_dict(),
                        "chi_square": "Not enough unique values for chi-square test",
                    }

            # Generate Domain-Specific Plots

            # Lab Results
            try:
                logger.info("Generating Lab Results Plot")
                plt.figure(figsize=(15, 10))
                melted_labs = df.melt(
                    id_vars="readmitted",
                    value_vars=FEATURE_GROUPS["labs"],
                    var_name="test",
                    value_name="result",
                )
                sns.countplot(data=melted_labs, x="result", hue="readmitted")
                plt.title("Lab Results Distribution and Readmission Status")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = plots_dir / "lab_results_distribution.png"
                plt.savefig(plot_path, bbox_inches="tight")
                logger.debug(f"Saved plot: {plot_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to generate Lab Results plot: {e}")

            # Service Utilization
            try:
                logger.info("Generating Service Utilization Plot")
                plt.figure(figsize=(15, 10))
                melted_service = df.melt(
                    id_vars="readmitted",
                    value_vars=FEATURE_GROUPS["service"],
                    var_name="metric",
                    value_name="value",
                )
                sns.kdeplot(
                    data=melted_service,
                    x="value",
                    hue="readmitted",
                    fill=True,
                    common_norm=False,
                    palette="crest",
                    alpha=0.5,
                )
                plt.title("Service Utilization Distribution and Readmission Status")
                plt.tight_layout()
                plot_path = plots_dir / "service_utilization_distribution.png"
                plt.savefig(plot_path, bbox_inches="tight")
                logger.debug(f"Saved plot: {plot_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to generate Service Utilization plot: {e}")

            # Clinical Metrics
            try:
                logger.info("Generating Clinical Metrics Boxplot")
                plt.figure(figsize=(15, 10))
                melted_clinical = df.melt(
                    id_vars="readmitted",
                    value_vars=FEATURE_GROUPS["clinical"],
                    var_name="metric",
                    value_name="value",
                )
                sns.boxplot(
                    data=melted_clinical,
                    x="readmitted",
                    y="value",
                    hue="readmitted",
                    palette="Set3",
                )
                plt.title("Clinical Metrics Distribution and Readmission Status")
                plt.tight_layout()
                plot_path = plots_dir / "clinical_metrics_boxplot.png"
                plt.savefig(plot_path, bbox_inches="tight")
                logger.debug(f"Saved plot: {plot_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to generate Clinical Metrics boxplot: {e}")

            # Demographics
            try:
                logger.info("Generating Demographics Plot")
                plt.figure(figsize=(15, 10))
                sns.histplot(
                    data=df,
                    x="age",
                    hue="readmitted",
                    multiple="stack",
                    kde=True,
                    palette="muted",
                )
                plt.title("Age Distribution and Readmission Status")
                plt.tight_layout()
                plot_path = plots_dir / "demographics_age_distribution.png"
                plt.savefig(plot_path, bbox_inches="tight")
                logger.debug(f"Saved plot: {plot_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to generate Demographics plot: {e}")

            # Correlation Matrix
            try:
                logger.info("Generating Correlation Matrix Heatmap")
                plt.figure(figsize=(12, 10))
                corr = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="vlag",
                    mask=np.triu(np.ones_like(corr, dtype=bool)),
                    square=True,
                    cbar_kws={"shrink": 0.5},
                )
                plt.title("Correlation Matrix Heatmap")
                plt.tight_layout()
                plot_path = plots_dir / "correlation_matrix_heatmap.png"
                plt.savefig(plot_path, bbox_inches="tight")
                logger.debug(f"Saved plot: {plot_path}")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to generate Correlation Matrix heatmap: {e}")

            # Calculate feature importance
            logger.info("Calculating feature importance using RandomForestClassifier")
            numeric_features = df.select_dtypes(include=[np.number]).columns.difference(["readmitted"])
            X = df[numeric_features].fillna(0)  # Handle missing values if any
            y = df["readmitted"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            importance_df = pd.DataFrame(
                {
                    "feature": numeric_features,
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)

            # Save feature importance plot
            try:
                logger.info("Generating Feature Importance Plot")
                plt.figure(figsize=(10, 8))
                sns.barplot(data=importance_df.head(20), y="feature", x="importance")
                plt.title("Top 20 Features by Importance")
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.tight_layout()
                plot_path = plots_dir / "feature_importance.png"
                plt.savefig(plot_path, bbox_inches="tight")
                logger.debug(f"Saved plot: {plot_path}")
                plt.close()

                # Save feature importance data
                importance_csv_path = plots_dir / "feature_importance.csv"
                importance_df.to_csv(importance_csv_path, index=False)
                logger.debug(f"Saved feature importance data to: {importance_csv_path}")
            except Exception as e:
                logger.error(f"Failed to generate Feature Importance plot: {e}")

            # Calculate high correlations
            logger.info("Calculating high correlations among numeric features")
            corr_matrix = df[numeric_features].corr()
            high_correlations = (
                corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool))
                .stack()
                .pipe(lambda x: x[x.abs() > 0.8])
                .reset_index()
                .rename(columns={"level_0": "feature1", "level_1": "feature2", 0: "correlation"})
                .to_dict("records")
            )

            # Add correlation analysis
            analysis["correlations"] = {
                "high_correlations": high_correlations,
                "mean_correlation": float(np.abs(corr_matrix.values).mean()),
                "max_correlation": float(np.abs(corr_matrix.values).max()),
            }

            # Save all metrics
            logger.info("Saving exploration metrics")
            self.save_metrics("metrics", analysis)
            logger.info("Exploration completed successfully")

        except Exception as e:
            logger.error(f"Failed to run Explore stage: {str(e)}")
            raise
