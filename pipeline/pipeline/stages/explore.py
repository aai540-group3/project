import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier

from pipeline.stages.base import PipelineStage


class Explore(PipelineStage):
    """Pipeline stage for data exploration and analysis."""

    def run(self):
        """Execute exploration analysis."""

        # Load data
        df = self.load_data("data_cleaned.parquet", "data/interim")
        logger.info(f"Dataset: {df.shape}")

        # Define feature groups
        FEATURE_GROUPS = {
            "demographic": ["race", "gender", "age"],
            "clinical": ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"],
            "service": ["number_outpatient", "number_emergency", "number_inpatient"],
            "labs": ["max_glu_serum", "a1cresult"],
            "medications": [
                col for col in df.columns if any(med in col for med in ["metformin", "insulin", "glyburide"])
            ],
            "diagnosis": [col for col in df.columns if "diag" in col],
        }  # fmt: skip

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
            # Split features by type
            numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df[cols].select_dtypes(exclude=[np.number]).columns.tolist()

            group_stats = {}

            # Handle numeric features
            if len(numeric_cols) > 0:
                group_stats.update(
                    {
                        "numeric_stats": df[numeric_cols].describe().to_dict(),
                        "numeric_correlations": df[numeric_cols].corr().to_dict(),
                        "readmission_means": df.groupby("readmitted")[numeric_cols].mean().to_dict(),
                    }
                )

            # Handle categorical features
            if len(categorical_cols) > 0:
                group_stats.update(
                    {
                        "categorical_stats": {
                            col: df[col].value_counts(normalize=True).to_dict() for col in categorical_cols
                        },
                        "categorical_counts": {
                            col: df.groupby("readmitted")[col]
                            .value_counts()
                            .reset_index(name="count")
                            .to_dict("records")
                            for col in categorical_cols
                        },
                    }
                )

            # Add missing rates for all columns
            group_stats["missing_rates"] = df[cols].isnull().mean().to_dict()
            group_analysis[group] = group_stats

        analysis["groups"] = group_analysis

        # Categorical associations (chi-square)
        analysis["categorical"] = {}
        for col in df.select_dtypes(exclude=[np.number]).columns:
            # Ensure that there are at least two unique values to perform chi-square test
            if df[col].nunique() > 1:
                contingency_table = pd.crosstab(df[col], df["readmitted"])
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                analysis["categorical"][col] = {
                    "distribution": df[col].value_counts(normalize=True).to_dict(),
                    "readmission_rate": df.groupby(col)["readmitted"].mean().to_dict(),
                    "chi_square": {"chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof},
                }
            else:
                analysis["categorical"][col] = {
                    "distribution": df[col].value_counts(normalize=True).to_dict(),
                    "readmission_rate": df.groupby(col)["readmitted"].mean().to_dict(),
                    "chi_square": "Not enough unique values for chi-square test",
                }

        # Generate domain-specific plots
        plots = {
            # Medical domain plots
            "lab_results": lambda: (
                sns.FacetGrid(
                    data=df.melt(
                        id_vars="readmitted",
                        value_vars=FEATURE_GROUPS["labs"],
                        var_name="test",
                        value_name="result",
                    ),
                    col="test",
                    hue="readmitted",
                    sharex=False,
                    sharey=False,
                )
                .map_dataframe(sns.countplot, x="result")
                .add_legend()
            ),
            # Service utilization
            "service_metrics": lambda: (
                sns.FacetGrid(
                    data=df.melt(
                        id_vars="readmitted",
                        value_vars=FEATURE_GROUPS["service"],
                        var_name="metric",
                        value_name="value",
                    ),
                    col="metric",
                    hue="readmitted",
                    col_wrap=2,
                    sharex=False,
                    sharey=False,
                )
                .map_dataframe(sns.kdeplot, x="value", fill=True, warn_singular=False)
                .add_legend()
            ),
            # Clinical metrics
            "clinical_metrics": lambda: (
                sns.FacetGrid(
                    data=df.melt(
                        id_vars="readmitted",
                        value_vars=FEATURE_GROUPS["clinical"],
                        var_name="metric",
                        value_name="value",
                    ),
                    col="metric",
                    col_wrap=2,
                    sharex=False,
                    sharey=False,
                )
                .map_dataframe(
                    sns.boxplot,
                    x="readmitted",
                    y="value",
                    hue="readmitted",
                    palette="dark:#4c72b0",
                )
                .add_legend()
            ),
            # Demographic analysis
            "demographics": lambda: (
                sns.FacetGrid(
                    data=df,
                    col="race",
                    row="gender",
                    hue="readmitted",
                    sharex=False,
                    sharey=False,
                )
                .map_dataframe(sns.histplot, x="age", bins=10, kde=True)
                .add_legend()
            ),
            # Correlation matrix
            "correlations": lambda: sns.heatmap(
                data=df.select_dtypes(include=[np.number]).corr(),
                center=0,
                cmap="vlag",
                mask=np.triu(np.ones_like(df.select_dtypes(include=[np.number]).corr(), dtype=bool)),
                square=True,
            ),
        }

        # Generate and save plots
        for name, plot_func in plots.items():
            self.save_plot(name, plot_func)

        # Calculate feature importance
        numeric_features = df.select_dtypes(include=[np.number]).columns.difference(["readmitted"])
        X = df[numeric_features]
        y = df["readmitted"]

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        importance_df = pd.DataFrame(
            {
                "feature": numeric_features,
                "importance": model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)

        # Save feature importance plot and data
        self.save_plot(
            "feature_importance",
            lambda data: sns.barplot(
            data=data.head(20),
            y="feature",
            x="importance",
            hue="feature",
            legend=False,
            palette="viridis"
            ).set(title="Top 20 Features by Importance"),
            data=importance_df,
        )  # fmt: skip
        importance_df.to_csv("plots/explore/feature_importance.csv", index=False)

        # Calculate high correlations
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
        self.save_metrics("metrics", analysis)
        logger.info("Exploration completed successfully")
