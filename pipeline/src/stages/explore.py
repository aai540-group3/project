import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier

from ..utils.experiment import ExperimentTracker
from ..utils.visualization import (
    plot_correlation_matrix,
    plot_distribution,
    plot_feature_importance,
)

logger = logging.getLogger(__name__)


class DataExplorationStage:
    """Data exploration and analysis stage."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tracker = ExperimentTracker(cfg, cfg.experiment.name)

    def run(self) -> None:
        """Execute exploration pipeline."""
        self.tracker.start_run(run_name="exploration")

        try:
            # Load both raw and processed data
            raw_df = pd.read_parquet(self.cfg.paths.raw / "data.parquet")
            processed_df = pd.read_parquet(
                self.cfg.paths.interim / "data_cleaned.parquet"
            )

            # Create output directories
            output_dir = Path(self.cfg.paths.reports)
            for subdir in ["figures", "cleanlab", "plots"]:
                (output_dir / subdir).mkdir(parents=True, exist_ok=True)

            # Compare distributions
            self._compare_distributions(raw_df, processed_df)

            # Generate data quality report
            self._generate_data_quality_report(processed_df)

            # Perform cleanlab analysis
            self._perform_cleanlab_analysis(processed_df)

            # Create visualizations
            self._create_visualizations(processed_df)

            logger.info("Exploration completed successfully")

        finally:
            self.tracker.end_run()

    def _compare_distributions(
        self, raw_df: pd.DataFrame, processed_df: pd.DataFrame
    ) -> None:
        """Compare distributions between raw and processed data."""
        # Compare target distributions
        target_col = self.cfg.preprocessing.target_column
        raw_dist = raw_df[target_col].value_counts()
        proc_dist = processed_df[target_col].value_counts()

        metrics = {
            "raw_class_distribution": raw_dist.to_dict(),
            "processed_class_distribution": proc_dist.to_dict(),
            "raw_class_imbalance": raw_dist.max() / raw_dist.min(),
            "processed_class_imbalance": proc_dist.max() / proc_dist.min(),
        }

        self.tracker.log_metrics(metrics)

        # Plot distributions
        plot_distribution(
            raw_dist,
            proc_dist,
            title="Target Distribution Comparison",
            output_path=self.cfg.paths.reports / "plots/target_distribution.png",
        )

    def _generate_data_quality_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive data quality report."""
        report = {
            "basic_stats": {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "n_features": len(df.columns) - 1,
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,
            },
            "data_types": df.dtypes.value_counts().to_dict(),
            "missing_values": {
                col: {"count": missing, "percentage": missing / len(df) * 100}
                for col, missing in df.isnull().sum().items()
                if missing > 0
            },
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "correlations": df.corr()[self.cfg.preprocessing.target_column]
            .sort_values(ascending=False)
            .to_dict(),
        }

        # Save report
        with open(self.cfg.paths.reports / "data_quality_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Log metrics
        self.tracker.log_metrics(
            {
                "data_quality": {
                    "completeness": 1
                    - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
                    "duplicates_ratio": 1 - len(df.drop_duplicates()) / len(df),
                    "feature_coverage": len(
                        [col for col in df.columns if df[col].nunique() > 1]
                    )
                    / len(df.columns),
                }
            }
        )

    def _perform_cleanlab_analysis(self, df: pd.DataFrame) -> None:
        """Perform cleanlab analysis for label quality."""
        X = df.drop(columns=[self.cfg.preprocessing.target_column])
        y = df[self.cfg.preprocessing.target_column]

        # Initialize and train model
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )

        # Initialize Cleanlab
        cl = CleanLearning(clf=clf)
        cl.fit(X, y)

        # Find label issues
        pred_probs = cl.predict_proba(X)
        label_issues = find_label_issues(
            labels=y.values,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )

        # Create label issues dataframe
        label_issues_df = pd.DataFrame(
            {
                "actual_label": y.iloc[label_issues],
                "predicted_label": cl.predict(X)[label_issues],
                "confidence": pred_probs[label_issues].max(axis=1),
            }
        )

        # Get feature importance
        feature_importance = pd.Series(cl.clf.feature_importances_, index=X.columns)

        # Generate analysis report
        analysis = {
            "total_issues": len(label_issues),
            "issue_percentage": len(label_issues) / len(y) * 100,
            "confidence_stats": {
                "mean": label_issues_df["confidence"].mean(),
                "median": label_issues_df["confidence"].median(),
                "std": label_issues_df["confidence"].std(),
            },
            "feature_importance": feature_importance.to_dict(),
        }

        # Save results
        label_issues_df.to_csv(
            self.cfg.paths.reports / "cleanlab/problematic_cases.csv"
        )
        feature_importance.to_csv(
            self.cfg.paths.reports / "cleanlab/feature_importance.csv"
        )

        with open(self.cfg.paths.reports / "cleanlab/analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Create visualizations
        self._create_cleanlab_visualizations(label_issues_df, feature_importance)

    def _create_cleanlab_visualizations(
        self, label_issues_df: pd.DataFrame, feature_importance: pd.Series
    ) -> None:
        """Create visualizations for cleanlab analysis."""
        # Confidence distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=label_issues_df, x="confidence", bins=30, kde=True)
        plt.title("Distribution of Confidence Scores in Label Issues")
        plt.savefig(self.cfg.paths.reports / "cleanlab/confidence_distribution.png")
        plt.close()

        # Feature importance plot
        plot_feature_importance(
            feature_importance,
            self.cfg.paths.reports / "cleanlab/feature_importance.png",
            n_features=20,
        )
