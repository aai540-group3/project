"""
Explore Stage
===================

.. module:: pipeline.stages.explore
   :synopsis: Data exploration and analysis stage

.. moduleauthor:: aai540-group3
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from ..stages.base import PipelineStage
from ..utils.visualization import (
    plot_correlation_matrix,
    plot_distribution,
    plot_feature_importance,
    plot_target_distribution,
)


class ExploreStage(PipelineStage):
    """Data exploration and analysis stage implementation.

    :param cfg: Stage configuration
    :type cfg: DictConfig
    """

    def run(self) -> None:
        """Execute exploration pipeline.

        :raises RuntimeError: If exploration fails
        """
        logger.info("Starting data exploration")
        self.tracker.start_run(run_name="explore")

        try:
            # Load data
            raw_df = self._load_raw_data()
            processed_df = self._load_processed_data()

            # Create output directories
            self._create_output_directories()

            # Perform analysis
            self._analyze_data_quality(raw_df, processed_df)
            self._analyze_distributions(raw_df, processed_df)
            self._analyze_correlations(processed_df)
            self._analyze_feature_importance(processed_df)
            self._perform_cleanlab_analysis(processed_df)

            logger.info("Data exploration completed successfully")

        except Exception as e:
            logger.error(f"Data exploration failed: {str(e)}")
            raise RuntimeError(f"Data exploration failed: {str(e)}")
        finally:
            self.tracker.end_run()

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw data.

        :return: Raw DataFrame
        :rtype: pd.DataFrame
        :raises FileNotFoundError: If raw data not found
        """
        raw_path = Path(self.cfg.paths.raw) / "data.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")
        return pd.read_parquet(raw_path)

    def _load_processed_data(self) -> pd.DataFrame:
        """Load processed data.

        :return: Processed DataFrame
        :rtype: pd.DataFrame
        :raises FileNotFoundError: If processed data not found
        """
        processed_path = Path(self.cfg.paths.interim) / "data_cleaned.parquet"
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found: {processed_path}")
        return pd.read_parquet(processed_path)

    def _create_output_directories(self) -> None:
        """Create output directories for analysis results."""
        output_dir = Path(self.cfg.paths.reports)
        for subdir in ["figures", "cleanlab", "analysis"]:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _analyze_data_quality(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
        """Analyze data quality metrics.

        :param raw_df: Raw DataFrame
        :type raw_df: pd.DataFrame
        :param processed_df: Processed DataFrame
        :type processed_df: pd.DataFrame
        """
        quality_metrics = {
            "raw": self._calculate_quality_metrics(raw_df),
            "processed": self._calculate_quality_metrics(processed_df),
        }

        # Save metrics
        metrics_path = Path(self.cfg.paths.reports) / "analysis/data_quality.json"
        with metrics_path.open("w") as f:
            json.dump(quality_metrics, f, indent=2)

        # Log to tracking system
        self.tracker.log_metrics({f"quality_{k}_{mk}": mv for k, m in quality_metrics.items() for mk, mv in m.items()})

        # Generate visualizations
        self._plot_quality_metrics(quality_metrics)

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        :return: Quality metrics
        :rtype: Dict
        """
        return {
            "missing_rate": df.isnull().mean().mean(),
            "duplicate_rate": df.duplicated().mean(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(exclude=[np.number]).columns),
        }

    def _analyze_distributions(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
        """Analyze feature distributions.

        :param raw_df: Raw DataFrame
        :type raw_df: pd.DataFrame
        :param processed_df: Processed DataFrame
        :type processed_df: pd.DataFrame
        """
        # Numeric distributions
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_distribution(
                raw_df[col] if col in raw_df else None,
                processed_df[col],
                title=f"Distribution of {col}",
                ax=ax,
            )
            plt.savefig(
                Path(self.cfg.paths.reports) / f"figures/dist_{col}.png",
                bbox_inches="tight",
            )
            plt.close()

        # Target distribution
        target_col = self.cfg.data.target_column
        if target_col in processed_df.columns:
            plot_target_distribution(
                raw_df[target_col],
                processed_df[target_col],
                save_path=Path(self.cfg.paths.reports) / "figures/target_dist.png",
            )

    def _analyze_correlations(self, df: pd.DataFrame) -> None:
        """Analyze feature correlations.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        """
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate correlations
        corr_matrix = numeric_df.corr()

        # Save correlation matrix
        corr_path = Path(self.cfg.paths.reports) / "analysis/correlations.json"
        with corr_path.open("w") as f:
            json.dump(corr_matrix.to_dict(), f, indent=2)

        # Plot correlation matrix
        plot_correlation_matrix(
            corr_matrix,
            save_path=Path(self.cfg.paths.reports) / "figures/correlation_matrix.png",
        )

        # Log high correlations
        high_corr = self._get_high_correlations(corr_matrix, threshold=self.cfg.explore.correlation_threshold)
        if high_corr:
            logger.warning(f"Found {len(high_corr)} high correlations")
            self.tracker.log_metrics({"high_correlations_count": len(high_corr)})

    def _analyze_feature_importance(self, df: pd.DataFrame) -> None:
        """Analyze feature importance.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        """
        target_col = self.cfg.data.target_column
        if target_col not in df.columns:
            logger.warning("Target column not found for feature importance analysis")
            return

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Train a simple random forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.cfg.seed)
        rf.fit(X, y)

        # Calculate feature importance
        importance = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        # Save feature importance
        importance.to_csv(
            Path(self.cfg.paths.reports) / "analysis/feature_importance.csv",
            index=False,
        )

        # Plot feature importance
        plot_feature_importance(
            importance,
            save_path=Path(self.cfg.paths.reports) / "figures/feature_importance.png",
        )

    def _perform_cleanlab_analysis(self, df: pd.DataFrame) -> None:
        """Perform cleanlab analysis for label quality.

        :param df: Input DataFrame
        :type df: pd.DataFrame
        """
        target_col = self.cfg.data.target_column
        if target_col not in df.columns:
            logger.warning("Target column not found for cleanlab analysis")
            return

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Initialize and train model
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.cfg.seed)

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
                "index": label_issues,
                "actual_label": y.iloc[label_issues],
                "predicted_label": cl.predict(X)[label_issues],
                "confidence": pred_probs[label_issues].max(axis=1),
            }
        )

        # Save results
        label_issues_df.to_csv(Path(self.cfg.paths.reports) / "cleanlab/label_issues.csv", index=False)

        # Log metrics
        self.tracker.log_metrics(
            {
                "label_issues_count": len(label_issues),
                "label_issues_rate": len(label_issues) / len(y),
            }
        )

    def _get_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Get highly correlated feature pairs.

        :param corr_matrix: Correlation matrix
        :type corr_matrix: pd.DataFrame
        :param threshold: Correlation threshold
        :type threshold: float
        :return: List of high correlations
        :rtype: List[Tuple[str, str, float]]
        """
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j],
                        )
                    )
        return high_corr

    def _plot_quality_metrics(self, metrics: Dict) -> None:
        """Plot data quality metrics.

        :param metrics: Quality metrics
        :type metrics: Dict
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle("Data Quality Metrics Comparison")

        # Missing values
        axes[0, 0].bar(
            ["Raw", "Processed"],
            [metrics["raw"]["missing_rate"], metrics["processed"]["missing_rate"]],
        )
        axes[0, 0].set_title("Missing Value Rate")

        # Duplicate rate
        axes[0, 1].bar(
            ["Raw", "Processed"],
            [metrics["raw"]["duplicate_rate"], metrics["processed"]["duplicate_rate"]],
        )
        axes[0, 1].set_title("Duplicate Rate")

        # Column counts
        axes[1, 0].bar(
            ["Raw", "Processed"],
            [metrics["raw"]["column_count"], metrics["processed"]["column_count"]],
        )
        axes[1, 0].set_title("Column Count")

        # Memory usage
        axes[1, 1].bar(
            ["Raw", "Processed"],
            [
                metrics["raw"]["memory_usage_mb"],
                metrics["processed"]["memory_usage_mb"],
            ],
        )
        axes[1, 1].set_title("Memory Usage (MB)")

        plt.tight_layout()
        plt.savefig(Path(self.cfg.paths.reports) / "figures/quality_metrics.png")
        plt.close()
