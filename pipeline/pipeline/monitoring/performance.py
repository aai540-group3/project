"""
Data Quality Monitoring
====================

.. module:: pipeline.monitoring.data_quality
   :synopsis: Data quality monitoring and validation

.. moduleauthor:: aai540-group3
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from .base import BaseMonitor


class DataQualityMonitor(BaseMonitor):
    """Monitor and validate data quality.

    :param cfg: Data quality configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize data quality monitor.

        :param cfg: Data quality configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.reference_stats = {}
        self.violations = []
        self._initialize_constraints()

    def _initialize_constraints(self) -> None:
        """Initialize data quality constraints."""
        self.constraints = {
            "missing_threshold": self.cfg.constraints.missing_threshold,
            "outlier_std": self.cfg.constraints.outlier_std,
            "categorical_levels": self.cfg.constraints.categorical_levels,
            "correlation_threshold": self.cfg.constraints.correlation_threshold,
            "value_ranges": self.cfg.constraints.value_ranges,
        }

    def check_quality(self, data: pd.DataFrame) -> Dict:
        """Check data quality against constraints.

        :param data: Input data to validate
        :type data: pd.DataFrame
        :return: Quality check results
        :rtype: Dict
        :raises ValueError: If data is empty or invalid
        """
        if data.empty:
            raise ValueError("Empty dataset provided")

        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "missing_values": self._check_missing_values(data),
                "outliers": self._check_outliers(data),
                "categorical_validity": self._check_categorical_validity(data),
                "value_ranges": self._check_value_ranges(data),
                "correlations": self._check_correlations(data),
            },
        }

        # Aggregate results
        results["summary"] = self._summarize_results(results["checks"])

        # Record violations
        self._record_violations(results)

        return results

    def _check_missing_values(self, data: pd.DataFrame) -> Dict:
        """Check for missing values.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Missing value analysis
        :rtype: Dict
        """
        missing_stats = {}
        for column in data.columns:
            missing_rate = data[column].isnull().mean()
            missing_stats[column] = {
                "rate": missing_rate,
                "count": data[column].isnull().sum(),
                "violated": missing_rate > self.constraints["missing_threshold"],
            }

        return {
            "details": missing_stats,
            "total_rate": data.isnull().mean().mean(),
            "violated_columns": [col for col, stats in missing_stats.items() if stats["violated"]],
        }

    def _check_outliers(self, data: pd.DataFrame) -> Dict:
        """Check for statistical outliers.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Outlier analysis
        :rtype: Dict
        """
        outlier_stats = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            outlier_mask = z_scores > self.constraints["outlier_std"]

            outlier_stats[column] = {
                "count": outlier_mask.sum(),
                "rate": outlier_mask.mean(),
                "violated": outlier_mask.mean() > self.cfg.constraints.outlier_threshold,
            }

        return {
            "details": outlier_stats,
            "violated_columns": [col for col, stats in outlier_stats.items() if stats["violated"]],
        }

    def _check_categorical_validity(self, data: pd.DataFrame) -> Dict:
        """Check categorical variable validity.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Categorical validity analysis
        :rtype: Dict
        """
        categorical_stats = {}
        expected_levels = self.constraints["categorical_levels"]

        for column, levels in expected_levels.items():
            if column in data.columns:
                unexpected = set(data[column].unique()) - set(levels)
                categorical_stats[column] = {
                    "unexpected_values": list(unexpected),
                    "unexpected_count": len(unexpected),
                    "violated": len(unexpected) > 0,
                }

        return {
            "details": categorical_stats,
            "violated_columns": [col for col, stats in categorical_stats.items() if stats["violated"]],
        }

    def _check_value_ranges(self, data: pd.DataFrame) -> Dict:
        """Check value ranges for numeric variables.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Value range analysis
        :rtype: Dict
        """
        range_stats = {}
        value_ranges = self.constraints["value_ranges"]

        for column, range_def in value_ranges.items():
            if column in data.columns:
                violations = ((data[column] < range_def["min"]) | (data[column] > range_def["max"])).sum()

                range_stats[column] = {
                    "violation_count": violations,
                    "violation_rate": violations / len(data),
                    "violated": violations > 0,
                }

        return {
            "details": range_stats,
            "violated_columns": [col for col, stats in range_stats.items() if stats["violated"]],
        }

    def _check_correlations(self, data: pd.DataFrame) -> Dict:
        """Check feature correlations.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Correlation analysis
        :rtype: Dict
        """
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {"details": {}, "violated_pairs": []}

        corr_matrix = numeric_data.corr()
        threshold = self.constraints["correlation_threshold"]

        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append(
                        {
                            "features": (
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                            ),
                            "correlation": corr_matrix.iloc[i, j],
                        }
                    )

        return {
            "details": {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlation_pairs": high_corr_pairs,
            },
            "violated_pairs": [pair["features"] for pair in high_corr_pairs],
        }

    def _summarize_results(self, checks: Dict) -> Dict:
        """Summarize quality check results.

        :param checks: Quality check results
        :type checks: Dict
        :return: Summary statistics
        :rtype: Dict
        """
        total_violations = sum(len(check.get("violated_columns", [])) for check in checks.values())

        return {
            "total_violations": total_violations,
            "checks_passed": total_violations == 0,
            "violation_types": {
                check_type: len(check.get("violated_columns", [])) for check_type, check in checks.items()
            },
        }

    def _record_violations(self, results: Dict) -> None:
        """Record quality violations for tracking.

        :param results: Quality check results
        :type results: Dict
        """
        if results["summary"]["total_violations"] > 0:
            violation = {
                "timestamp": results["timestamp"],
                "violations": results["summary"]["violation_types"],
                "details": results["checks"],
            }
            self.violations.append(violation)

            # Send alerts if configured
            if self.cfg.alerts.enabled:
                self._send_alerts(violation)

    def get_violation_history(self, days: Optional[int] = None) -> List[Dict]:
        """Get historical violation records.

        :param days: Number of days of history
        :type days: Optional[int]
        :return: Violation history
        :rtype: List[Dict]
        """
        if not days:
            return self.violations

        cutoff = datetime.now() - timedelta(days=days)
        return [v for v in self.violations if datetime.fromisoformat(v["timestamp"]) > cutoff]

    def update_reference_statistics(self, data: pd.DataFrame) -> None:
        """Update reference statistics for monitoring.

        :param data: Reference data
        :type data: pd.DataFrame
        """
        self.reference_stats = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self._calculate_reference_statistics(data),
        }
        logger.info("Updated reference statistics")

    def _calculate_reference_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate reference statistics for monitoring.

        :param data: Reference data
        :type data: pd.DataFrame
        :return: Reference statistics
        :rtype: Dict
        """
        stats = {}

        # Numeric statistics
        numeric_data = data.select_dtypes(include=[np.number])
        for column in numeric_data.columns:
            stats[column] = {
                "mean": data[column].mean(),
                "std": data[column].std(),
                "quantiles": data[column].quantile([0.25, 0.5, 0.75]).to_dict(),
                "range": (data[column].min(), data[column].max()),
            }

        # Categorical statistics
        categorical_data = data.select_dtypes(exclude=[np.number])
        for column in categorical_data.columns:
            stats[column] = {
                "value_counts": data[column].value_counts().to_dict(),
                "unique_count": data[column].nunique(),
            }

        return stats
