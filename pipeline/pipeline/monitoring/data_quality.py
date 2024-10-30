import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor data quality."""

    def __init__(self, cfg: Dict):
        """Initialize data quality monitor."""
        self.cfg = cfg
        self.reference_stats = None

    def set_reference(self, data: pd.DataFrame) -> None:
        """Set reference data statistics.

        Args:
            data: Reference data
        """
        self.reference_stats = self._calculate_statistics(data)

    def check_quality(self, data: pd.DataFrame) -> Dict:
        """Check data quality.

        Args:
            data: Data to check

        Returns:
            Dictionary of quality metrics
        """
        current_stats = self._calculate_statistics(data)

        metrics = {
            "missing_values": self._check_missing_values(data),
            "duplicates": self._check_duplicates(data),
            "outliers": self._check_outliers(data),
            "drift": self._check_distribution_drift(self.reference_stats, current_stats)
            if self.reference_stats
            else {},
        }

        return metrics

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate data statistics.

        Args:
            data: Input data

        Returns:
            Dictionary of statistics
        """
        stats = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            stats[column] = {
                "mean": data[column].mean(),
                "std": data[column].std(),
                "quantiles": data[column].quantile([0.25, 0.5, 0.75]).to_dict(),
            }
        return stats

    def _check_missing_values(self, data: pd.DataFrame) -> Dict:
        """Check missing values.

        Args:
            data: Input data

        Returns:
            Dictionary of missing value metrics
        """
        missing = data.isnull().sum()
        return {
            "total": missing.sum(),
            "percentage": (missing / len(data) * 100).to_dict(),
        }

    def _check_duplicates(self, data: pd.DataFrame) -> Dict:
        """Check duplicates.

        Args:
            data: Input data

        Returns:
            Dictionary of duplicate metrics
        """
        duplicates = data.duplicated()
        return {"total": duplicates.sum(), "percentage": duplicates.mean() * 100}

    def _check_outliers(self, data: pd.DataFrame) -> Dict:
        """Check outliers using IQR method.

        Args:
            data: Input data

        Returns:
            Dictionary of outlier metrics
        """
        outliers = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers[column] = {
                "total": (
                    (data[column] < (Q1 - 1.5 * IQR))
                    | (data[column] > (Q3 + 1.5 * IQR))
                ).sum(),
                "percentage": (
                    (data[column] < (Q1 - 1.5 * IQR))
                    | (data[column] > (Q3 + 1.5 * IQR))
                ).mean()
                * 100,
            }
        return outliers

    def _check_distribution_drift(
        self, reference_stats: Dict, current_stats: Dict
    ) -> Dict:
        """Check distribution drift using KS test.

        Args:
            reference_stats: Reference statistics
            current_stats: Current statistics

        Returns:
            Dictionary of drift metrics
        """
        drift = {}
        for column in reference_stats.keys():
            ref_mean = reference_stats[column]["mean"]
            ref_std = reference_stats[column]["std"]
            curr_mean = current_stats[column]["mean"]
            curr_std = current_stats[column]["std"]

            # Calculate standardized difference
            effect_size = abs(ref_mean - curr_mean) / np.sqrt(
                (ref_std**2 + curr_std**2) / 2
            )

            drift[column] = {
                "effect_size": effect_size,
                "significant_drift": effect_size > self.cfg.drift_threshold,
            }

        return drift
