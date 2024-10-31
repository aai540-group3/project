"""
Model Drift Detection
==================

.. module:: pipeline.monitoring.drift
   :synopsis: Feature and prediction drift detection

.. moduleauthor:: aai540-group3
"""

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from scipy import stats


class DriftDetector:
    """Detect and monitor data drift in features and predictions.

    :param cfg: Drift detection configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize drift detector.

        :param cfg: Drift detection configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.reference_stats = {}
        self.current_stats = {}
        self.drift_scores = {}
        self.last_update = None
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        :raises ValueError: If required parameters are missing
        """
        required = ["window_size", "threshold", "min_samples"]
        if not all(hasattr(self.cfg, param) for param in required):
            raise ValueError(f"Missing required configuration parameters: {required}")

    def update_reference(self, data: pd.DataFrame) -> None:
        """Update reference statistics.

        :param data: New reference data
        :type data: pd.DataFrame
        """
        self.reference_stats = self._calculate_statistics(data)
        self.last_update = datetime.now()
        logger.info("Updated reference statistics")

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift in current data.

        :param current_data: Current data to check for drift
        :type current_data: pd.DataFrame
        :return: Drift scores by feature
        :rtype: Dict[str, float]
        """
        if len(current_data) < self.cfg.min_samples:
            logger.warning(f"Insufficient samples for drift detection: {len(current_data)}")
            return {}

        self.current_stats = self._calculate_statistics(current_data)
        self.drift_scores = self._compute_drift_scores()

        return self.drift_scores

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate distribution statistics.

        :param data: Input data
        :type data: pd.DataFrame
        :return: Statistics by feature
        :rtype: Dict[str, Dict]
        """
        stats = {}
        for column in data.columns:
            if data[column].dtype in [np.number]:
                stats[column] = {
                    "mean": data[column].mean(),
                    "std": data[column].std(),
                    "quantiles": data[column].quantile([0.25, 0.5, 0.75]).to_dict(),
                    "histogram": np.histogram(data[column], bins=50),
                }
            else:
                stats[column] = {
                    "distribution": data[column].value_counts(normalize=True).to_dict(),
                    "unique_count": data[column].nunique(),
                }
        return stats

    def _compute_drift_scores(self) -> Dict[str, float]:
        """Compute drift scores between reference and current data.

        :return: Drift scores by feature
        :rtype: Dict[str, float]
        """
        drift_scores = {}
        for feature in self.reference_stats.keys():
            if feature in self.current_stats:
                if isinstance(self.reference_stats[feature].get("mean"), (int, float)):
                    drift_scores[feature] = self._compute_numerical_drift(feature)
                else:
                    drift_scores[feature] = self._compute_categorical_drift(feature)
        return drift_scores

    def _compute_numerical_drift(self, feature: str) -> float:
        """Compute drift score for numerical features.

        :param feature: Feature name
        :type feature: str
        :return: Drift score
        :rtype: float
        """
        ref_stats = self.reference_stats[feature]
        curr_stats = self.current_stats[feature]

        # Compute standardized difference
        pooled_std = np.sqrt((ref_stats["std"] ** 2 + curr_stats["std"] ** 2) / 2)
        if pooled_std == 0:
            return 0.0

        effect_size = abs(ref_stats["mean"] - curr_stats["mean"]) / pooled_std
        return effect_size

    def _compute_categorical_drift(self, feature: str) -> float:
        """Compute drift score for categorical features.

        :param feature: Feature name
        :type feature: str
        :return: Drift score
        :rtype: float
        """
        ref_dist = pd.Series(self.reference_stats[feature]["distribution"])
        curr_dist = pd.Series(self.current_stats[feature]["distribution"])

        # Align distributions
        ref_dist, curr_dist = ref_dist.align(curr_dist, fill_value=0)

        # Chi-square test
        chi2, p_value = stats.chisquare(curr_dist, ref_dist)
        return 1 - p_value  # Convert p-value to drift score

    def get_drift_report(self) -> Dict:
        """Generate comprehensive drift report.

        :return: Drift analysis report
        :rtype: Dict
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "reference_update": self.last_update.isoformat() if self.last_update else None,
            "drift_scores": self.drift_scores,
            "drifted_features": [feature for feature, score in self.drift_scores.items() if score > self.cfg.threshold],
            "summary": {
                "max_drift": max(self.drift_scores.values()) if self.drift_scores else 0,
                "mean_drift": np.mean(list(self.drift_scores.values())) if self.drift_scores else 0,
                "drifted_feature_count": sum(1 for score in self.drift_scores.values() if score > self.cfg.threshold),
            },
        }

    def should_update_reference(self) -> bool:
        """Check if reference statistics should be updated.

        :return: Whether reference should be updated
        :rtype: bool
        """
        if not self.last_update:
            return True

        window_size = timedelta(days=self.cfg.window_size)
        return datetime.now() - self.last_update > window_size
