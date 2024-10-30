import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

class MetricsMonitor:
    """Monitor model metrics."""

    def __init__(self, cfg: Dict):
        """Initialize metrics monitor."""
        self.cfg = cfg
        self.metrics_history = []

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """Calculate model metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            timestamp: Timestamp for metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": np.mean(y_true == y_pred),
            "precision": np.mean(y_true[y_pred == 1]),
            "recall": np.mean(y_pred[y_true == 1]),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "timestamp": timestamp or datetime.now()
        }

        self.metrics_history.append(metrics)
        return metrics

    def check_drift(
        self,
        window_size: timedelta = timedelta(days=7)
    ) -> Dict:
        """Check for metric drift.

        Args:
            window_size: Time window for drift detection

        Returns:
            Dictionary of drift metrics
        """
        if not self.metrics_history:
            return {}

        df = pd.DataFrame(self.metrics_history)
        current_time = df["timestamp"].max()
        window_start = current_time - window_size

        recent_metrics = df[df["timestamp"] > window_start]
        baseline_metrics = df[df["timestamp"] <= window_start]

        drift_metrics = {}
        for metric in ["accuracy", "roc_auc"]:
            if not baseline_metrics.empty:
                baseline_mean = baseline_metrics[metric].mean()
                recent_mean = recent_metrics[metric].mean()
                drift = (recent_mean - baseline_mean) / baseline_mean
                drift_metrics[f"{metric}_drift"] = drift

        return drift_metrics
