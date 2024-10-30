import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor model performance."""

    def __init__(self, cfg: Dict):
        """Initialize performance monitor."""
        self.cfg = cfg
        self.latencies = []
        self.error_counts = []
        self.prediction_counts = []

    def record_prediction(
        self,
        latency: float,
        error: bool = False,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record prediction performance.

        Args:
            latency: Prediction latency in milliseconds
            error: Whether prediction resulted in error
            timestamp: Prediction timestamp
        """
        timestamp = timestamp or datetime.now()

        self.latencies.append({
            "value": latency,
            "timestamp": timestamp
        })

        self.error_counts.append({
            "value": int(error),
            "timestamp": timestamp
        })

        self.prediction_counts.append({
            "value": 1,
            "timestamp": timestamp
        })

    def get_metrics(
        self,
        window_size: timedelta = timedelta(minutes=5)
    ) -> Dict:
        """Get performance metrics.

        Args:
            window_size: Time window for metrics

        Returns:
            Dictionary of performance metrics
        """
        current_time = datetime.now()
        window_start = current_time - window_size

        # Filter metrics within window
        recent_latencies = [
            l["value"] for l in self.latencies
            if l["timestamp"] > window_start
        ]

        recent_errors = [
            e["value"] for e in self.error_counts
            if e["timestamp"] > window_start
        ]

        recent_predictions = [
            p["value"] for p in self.prediction_counts
            if p["timestamp"] > window_start
        ]

        # Calculate metrics
        metrics = {
            "latency": {
                "p50": np.percentile(recent_latencies, 50),
                "p95": np.percentile(recent_latencies, 95),
                "p99": np.percentile(recent_latencies, 99),
                "mean": np.mean(recent_latencies)
            },
            "error_rate": np.mean(recent_errors) if recent_errors else 0,
            "throughput": len(recent_predictions) / window_size.total_seconds()
        }

        return metrics

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for performance alerts.

        Args:
            metrics: Current performance metrics

        Returns:
            List of alerts
        """
        alerts = []

        # Check latency thresholds
        for percentile in ["p95", "p99"]:
            threshold = self.cfg.thresholds.latency[percentile]
            if metrics["latency"][percentile] > threshold:
                alerts.append({
                    "type": "latency",
                    "metric": percentile,
                    "value": metrics["latency"][percentile],
                    "threshold": threshold,
                    "timestamp": datetime.now()
                })

        # Check error rate
        if metrics["error_rate"] > self.cfg.thresholds.error_rate:
            alerts.append({
                "type": "error_rate",
                "value": metrics["error_rate"],
                "threshold": self.cfg.thresholds.error_rate,
                "timestamp": datetime.now()
            })

        # Check throughput
        if metrics["throughput"] < self.cfg.thresholds.min_throughput:
            alerts.append({
                "type": "throughput",
                "value": metrics["throughput"],
                "threshold": self.cfg.thresholds.min_throughput,
                "timestamp": datetime.now()
            })

        return alerts
