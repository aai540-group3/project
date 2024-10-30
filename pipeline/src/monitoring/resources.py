import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources."""

    def __init__(self, cfg: Dict):
        """Initialize resource monitor."""
        self.cfg = cfg
        self.metrics_history = []

    def collect_metrics(self) -> Dict:
        """Collect resource metrics.

        Returns:
            Dictionary of resource metrics
        """
        metrics = {
            "cpu": {
                "usage": psutil.cpu_percent(interval=1) / 100,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent / 100
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent / 100
            },
            "timestamp": datetime.now()
        }

        self.metrics_history.append(metrics)
        return metrics

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for resource alerts.

        Args:
            metrics: Current resource metrics

        Returns:
            List of alerts
        """
        alerts = []

        # Check CPU usage
        if metrics["cpu"]["usage"] > self.cfg.thresholds.cpu:
            alerts.append({
                "type": "cpu",
                "value": metrics["cpu"]["usage"],
                "threshold": self.cfg.thresholds.cpu,
                "timestamp": datetime.now()
            })

        # Check memory usage
        if metrics["memory"]["percent"] > self.cfg.thresholds.memory:
            alerts.append({
                "type": "memory",
                "value": metrics["memory"]["percent"],
                "threshold": self.cfg.thresholds.memory,
                "timestamp": datetime.now()
            })

        # Check disk usage
        if metrics["disk"]["percent"] > self.cfg.thresholds.disk:
            alerts.append({
                "type": "disk",
                "value": metrics["disk"]["percent"],
                "threshold": self.cfg.thresholds.disk,
                "timestamp": datetime.now()
            })

        return alerts

    def get_trends(
        self,
        window_size: timedelta = timedelta(hours=1)
    ) -> Dict:
        """Get resource usage trends.

        Args:
            window_size: Time window for trends

        Returns:
            Dictionary of resource trends
        """
        if not self.metrics_history:
            return {}

        current_time = datetime.now()
        window_start = current_time - window_size

        # Filter metrics within window
        recent_metrics = [
            m for m in self.metrics_history
            if m["timestamp"] > window_start
        ]

        # Calculate trends
        trends = {
            "cpu": self._calculate_trend([m["cpu"]["usage"] for m in recent_metrics]),
            "memory": self._calculate_trend([m["memory"]["percent"] for m in recent_metrics]),
            "disk": self._calculate_trend([m["disk"]["percent"] for m in recent_metrics])
        }

        return trends

    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend statistics.

        Args:
            values: List of metric values

        Returns:
            Dictionary of trend statistics
        """
        if not values:
            return {}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "trend": np.polyfit(range(len(values)), values, 1)[0]
        }
