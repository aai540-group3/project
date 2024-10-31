"""
Resource Monitoring
================

.. module:: pipeline.monitoring.resources
   :synopsis: System resource monitoring and management

.. moduleauthor:: aai540-group3
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import GPUtil
import psutil
from prometheus_client import Gauge

from ..utils.logging import get_logger
from .base import BaseMonitor

logger = get_logger(__name__)

# Prometheus metrics
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
GPU_USAGE = Gauge("system_gpu_usage_percent", "GPU usage percentage", ["gpu_id"])
DISK_USAGE = Gauge(
    "system_disk_usage_percent", "Disk usage percentage", ["mount_point"]
)


class ResourceMonitor(BaseMonitor):
    """Monitor system resources and performance.

    :param cfg: Resource monitoring configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize resource monitor.

        :param cfg: Resource monitoring configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.metrics_history = []
        self.thresholds = self.cfg.thresholds
        self._initialize_monitoring()

    def _initialize_monitoring(self) -> None:
        """Initialize resource monitoring settings."""
        self.monitoring_config = {
            "cpu": self.cfg.monitor.cpu,
            "memory": self.cfg.monitor.memory,
            "gpu": self.cfg.monitor.gpu,
            "disk": self.cfg.monitor.disk,
            "network": self.cfg.monitor.network,
        }

    def collect_metrics(self) -> Dict:
        """Collect current resource metrics.

        :return: Current resource metrics
        :rtype: Dict
        :raises RuntimeError: If metric collection fails
        """
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": self._collect_cpu_metrics(),
                "memory": self._collect_memory_metrics(),
                "disk": self._collect_disk_metrics(),
                "network": self._collect_network_metrics(),
            }

            if self.monitoring_config["gpu"]:
                metrics["gpu"] = self._collect_gpu_metrics()

            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)

            # Store metrics history
            self.metrics_history.append(metrics)

            # Check for alerts
            self._check_resource_alerts(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            raise RuntimeError(f"Resource metric collection failed: {e}")

    def _collect_cpu_metrics(self) -> Dict:
        """Collect CPU metrics.

        :return: CPU metrics
        :rtype: Dict
        """
        cpu_metrics = {
            "usage_percent": psutil.cpu_percent(interval=1),
            "per_cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
            "load_avg": psutil.getloadavg(),
            "context_switches": psutil.cpu_stats().ctx_switches,
            "interrupts": psutil.cpu_stats().interrupts,
            "soft_interrupts": psutil.cpu_stats().soft_interrupts,
            "syscalls": psutil.cpu_stats().syscalls,
        }

        if hasattr(psutil, "cpu_freq"):
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_metrics.update(
                    {
                        "freq_current": cpu_freq.current,
                        "freq_min": cpu_freq.min,
                        "freq_max": cpu_freq.max,
                    }
                )

        return cpu_metrics

    def _collect_memory_metrics(self) -> Dict:
        """Collect memory metrics.

        :return: Memory metrics
        :rtype: Dict
        """
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()

        return {
            "virtual": {
                "total": virtual_memory.total,
                "available": virtual_memory.available,
                "used": virtual_memory.used,
                "free": virtual_memory.free,
                "percent": virtual_memory.percent,
                "cached": getattr(virtual_memory, "cached", None),
                "buffers": getattr(virtual_memory, "buffers", None),
            },
            "swap": {
                "total": swap_memory.total,
                "used": swap_memory.used,
                "free": swap_memory.free,
                "percent": swap_memory.percent,
                "sin": swap_memory.sin,
                "sout": swap_memory.sout,
            },
        }

    def _collect_gpu_metrics(self) -> Dict:
        """Collect GPU metrics if available.

        :return: GPU metrics
        :rtype: Dict
        """
        gpu_metrics = {}
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics[gpu.id] = {
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                        "percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    },
                    "temperature": gpu.temperature,
                    "powerDraw": getattr(gpu, "powerDraw", None),
                    "powerLimit": getattr(gpu, "powerLimit", None),
                }
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")

        return gpu_metrics

    def _collect_disk_metrics(self) -> Dict:
        """Collect disk metrics.

        :return: Disk metrics
        :rtype: Dict
        """
        disk_metrics = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_metrics[partition.mountpoint] = {
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                    "fstype": partition.fstype,
                    "device": partition.device,
                }
            except Exception as e:
                logger.warning(
                    f"Failed to collect disk metrics for {partition.mountpoint}: {e}"
                )

        # Add disk I/O metrics
        try:
            disk_io = psutil.disk_io_counters()
            disk_metrics["io"] = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count,
                "read_time": disk_io.read_time,
                "write_time": disk_io.write_time,
            }
        except Exception as e:
            logger.warning(f"Failed to collect disk I/O metrics: {e}")

        return disk_metrics

    def _collect_network_metrics(self) -> Dict:
        """Collect network metrics.

        :return: Network metrics
        :rtype: Dict
        """
        network_metrics = {}
        try:
            net_io = psutil.net_io_counters()
            network_metrics["io"] = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout,
            }

            # Add per-interface metrics
            net_if = psutil.net_if_stats()
            network_metrics["interfaces"] = {
                interface: {"isup": stats.isup, "speed": stats.speed, "mtu": stats.mtu}
                for interface, stats in net_if.items()
            }

        except Exception as e:
            logger.warning(f"Failed to collect network metrics: {e}")

        return network_metrics

    def _update_prometheus_metrics(self, metrics: Dict) -> None:
        """Update Prometheus metrics.

        :param metrics: Current metrics
        :type metrics: Dict
        """
        # CPU metrics
        CPU_USAGE.set(metrics["cpu"]["usage_percent"])

        # Memory metrics
        MEMORY_USAGE.set(metrics["memory"]["virtual"]["used"])

        # GPU metrics
        if "gpu" in metrics:
            for gpu_id, gpu_metrics in metrics["gpu"].items():
                GPU_USAGE.labels(gpu_id=gpu_id).set(gpu_metrics["load"])

        # Disk metrics
        for mount_point, disk_metrics in metrics["disk"].items():
            if isinstance(disk_metrics, dict) and "percent" in disk_metrics:
                DISK_USAGE.labels(mount_point=mount_point).set(disk_metrics["percent"])

    def _check_resource_alerts(self, metrics: Dict) -> None:
        """Check for resource usage alerts.

        :param metrics: Current metrics
        :type metrics: Dict
        """
        alerts = []

        # CPU alerts
        if metrics["cpu"]["usage_percent"] > self.thresholds.cpu.critical:
            alerts.append(
                {
                    "type": "cpu",
                    "severity": "critical",
                    "message": f"CPU usage critical: {metrics['cpu']['usage_percent']}%",
                }
            )
        elif metrics["cpu"]["usage_percent"] > self.thresholds.cpu.warning:
            alerts.append(
                {
                    "type": "cpu",
                    "severity": "warning",
                    "message": f"CPU usage high: {metrics['cpu']['usage_percent']}%",
                }
            )

        # Memory alerts
        memory_used_percent = metrics["memory"]["virtual"]["percent"]
        if memory_used_percent > self.thresholds.memory.critical:
            alerts.append(
                {
                    "type": "memory",
                    "severity": "critical",
                    "message": f"Memory usage critical: {memory_used_percent}%",
                }
            )
        elif memory_used_percent > self.thresholds.memory.warning:
            alerts.append(
                {
                    "type": "memory",
                    "severity": "warning",
                    "message": f"Memory usage high: {memory_used_percent}%",
                }
            )

        # Send alerts
        for alert in alerts:
            self._send_alerts(alert)

    def get_metrics_history(self, minutes: Optional[int] = None) -> List[Dict]:
        """Get historical metrics.

        :param minutes: Time window in minutes
        :type minutes: Optional[int]
        :return: Historical metrics
        :rtype: List[Dict]
        """
        if not minutes:
            return self.metrics_history

        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

    def cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        retention_days = self.cfg.retention_days
        cutoff = datetime.now() - timedelta(days=retention_days)

        self.metrics_history = [
            m
            for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
