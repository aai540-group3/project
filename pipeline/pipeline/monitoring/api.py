"""
Monitoring API
===========

.. module:: pipeline.monitoring.api
   :synopsis: REST API for monitoring system

.. moduleauthor:: aai540-group3
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Path, Query
from loguru import logger
from pydantic import BaseModel, Field

from .alerts import AlertManager
from .data_quality import DataQualityMonitor
from .drift import DriftDetector
from .performance import PerformanceMonitor
from .resources import ResourceMonitor

app = FastAPI(
    title="MLOps Monitoring API",
    description="API for model and system monitoring",
    version="1.0.0",
)


# Models for request/response
class Alert(BaseModel):
    """Alert model."""

    type: str = Field(..., description="Alert type")
    message: str = Field(..., description="Alert message")
    severity: str = Field(..., description="Alert severity")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class MetricValue(BaseModel):
    """Metric value model."""

    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(..., description="Metric timestamp")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


@app.post("/alerts", response_model=Dict)
async def create_alert(alert: Alert):
    """Create new alert.

    :param alert: Alert details
    :type alert: Alert
    :return: Created alert
    :rtype: Dict
    """
    try:
        created_alert = AlertManager.create_alert(
            alert_type=alert.type,
            message=alert.message,
            severity=alert.severity,
            metadata=alert.metadata,
        )
        return created_alert
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/active", response_model=List[Dict])
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    alert_type: Optional[str] = Query(None, description="Filter by type"),
):
    """Get active alerts.

    :param severity: Filter by severity
    :type severity: Optional[str]
    :param alert_type: Filter by type
    :type alert_type: Optional[str]
    :return: List of active alerts
    :rtype: List[Dict]
    """
    return AlertManager.get_active_alerts(severity=severity, alert_type=alert_type)


@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str = Path(..., description="Alert ID")):
    """Acknowledge alert.

    :param alert_id: Alert ID
    :type alert_id: str
    :raises HTTPException: If alert not found
    """
    try:
        AlertManager.acknowledge_alert(alert_id)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/metrics/{metric_name}", response_model=Dict)
async def record_metric(
    metric_name: str = Path(..., description="Metric name"),
    value: MetricValue = Body(..., description="Metric value"),
):
    """Record metric value.

    :param metric_name: Metric name
    :type metric_name: str
    :param value: Metric value
    :type value: MetricValue
    :return: Recorded metric
    :rtype: Dict
    """
    try:
        PerformanceMonitor.record_metric(metric_name, value.value, value.timestamp, value.metadata)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/performance", response_model=Dict)
async def get_performance_metrics(
    window_minutes: Optional[int] = Query(5, description="Time window in minutes"),
):
    """Get performance metrics.

    :param window_minutes: Time window in minutes
    :type window_minutes: Optional[int]
    :return: Performance metrics
    :rtype: Dict
    """
    return PerformanceMonitor.get_metrics(window_minutes)


@app.get("/metrics/resources", response_model=Dict)
async def get_resource_metrics():
    """Get resource metrics.

    :return: Resource metrics
    :rtype: Dict
    """
    return ResourceMonitor.collect_metrics()


@app.get("/metrics/data-quality", response_model=Dict)
async def get_data_quality_metrics():
    """Get data quality metrics.

    :return: Data quality metrics
    :rtype: Dict
    """
    return DataQualityMonitor.get_metrics()


@app.get("/metrics/drift", response_model=Dict)
async def get_drift_metrics():
    """Get drift metrics.

    :return: Drift metrics
    :rtype: Dict
    """
    return DriftDetector.get_drift_report()


@app.get("/health")
async def health_check():
    """Health check endpoint.

    :return: Health status
    :rtype: Dict
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }
