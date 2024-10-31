"""
AWS Integration
============

.. module:: pipeline.cloud.aws_integration
   :synopsis: AWS service integration for monitoring

.. moduleauthor:: aai540-group3
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from omegaconf import DictConfig

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AWSIntegration:
    """AWS service integration for monitoring.

    :param cfg: AWS configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize AWS integration.

        :param cfg: AWS configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize AWS service clients."""
        session = boto3.Session(
            profile_name=self.cfg.profile, region_name=self.cfg.region
        )

        self.cloudwatch = session.client("cloudwatch")
        self.logs = session.client("logs")
        self.s3 = session.client("s3")
        self.sagemaker = session.client("sagemaker")

    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        dimensions: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Log metric to CloudWatch.

        :param metric_name: Metric name
        :type metric_name: str
        :param value: Metric value
        :type value: float
        :param unit: Metric unit
        :type unit: str
        :param dimensions: Metric dimensions
        :type dimensions: Optional[List[Dict[str, str]]]
        :raises ClientError: If CloudWatch API call fails
        """
        try:
            metric_data = {
                "MetricName": metric_name,
                "Value": value,
                "Unit": unit,
                "Timestamp": datetime.utcnow(),
            }

            if dimensions:
                metric_data["Dimensions"] = dimensions

            self.cloudwatch.put_metric_data(
                Namespace=self.cfg.cloudwatch.namespace, MetricData=[metric_data]
            )

        except ClientError as e:
            logger.error(f"Failed to log metric to CloudWatch: {e}")
            raise

    def create_log_stream(
        self, log_stream_name: str, log_group_name: Optional[str] = None
    ) -> None:
        """Create CloudWatch log stream.

        :param log_stream_name: Log stream name
        :type log_stream_name: str
        :param log_group_name: Log group name
        :type log_group_name: Optional[str]
        :raises ClientError: If CloudWatch Logs API call fails
        """
        try:
            log_group = log_group_name or self.cfg.cloudwatch.log_group
            self.logs.create_log_stream(
                logGroupName=log_group, logStreamName=log_stream_name
            )
        except ClientError as e:
            logger.error(f"Failed to create log stream: {e}")
            raise

    def put_log_events(
        self,
        log_stream_name: str,
        events: List[Dict],
        log_group_name: Optional[str] = None,
    ) -> None:
        """Put events to CloudWatch log stream.

        :param log_stream_name: Log stream name
        :type log_stream_name: str
        :param events: Log events
        :type events: List[Dict]
        :param log_group_name: Log group name
        :type log_group_name: Optional[str]
        :raises ClientError: If CloudWatch Logs API call fails
        """
        try:
            log_group = log_group_name or self.cfg.cloudwatch.log_group
            self.logs.put_log_events(
                logGroupName=log_group,
                logStreamName=log_stream_name,
                logEvents=[
                    {
                        "timestamp": int(datetime.utcnow().timestamp() * 1000),
                        "message": json.dumps(event),
                    }
                    for event in events
                ],
            )
        except ClientError as e:
            logger.error(f"Failed to put log events: {e}")
            raise

    def upload_artifact(
        self, file_path: str, s3_key: str, metadata: Optional[Dict] = None
    ) -> None:
        """Upload artifact to S3.

        :param file_path: Local file path
        :type file_path: str
        :param s3_key: S3 key
        :type s3_key: str
        :param metadata: File metadata
        :type metadata: Optional[Dict]
        :raises ClientError: If S3 API call fails
        """
        try:
            extra_args = {"Metadata": metadata} if metadata else {}
            self.s3.upload_file(
                file_path,
                self.cfg.s3.bucket,
                f"{self.cfg.s3.prefix}/{s3_key}",
                ExtraArgs=extra_args,
            )
        except ClientError as e:
            logger.error(f"Failed to upload artifact to S3: {e}")
            raise

    def create_model_monitor(
        self,
        endpoint_name: str,
        schedule_expression: str,
        monitoring_type: str = "DataQuality",
    ) -> None:
        """Create SageMaker model monitor.

        :param endpoint_name: SageMaker endpoint name
        :type endpoint_name: str
        :param schedule_expression: Schedule expression
        :type schedule_expression: str
        :param monitoring_type: Type of monitoring
        :type monitoring_type: str
        :raises ClientError: If SageMaker API call fails
        """
        try:
            self.sagemaker.create_monitoring_schedule(
                MonitoringScheduleName=f"{endpoint_name}-monitoring",
                MonitoringScheduleConfig={
                    "ScheduleConfig": {"ScheduleExpression": schedule_expression},
                    "MonitoringJobDefinition": {
                        "MonitoringType": monitoring_type,
                        "EndpointName": endpoint_name,
                    },
                },
            )
        except ClientError as e:
            logger.error(f"Failed to create model monitor: {e}")
            raise

    def get_monitoring_metrics(
        self,
        endpoint_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
    ) -> Dict:
        """Get monitoring metrics from CloudWatch.

        :param endpoint_name: SageMaker endpoint name
        :type endpoint_name: str
        :param start_time: Start time
        :type start_time: datetime
        :param end_time: End time
        :type end_time: datetime
        :param period: Period in seconds
        :type period: int
        :return: Monitoring metrics
        :rtype: Dict
        :raises ClientError: If CloudWatch API call fails
        """
        try:
            metrics = {}
            for metric in self.cfg.cloudwatch.metrics:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=self.cfg.cloudwatch.namespace,
                    MetricName=metric["name"],
                    Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=period,
                    Statistics=["Average", "Maximum", "Minimum"],
                )
                metrics[metric["name"]] = response["Datapoints"]

            return metrics

        except ClientError as e:
            logger.error(f"Failed to get monitoring metrics: {e}")
            raise
