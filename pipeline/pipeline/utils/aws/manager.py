"""
AWS Integration Utilities
=======================

.. module:: pipeline.utils.aws.manager
   :synopsis: AWS service management and integration

.. moduleauthor:: aai540-group3
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from loguru import logger
from omegaconf import DictConfig


class AWSManager:
    """Base manager for AWS services."""

    def __init__(self, cfg: DictConfig):
        """Initialize AWS manager.

        :param cfg: AWS configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.session = self._create_session()

    def _create_session(self) -> boto3.Session:
        """Create AWS session with configuration.

        :return: Configured AWS session
        :rtype: boto3.Session
        """
        try:
            return boto3.Session(
                aws_access_key_id=self.cfg.credentials.get("access_key"),
                aws_secret_access_key=self.cfg.credentials.get("secret_key"),
                region_name=self.cfg.region,
                profile_name=self.cfg.get("profile"),
            )
        except Exception as e:
            logger.error(f"Failed to create AWS session: {e}")
            raise


class S3Manager(AWSManager):
    """Manager for S3 operations."""

    def __init__(self, cfg: DictConfig):
        """Initialize S3 manager.

        :param cfg: AWS configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.s3_client = self.session.client("s3")
        self.bucket = self.cfg.s3.bucket.name

    def upload_file(self, local_path: Union[str, Path], s3_key: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload file to S3.

        :param local_path: Local file path
        :type local_path: Union[str, Path]
        :param s3_key: S3 object key
        :type s3_key: str
        :param metadata: Optional metadata
        :type metadata: Optional[Dict[str, str]]
        :return: Success status
        :rtype: bool
        """
        try:
            extra_args = {"Metadata": metadata} if metadata else {}
            self.s3_client.upload_file(str(local_path), self.bucket, s3_key, ExtraArgs=extra_args)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> bool:
        """Download file from S3.

        :param s3_key: S3 object key
        :type s3_key: str
        :param local_path: Local file path
        :type local_path: Union[str, Path]
        :return: Success status
        :rtype: bool
        """
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            logger.info(f"Downloaded s3://{self.bucket}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False

    def list_objects(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in S3 bucket.

        :param prefix: Object prefix filter
        :type prefix: str
        :return: List of object information
        :rtype: List[Dict[str, Any]]
        """
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            objects = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    objects.extend(page["Contents"])
            return objects
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []

    def delete_object(self, s3_key: str) -> bool:
        """Delete object from S3.

        :param s3_key: S3 object key
        :type s3_key: str
        :return: Success status
        :rtype: bool
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return False

    def get_object_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata.

        :param s3_key: S3 object key
        :type s3_key: str
        :return: Object metadata
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "metadata": response.get("Metadata", {}),
                "version_id": response.get("VersionId"),
            }
        except Exception as e:
            logger.error(f"Failed to get S3 object metadata: {e}")
            return None


class DynamoDBManager(AWSManager):
    """Manager for DynamoDB operations."""

    def __init__(self, cfg: DictConfig):
        """Initialize DynamoDB manager.

        :param cfg: AWS configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.dynamodb = self.session.resource("dynamodb")
        self.table_name = self.cfg.dynamodb.table.name
        self.table = self.dynamodb.Table(self.table_name)

    def put_item(self, item: Dict[str, Any]) -> bool:
        """Put item in DynamoDB table.

        :param item: Item to store
        :type item: Dict[str, Any]
        :return: Success status
        :rtype: bool
        """
        try:
            self.table.put_item(Item=item)
            return True
        except Exception as e:
            logger.error(f"Failed to put item in DynamoDB: {e}")
            return False

    def get_item(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get item from DynamoDB table.

        :param key: Item key
        :type key: Dict[str, Any]
        :return: Item data
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            response = self.table.get_item(Key=key)
            return response.get("Item")
        except Exception as e:
            logger.error(f"Failed to get item from DynamoDB: {e}")
            return None

    def delete_item(self, key: Dict[str, Any]) -> bool:
        """Delete item from DynamoDB table.

        :param key: Item key
        :type key: Dict[str, Any]
        :return: Success status
        :rtype: bool
        """
        try:
            self.table.delete_item(Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete item from DynamoDB: {e}")
            return False


class CloudWatchManager(AWSManager):
    """Manager for CloudWatch operations."""

    def __init__(self, cfg: DictConfig):
        """Initialize CloudWatch manager.

        :param cfg: AWS configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.cloudwatch = self.session.client("cloudwatch")
        self.namespace = self.cfg.cloudwatch.namespace

    def put_metric(
        self, metric_name: str, value: float, unit: str = "None", dimensions: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """Put metric data point.

        :param metric_name: Metric name
        :type metric_name: str
        :param value: Metric value
        :type value: float
        :param unit: Metric unit
        :type unit: str
        :param dimensions: Metric dimensions
        :type dimensions: Optional[List[Dict[str, str]]]
        :return: Success status
        :rtype: bool
        """
        try:
            metric_data = {
                "MetricName": metric_name,
                "Value": value,
                "Unit": unit,
            }
            if dimensions:
                metric_data["Dimensions"] = dimensions

            self.cloudwatch.put_metric_data(Namespace=self.namespace, MetricData=[metric_data])
            return True
        except Exception as e:
            logger.error(f"Failed to put CloudWatch metric: {e}")
            return False

    def get_metric_statistics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: Optional[List[str]] = None,
        dimensions: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get metric statistics.

        :param metric_name: Metric name
        :type metric_name: str
        :param start_time: Start time
        :type start_time: datetime
        :param end_time: End time
        :type end_time: datetime
        :param period: Period in seconds
        :type period: int
        :param statistics: Statistics to retrieve
        :type statistics: Optional[List[str]]
        :param dimensions: Metric dimensions
        :type dimensions: Optional[List[Dict[str, str]]]
        :return: Metric statistics
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            statistics = statistics or ["Average", "Maximum", "Minimum", "Sum"]
            response = self.cloudwatch.get_metric_statistics(
                Namespace=self.namespace,
                MetricName=metric_name,
                Dimensions=dimensions or [],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=statistics,
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get CloudWatch metric statistics: {e}")
            return None
