"""
Infrastruct Stage
=================

.. module:: pipeline.stages.infrastruct
   :synopsis: Infrastructure setup with comprehensive logging and tracking, including AWS services.

.. moduleauthor:: aai540-group3
"""

import json
import time
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError
from loguru import logger
from omegaconf import OmegaConf

from pipeline.stages.base import PipelineStage


class InfrastructStage(PipelineStage):
    """Infrastructure setup stage implementation with detailed logging and tracking."""

    def __init__(self):
        """Initialize the infrastructure stage.

        Sets up AWS configuration, initializes an AWS session, and prepares metadata tracking.
        """
        super().__init__()
        self.aws_config = self._extract_aws_config()
        self.session = self._init_aws_session()
        self.start_time = time.time()
        self.resources_created: List[str] = []
        self.resources_existing: List[str] = []
        self.resource_errors: List[Dict[str, str]] = []

        # Initialize metadata structure
        self.metadata: Dict[str, Any] = {
            "aws": {
                "region": self.aws_config.get("region", "unknown"),
                "profile": self.aws_config.get("profile", "default"),
                "execution_time": None,
                "resources": {},
                "status": {"created": [], "existing": [], "failed": []},
            },
            "config": OmegaConf.to_container(self.stage_config, resolve=True),
        }

    def run(self) -> None:
        """Execute the infrastructure setup by provisioning AWS services.

        Sets up S3, DynamoDB, and optionally CloudWatch services. Logs the
        resources and any encountered errors.

        :raises RuntimeError: If infrastructure setup encounters an error.
        """
        logger.info("Starting infrastructure setup")

        try:
            self.metadata["aws"]["resources"]["s3"] = self._setup_s3()
            self.metadata["aws"]["resources"]["dynamodb"] = self._setup_dynamodb()

            if self.aws_config.get("monitoring", {}).get("enabled", False):
                self.metadata["aws"]["resources"]["cloudwatch"] = self._setup_cloudwatch()

            self.metadata["aws"]["execution_time"] = time.time() - self.start_time
            self._save_metadata()
            self._log_resource_status()
            logger.info("Infrastructure setup completed successfully")

        except Exception as e:
            logger.error(f"Infrastructure setup failed: {str(e)}")
            self.resource_errors.append({"error": str(e)})
            raise RuntimeError(f"Infrastructure setup failed: {str(e)}") from e

    def _extract_aws_config(self) -> Dict[str, Any]:
        """Extract AWS-specific configuration from the main configuration file.

        :return: Extracted AWS configuration as a dictionary.
        :rtype: Dict[str, Any]
        """
        return OmegaConf.to_container(self.cfg.aws, resolve=True)

    def _init_aws_session(self) -> Any:
        """Initialize an AWS session using boto3.

        :return: Initialized AWS session object.
        :rtype: boto3.Session
        """
        session = boto3.Session(
            region_name=self.aws_config.get("region"),
            profile_name=self.aws_config.get("profile"),
        )
        logger.debug("AWS session initialized.")
        return session

    def _setup_s3(self) -> Dict[str, Any]:
        """Set up an S3 bucket with logging, versioning, and encryption options.

        :return: Information about the S3 bucket setup process.
        :rtype: Dict[str, Any]
        :raises Exception: If an error occurs during the S3 bucket setup.
        """
        s3_client = self.session.client("s3")
        bucket_name = self.aws_config["s3"]["bucket"]["name"]
        resource_info = {
            "bucket": bucket_name,
            "versioning": self.aws_config["s3"]["bucket"].get("versioning", "disabled"),
            "encryption": self.aws_config["s3"]["bucket"].get("encryption", False),
            "status": "unknown",
            "timestamp": time.time(),
        }

        try:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"S3 bucket '{bucket_name}' already exists.")
                resource_info["status"] = "existing"
                self.resources_existing.append(f"s3_bucket_{bucket_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] in ["404", "NoSuchBucket"]:
                    logger.info(f"Creating S3 bucket '{bucket_name}'.")
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.aws_config.get("region")},
                    )
                    if self.aws_config["s3"]["bucket"].get("versioning") == "enabled":
                        s3_client.put_bucket_versioning(
                            Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
                        )
                    if self.aws_config["s3"]["bucket"].get("encryption"):
                        s3_client.put_bucket_encryption(
                            Bucket=bucket_name,
                            ServerSideEncryptionConfiguration={
                                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
                            },
                        )
                    self.resources_created.append(f"s3_bucket_{bucket_name}")
                    resource_info["status"] = "created"
                else:
                    raise
            return resource_info

        except Exception as e:
            logger.error(f"Error setting up S3 bucket '{bucket_name}': {str(e)}")
            raise

    def _setup_dynamodb(self) -> Dict[str, Any]:
        """Set up a DynamoDB table with logging.

        :return: Information about the DynamoDB setup process.
        :rtype: Dict[str, Any]
        :raises Exception: If an error occurs during the DynamoDB setup.
        """
        dynamodb = self.session.client("dynamodb")
        table_name = self.aws_config["dynamodb"]["table"]["name"]
        resource_info = {
            "table": table_name,
            "status": "unknown",
            "timestamp": time.time(),
        }

        try:
            try:
                dynamodb.describe_table(TableName=table_name)
                resource_info["status"] = "existing"
                self.resources_existing.append(f"dynamodb_table_{table_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.info(f"Creating DynamoDB table '{table_name}'.")
                    dynamodb.create_table(
                        TableName=table_name,
                        KeySchema=[
                            {"AttributeName": self.aws_config["dynamodb"]["table"]["hash_key"], "KeyType": "HASH"}
                        ],
                        AttributeDefinitions=[
                            {"AttributeName": self.aws_config["dynamodb"]["table"]["hash_key"], "AttributeType": "S"}
                        ],
                        BillingMode=self.aws_config["dynamodb"]["table"].get("billing_mode", "PAY_PER_REQUEST"),
                    )
                    waiter = dynamodb.get_waiter("table_exists")
                    waiter.wait(TableName=table_name)
                    resource_info["status"] = "created"
                    self.resources_created.append(f"dynamodb_table_{table_name}")
                else:
                    raise
            return resource_info

        except Exception as e:
            logger.error(f"Error setting up DynamoDB table '{table_name}': {str(e)}")
            raise

    def _setup_cloudwatch(self) -> Dict[str, Any]:
        """Set up CloudWatch monitoring with metrics and alarms.

        :return: Information about the CloudWatch setup process.
        :rtype: Dict[str, Any]
        :raises Exception: If an error occurs during the CloudWatch setup.
        """
        cloudwatch_client = self.session.client("cloudwatch")
        namespace = self.aws_config["monitoring"].get("namespace", "DefaultNamespace")
        metrics = self.aws_config["monitoring"].get("metrics", [])
        resource_info = {"namespace": namespace, "metrics": [], "status": "unknown", "timestamp": time.time()}

        try:
            for metric in metrics:
                alarm_name = f"{metric['name']}_alarm"
                cloudwatch_client.put_metric_alarm(
                    AlarmName=alarm_name,
                    MetricName=metric["name"],
                    Namespace=namespace,
                    Statistic=metric.get("statistic", "Average"),
                    Period=metric.get("period", 300),
                    EvaluationPeriods=metric.get("evaluation_periods", 2),
                    Threshold=metric.get("threshold", 80),
                    ComparisonOperator=metric.get("comparison_operator", "GreaterThanThreshold"),
                )
                resource_info["metrics"].append(alarm_name)

            resource_info["status"] = "created"
            self.resources_created.append("cloudwatch_monitoring")
            return resource_info

        except Exception as e:
            logger.error(f"Error setting up CloudWatch monitoring: {str(e)}")
            raise

    def _save_metadata(self) -> None:
        """Save comprehensive infrastructure metadata to a JSON file.

        :raises Exception: If saving metadata encounters an error.
        """
        output_path = self.get_path("infrastruct") / "metadata.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata["summary"] = {
            "resources_created": self.resources_created,
            "resources_existing": self.resources_existing,
            "resource_errors": self.resource_errors,
            "execution_time": time.time() - self.start_time,
        }
        try:
            with open(output_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved infrastructure metadata to {output_path}")
            if self.tracking_initialized:
                self.log_artifact(str(output_path))
        except Exception as e:
            logger.warning(f"Failed to save infrastructure metadata: {e}")


if __name__ == "__main__":
    InfrastructStage().execute()
