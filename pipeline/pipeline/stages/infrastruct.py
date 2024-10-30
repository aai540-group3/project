"""Infrastructure setup stage for ML pipeline.

This module handles the setup of AWS infrastructure components including
S3 buckets, DynamoDB tables, and CloudWatch monitoring.
"""

import json
import logging
from pathlib import Path

import boto3
import hydra
from omegaconf import DictConfig

from pipeline.stages.base import PipelineStage

logger = logging.getLogger(__name__)


class InfrastructureStage(PipelineStage):
    """Infrastructure setup stage implementation.

    :param cfg: Stage configuration
    :type cfg: DictConfig
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.stage_name = "infrastruct"

    def run(self) -> None:
        """Execute infrastructure setup."""
        logger.info("Starting infrastructure setup")
        self.tracker.start_run(run_name=self.stage_name)

        try:
            # Initialize AWS session
            session = self._init_aws_session()
            metadata = self._init_metadata()

            # Setup S3
            metadata["aws"]["resources"]["s3"] = self._setup_s3(session)

            # Setup DynamoDB
            metadata["aws"]["resources"]["dynamodb"] = self._setup_dynamodb(session)

            # Setup CloudWatch
            if self.cfg.aws.monitoring.enabled:
                metadata["aws"]["resources"]["cloudwatch"] = self._setup_cloudwatch(
                    session
                )

            # Save metadata
            self._save_metadata(metadata)

            # Log success metrics
            self.tracker.log_metrics(
                {
                    "setup_success": 1.0,
                    "resources_created": len(metadata["aws"]["resources"]),
                }
            )

            logger.info("Infrastructure setup completed successfully")

        except Exception as e:
            logger.error(f"Infrastructure setup failed: {str(e)}")
            self.tracker.log_metrics({"setup_success": 0.0})
            raise
        finally:
            self.tracker.end_run()

    def _init_aws_session(self) -> boto3.Session:
        """Initialize AWS session.

        :return: Initialized AWS session
        :rtype: boto3.Session
        """
        return boto3.Session(
            region_name=self.cfg.aws.region, profile_name=self.cfg.aws.profile
        )

    def _init_metadata(self) -> dict:
        """Initialize metadata structure.

        :return: Initial metadata dictionary
        :rtype: dict
        """
        return {"aws": {"region": self.cfg.aws.region, "resources": {}}}

    def _setup_s3(self, session: boto3.Session) -> dict:
        """Setup S3 bucket.

        :param session: AWS session
        :type session: boto3.Session
        :return: S3 configuration metadata
        :rtype: dict
        """
        s3_client = session.client("s3")
        bucket_name = self.cfg.aws.s3.bucket.name

        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"S3 bucket {bucket_name} exists")
        except:
            logger.info(f"Creating S3 bucket {bucket_name}")
            s3_client.create_bucket(Bucket=bucket_name)

            # Configure bucket versioning
            if self.cfg.aws.s3.bucket.versioning == "enabled":
                s3_client.put_bucket_versioning(
                    Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
                )

            # Configure encryption
            if self.cfg.aws.s3.bucket.encryption:
                s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration={
                        "Rules": [
                            {
                                "ApplyServerSideEncryptionByDefault": {
                                    "SSEAlgorithm": "AES256"
                                }
                            }
                        ]
                    },
                )

        return {
            "bucket": bucket_name,
            "versioning": self.cfg.aws.s3.bucket.versioning,
            "encryption": self.cfg.aws.s3.bucket.encryption,
        }

    def _setup_dynamodb(self, session: boto3.Session) -> dict:
        """Setup DynamoDB table.

        :param session: AWS session
        :type session: boto3.Session
        :return: DynamoDB configuration metadata
        :rtype: dict
        """
        dynamodb = session.client("dynamodb")
        table_name = self.cfg.aws.dynamodb.table.name

        try:
            dynamodb.describe_table(TableName=table_name)
            logger.info(f"DynamoDB table {table_name} exists")
        except:
            logger.info(f"Creating DynamoDB table {table_name}")
            dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        "AttributeName": self.cfg.aws.dynamodb.table.hash_key,
                        "KeyType": "HASH",
                    }
                ],
                AttributeDefinitions=[
                    {
                        "AttributeName": self.cfg.aws.dynamodb.table.hash_key,
                        "AttributeType": "S",
                    }
                ],
                BillingMode=self.cfg.aws.dynamodb.table.billing_mode,
            )

            # Configure TTL if enabled
            if self.cfg.aws.dynamodb.table.ttl.enabled:
                dynamodb.update_time_to_live(
                    TableName=table_name,
                    TimeToLiveSpecification={
                        "Enabled": True,
                        "AttributeName": self.cfg.aws.dynamodb.table.ttl.attribute_name,
                    },
                )

        return {"table": table_name, "hash_key": self.cfg.aws.dynamodb.table.hash_key}

    def _setup_cloudwatch(self, session: boto3.Session) -> dict:
        """Setup CloudWatch monitoring.

        :param session: AWS session
        :type session: boto3.Session
        :return: CloudWatch configuration metadata
        :rtype: dict
        """
        cloudwatch = session.client("cloudwatch")
        namespace = self.cfg.aws.monitoring.namespace

        for metric in self.cfg.aws.monitoring.metrics:
            logger.info(f"Setting up CloudWatch metric: {metric['name']}")
            cloudwatch.put_metric_alarm(
                AlarmName=f"{metric['name']}_alarm",
                MetricName=metric["name"],
                Namespace=namespace,
                Statistic="Average",
                Period=300,
                EvaluationPeriods=2,
                Threshold=80,
                ComparisonOperator="GreaterThanThreshold",
            )

        return {"namespace": namespace, "metrics": self.cfg.aws.monitoring.metrics}

    def _save_metadata(self, metadata: dict) -> None:
        """Save infrastructure metadata.

        :param metadata: Metadata to save
        :type metadata: dict
        """
        output_path = Path(self.cfg.artifacts.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved infrastructure metadata to {output_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for infrastructure setup.

    :param cfg: Configuration object
    :type cfg: DictConfig
    """
    stage = InfrastructureStage(cfg)
    stage.run()


if __name__ == "__main__":
    main()
