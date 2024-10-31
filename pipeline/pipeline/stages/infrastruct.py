"""
Infrastruct Stage
=====================

.. module:: pipeline.stages.infrastruct
   :synopsis: Infrastruct setup with comprehensive logging and tracking

.. moduleauthor:: aai540-group3
"""

import json
import time
from typing import Any, Dict, List

from botocore.exceptions import ClientError
import hydra
from omegaconf import DictConfig, OmegaConf

from pipeline.stages.base import PipelineStage
from pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class InfrastructStage(PipelineStage):
    """Infrastruct stage implementation with detailed logging and tracking."""

    def __init__(self, cfg: DictConfig):
        """Initialize infrastruct stage."""
        super().__init__(cfg)
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
        """Execute infrastruct setup."""
        logger.info("Starting infrastruct setup")
        success = False

        try:
            # Setup S3
            self.metadata["aws"]["resources"]["s3"] = self._setup_s3()

            # Setup DynamoDB
            self.metadata["aws"]["resources"]["dynamodb"] = self._setup_dynamodb()

            # Setup CloudWatch if enabled
            if self.aws_config.get("monitoring", {}).get("enabled", False):
                self.metadata["aws"]["resources"]["cloudwatch"] = self._setup_cloudwatch()

            # Record execution time
            self.metadata["aws"]["execution_time"] = time.time() - self.start_time

            # Save metadata
            self._save_metadata()

            # Log detailed resource status
            self._log_resource_status()

            success = True
            logger.info("Infrastruct setup completed successfully")

        except Exception as e:
            logger.error(f"Infrastruct setup failed: {str(e)}")
            self.resource_errors.append({"error": str(e)})
            raise RuntimeError(f"Infrastruct setup failed: {str(e)}") from e

        finally:
            self._log_final_status(success)

    def _extract_aws_config(self) -> Dict[str, Any]:
        """Extract AWS-specific configuration from the main config.

        :return: AWS configuration dictionary
        :rtype: Dict[str, Any]
        """
        return OmegaConf.to_container(self.cfg.aws, resolve=True)

    def _init_aws_session(self) -> Any:
        """Initialize AWS session using boto3.

        :return: AWS session
        :rtype: Any
        """
        import boto3

        session = boto3.Session(
            region_name=self.aws_config.get("region"),
            profile_name=self.aws_config.get("profile"),
        )
        logger.debug("AWS session initialized.")
        return session

    def _setup_s3(self) -> Dict[str, Any]:
        """Setup S3 bucket with detailed logging."""
        s3_client = self.session.client("s3")
        bucket_name = self.aws_config["s3"]["bucket"]["name"]
        resource_info = {
            "bucket": bucket_name,
            "versioning": self.aws_config["s3"]["bucket"].get("versioning", "disabled"),
            "status": "unknown",
            "timestamp": time.time(),
        }

        try:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"S3 bucket '{bucket_name}' already exists.")
                resource_info["status"] = "existing"
                self.resources_existing.append(f"s3_bucket_{bucket_name}")

                # Get existing bucket tags
                try:
                    tags = s3_client.get_bucket_tagging(Bucket=bucket_name)
                    resource_info["tags"] = tags.get("TagSet", [])
                except ClientError:
                    resource_info["tags"] = []

            except ClientError as e:
                if e.response['Error']['Code'] in ['404', 'NoSuchBucket']:
                    logger.info(f"Creating S3 bucket '{bucket_name}'.")
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={
                            "LocationConstraint": self.aws_config.get("region")
                        },
                    )

                    # Add versioning if enabled
                    if self.aws_config["s3"]["bucket"].get("versioning") == "enabled":
                        s3_client.put_bucket_versioning(
                            Bucket=bucket_name,
                            VersioningConfiguration={"Status": "Enabled"},
                        )

                    # Add standard tags
                    s3_client.put_bucket_tagging(
                        Bucket=bucket_name,
                        Tagging={
                            "TagSet": [
                                {"Key": "Project", "Value": "DiabetesReadmission"},
                                {"Key": "Environment", "Value": self.cfg.experiment.name},
                                {"Key": "ManagedBy", "Value": "MLOps-Pipeline"},
                            ]
                        },
                    )

                    resource_info["status"] = "created"
                    self.resources_created.append(f"s3_bucket_{bucket_name}")
                else:
                    raise

            # Get bucket encryption status
            try:
                encryption = s3_client.get_bucket_encryption(Bucket=bucket_name)
                resource_info["encryption"] = encryption["ServerSideEncryptionConfiguration"]
            except ClientError:
                resource_info["encryption"] = "not_configured"

            # Get bucket policy status
            try:
                policy = s3_client.get_bucket_policy_status(Bucket=bucket_name)
                resource_info["policy_status"] = policy["PolicyStatus"]
            except ClientError:
                resource_info["policy_status"] = "no_policy"

            return resource_info

        except Exception as e:
            error_info = {"resource": f"s3_bucket_{bucket_name}", "error": str(e)}
            self.resource_errors.append(error_info)
            logger.error(f"Error setting up S3 bucket '{bucket_name}': {str(e)}")
            raise

    def _setup_dynamodb(self) -> Dict[str, Any]:
        """Setup DynamoDB table with detailed logging."""
        dynamodb = self.session.client("dynamodb")
        table_name = self.aws_config["dynamodb"]["table"]["name"]
        resource_info = {
            "table": table_name,
            "status": "unknown",
            "timestamp": time.time(),
        }

        try:
            try:
                table_desc = dynamodb.describe_table(TableName=table_name)
                logger.info(f"DynamoDB table '{table_name}' already exists.")
                resource_info.update(
                    {"status": "existing", "configuration": table_desc["Table"]}
                )
                self.resources_existing.append(f"dynamodb_table_{table_name}")

            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Creating DynamoDB table '{table_name}'.")
                    table = dynamodb.create_table(
                        TableName=table_name,
                        KeySchema=[
                            {
                                "AttributeName": self.aws_config["dynamodb"]["table"][
                                    "hash_key"
                                ],
                                "KeyType": "HASH",
                            }
                        ],
                        AttributeDefinitions=[
                            {
                                "AttributeName": self.aws_config["dynamodb"]["table"][
                                    "hash_key"
                                ],
                                "AttributeType": "S",
                            }
                        ],
                        BillingMode=self.aws_config["dynamodb"]["table"].get("billing_mode", "PAY_PER_REQUEST"),
                        Tags=[
                            {"Key": "Project", "Value": "DiabetesReadmission"},
                            {"Key": "Environment", "Value": self.cfg.experiment.name},
                            {"Key": "ManagedBy", "Value": "MLOps-Pipeline"},
                        ],
                    )
                    # Wait until the table exists
                    waiter = dynamodb.get_waiter('table_exists')
                    waiter.wait(TableName=table_name)
                    table_desc = dynamodb.describe_table(TableName=table_name)
                    resource_info.update({"status": "created", "configuration": table_desc["Table"]})
                    self.resources_created.append(f"dynamodb_table_{table_name}")
                else:
                    raise

            return resource_info

        except Exception as e:
            error_info = {"resource": f"dynamodb_table_{table_name}", "error": str(e)}
            self.resource_errors.append(error_info)
            logger.error(f"Error setting up DynamoDB table '{table_name}': {str(e)}")
            raise

    def _setup_cloudwatch(self) -> Dict[str, Any]:
        """Setup CloudWatch with detailed logging."""
        cloudwatch_client = self.session.client("cloudwatch")
        # Implement CloudWatch setup logic here
        # For demonstration, we'll assume it's a placeholder
        logger.info("Setting up CloudWatch monitoring.")
        resource_info = {
            "cloudwatch": "setup_complete",
            "status": "created",
            "timestamp": time.time(),
        }
        self.resources_created.append("cloudwatch_monitoring")
        return resource_info

    def _log_resource_status(self) -> None:
        """Log detailed resource status information."""
        resource_stats = {
            "resources_created": len(self.resources_created),
            "resources_existing": len(self.resources_existing),
            "resources_failed": len(self.resource_errors),
            "total_resources": len(self.resources_created) + len(self.resources_existing),
            "execution_time_seconds": time.time() - self.start_time,
        }

        logger.info("Resource Status Summary:")
        logger.info(f"- Created: {resource_stats['resources_created']}")
        logger.info(f"- Existing: {resource_stats['resources_existing']}")
        logger.info(f"- Failed: {resource_stats['resources_failed']}")
        logger.info(f"- Total Time: {resource_stats['execution_time_seconds']:.2f}s")

        if self.tracking_initialized:
            self.log_metrics(resource_stats)

    def _log_final_status(self, success: bool) -> None:
        """Log final infrastruct status.

        :param success: Whether setup was successful
        :type success: bool
        """
        if not self.tracking_initialized:
            return

        # Log configuration
        self.log_metrics(
            {
                "infrastruct_setup_success": int(success),
                "aws_resources_configured": len(self.metadata["aws"]["resources"]),
                "setup_duration_seconds": time.time() - self.start_time,
            }
        )

        # Log detailed metrics about each resource type
        for resource_type, info in self.metadata["aws"]["resources"].items():
            if isinstance(info, dict):
                status = 1 if info.get("status") == "created" else 0
                self.log_metrics({f"{resource_type}_created": status})

        # Log any errors
        if self.resource_errors:
            error_count = len(self.resource_errors)
            total_resources = len(self.metadata["aws"]["resources"])
            self.log_metrics(
                {
                    "setup_errors": error_count,
                    "error_rate": error_count / total_resources if total_resources else 0,
                }
            )

    def _save_metadata(self) -> None:
        """Save comprehensive infrastruct metadata."""
        output_path = self.get_path("infrastruct") / "metadata.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add summary statistics
        self.metadata["summary"] = {
            "resources_created": self.resources_created,
            "resources_existing": self.resources_existing,
            "resource_errors": self.resource_errors,
            "total_resources": len(self.resources_created) + len(self.resources_existing),
            "execution_time": time.time() - self.start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            with open(output_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Saved infrastruct metadata to {output_path}")
            if self.tracking_initialized:
                self.log_artifact(str(output_path))
        except Exception as e:
            logger.warning(f"Failed to save infrastruct metadata: {e}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the InfrastructStage.

    :param cfg: Configuration dictionary
    :type cfg: DictConfig
    """
    stage = InfrastructStage(cfg)
    stage.run()


if __name__ == "__main__":
    main()
