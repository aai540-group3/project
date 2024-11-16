"""
Infrastruct Stage
=================

.. module:: pipeline.stages.infrastruct
   :synopsis: This module manages the setup of AWS infrastructure resources

.. moduleauthor:: aai540-group3
"""

import boto3
from loguru import logger

from .stage import Stage


class Infrastruct(Stage):
    """Pipeline stage for setting up AWS infrastructure."""

    def run(self):
        """Set up AWS infrastructure.

        This method performs the following steps:
            1. Initializes AWS session and S3 client.
            2. Checks if the specified S3 bucket exists.
            3. Creates the S3 bucket if it does not exist.
            4. Logs the status of each AWS resource.
            5. Saves setup metrics.
        """
        session = boto3.Session(region_name=self.cfg.aws.region)
        s3_client = session.client("s3")

        resources = []

        try:
            s3_client.head_bucket(Bucket=self.cfg.aws.bucket_name)
            logger.info(f"BUCKET EXISTS: {self.cfg.aws.bucket_name}")
            resources.append({
                "type": "s3_bucket", 
                "name": self.cfg.aws.bucket_name, 
                "status": "exists",
            })  # fmt: off

        except s3_client.exceptions.NoSuchBucket:
            s3_client.create_bucket(
                Bucket=self.cfg.aws.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.cfg.aws.region},
            )
            logger.info(f"BUCKET CREATED: {self.cfg.aws.bucket_name}")
            resources.append({
                "type": "s3_bucket", 
                "name": self.cfg.aws.bucket_name, 
                "status": "created",
            })  # fmt: off

        # Save metrics
        self.save_metrics(
            "metrics", {
                "region": self.cfg.aws.region,
                "resources": resources,
            },
        )  # fmt: off
