import boto3
from loguru import logger

from .stage import Stage


class Infrastruct(Stage):
    """Pipeline stage for setting up AWS infrastructure."""

    def run(self):
        """Set up AWS infrastructure."""
        # INITIALIZE AWS CLIENTS
        session = boto3.Session(region_name=self.cfg.aws.region)
        s3_client = session.client("s3")

        resources = []

        # HANDLE S3 BUCKET
        try:
            # CHECK IF BUCKET EXISTS
            s3_client.head_bucket(Bucket=self.cfg.aws.bucket_name)
            logger.info(f"BUCKET EXISTS: {self.cfg.aws.bucket_name}")
            resources.append({"type": "s3_bucket", "name": self.cfg.aws.bucket_name, "status": "exists"})

        except s3_client.exceptions.NoSuchBucket:
            # CREATE BUCKET IF IT DOES NOT EXIST
            s3_client.create_bucket(
                Bucket=self.cfg.aws.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.cfg.aws.region},
            )
            logger.info(f"BUCKET CREATED: {self.cfg.aws.bucket_name}")
            resources.append({
                "type": "s3_bucket",
                "name": self.cfg.aws.bucket_name,
                "status": "created",
            })  # fmt: skip

        # Save metrics
        self.save_metrics(
            "metrics", {
                "region": self.cfg.aws.region,
                "resources": resources,
                "status": "cleaned_up",
            },
        )  # fmt: skip
