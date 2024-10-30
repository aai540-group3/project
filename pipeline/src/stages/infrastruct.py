import logging
from pathlib import Path

import boto3

from base import PipelineStage

logger = logging.getLogger(__name__)

class InfrastructureStage(PipelineStage):
    """Infrastructure setup stage."""

    def run(self) -> None:
        """Execute infrastructure setup."""
        self.tracker.start_run(run_name="infrastructure")

        try:
            # Initialize AWS session
            session = boto3.Session(
                region_name=self.cfg.aws.region,
                profile_name=self.cfg.aws.profile
            )

            # Setup IAM
            self._setup_iam(session)

            # Setup S3
            self._setup_s3(session)

            # Setup DynamoDB
            self._setup_dynamodb(session)

            # Setup monitoring
            self._setup_monitoring(session)

            # Save infrastructure metadata
            self._save_metadata()

            logger.info("Infrastructure setup completed successfully")

        finally:
            self.tracker.end_run()

    def _setup_iam(self, session: boto3.Session) -> None:
        """Setup IAM users and groups."""
        iam = session.client('iam')

        # Create group
        group_name = self.cfg.aws.iam.group.name
        try:
            iam.create_group(GroupName=group_name)
            logger.info(f"Created IAM group: {group_name}")

            # Attach policies
            for policy in self.cfg.aws.iam.group.policies:
                iam.attach_group_policy(
                    GroupName=group_name,
                    PolicyArn=f"arn:aws:iam::aws:policy/{policy}"
                )
        except iam.exceptions.EntityAlreadyExistsException:
            logger.info(f"IAM group already exists: {group_name}")

        # Create users
        for email in self.cfg.aws.iam.users.emails.split(','):
            username = email.split('@')[0]
            try:
                iam.create_user(UserName=username)
                iam.add_user_to_group(
                    GroupName=group_name,
                    UserName=username
                )
                logger.info(f"Created IAM user: {username}")

                # Create access keys
                if self.cfg.aws.iam.users.create_access_keys:
                    response = iam.create_access_key(UserName=username)
                    logger.info(
                        f"Access key created for {username}: "
                        f"{response['AccessKey']['AccessKeyId']}"
                    )
            except iam.exceptions.EntityAlreadyExistsException:
                logger.info(f"IAM user already exists: {username}")

    def _setup_s3(self, session: boto3.Session) -> None:
        """Setup S3 bucket and folders."""
        s3 = session.client('s3')

        # Create bucket
        bucket_name = self.cfg.aws.s3.bucket.name
        try:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.cfg.aws.region
                }
            )
            logger.info(f"Created S3 bucket: {bucket_name}")

            # Enable versioning
            s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            # Block public access
            s3.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True
                }
            )

            # Create folders
            for key in self.cfg.aws.s3.bucket.keys.split(','):
                s3.put_object(
                    Bucket=bucket_name,
                    Key=f"{key.strip()}/"
                )
                logger.info(f"Created S3 folder: {key}/")

        except s3.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"S3 bucket already exists: {bucket_name}")

    def _setup_dynamodb(self, session: boto3.Session) -> None:
        """Setup DynamoDB table."""
        dynamodb = session.client('dynamodb')

        # Create table
        table_name = self.cfg.aws.dynamodb.table.name
        try:
            dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'id',
                        'KeyType': 'HASH'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'id',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            logger.info(f"Created DynamoDB table: {table_name}")

        except dynamodb.exceptions.ResourceInUseException:
            logger.info(f"DynamoDB table already exists: {table_name}")

    def _setup_monitoring(self, session: boto3.Session) -> None:
        """Setup monitoring and alerts."""
        cloudwatch = session.client('cloudwatch')
        sns = session.client('sns')

        # Create SNS topic
        topic_name = f"{self.cfg.aws.sns.topics[0].name}"
        try:
            response = sns.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            logger.info(f"Created SNS topic: {topic_name}")

            # Subscribe emails
            for email in self.cfg.aws.iam.users.emails.split(','):
                sns.subscribe(
                    TopicArn=topic_arn,
                    Protocol='email',
                    Endpoint=email
                )
                logger.info(f"Subscribed {email} to SNS topic")

        except Exception as e:
            logger.error(f"Error setting up SNS: {str(e)}")

    def _save_metadata(self) -> None:
        """Save infrastructure metadata."""
        metadata = {
            'aws_region': self.cfg.aws.region,
            'bucket_name': self.cfg.aws.s3.bucket.name,
            'dynamodb_table': self.cfg.aws.dynamodb.table.name,
            'iam_group': self.cfg.aws.iam.group.name,
            'users': [
                email.split('@')[0]
                for email in self.cfg.aws.iam.users.emails.split(',')
            ]
        }

        metadata_path = Path(self.cfg.paths.infrastructure) / 'metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved infrastructure metadata to {metadata_path}")
