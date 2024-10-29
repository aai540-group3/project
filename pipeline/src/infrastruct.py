#!/usr/bin/env python3
"""
pipeline.src.infrastruct

This script sets up AWS infrastructure resources such as IAM users and policies, SNS topics,
CloudWatch alarms, S3 buckets, and DynamoDB tables. It includes helper functions for creating,
configuring, and managing these resources as per the environment variables set in a .env file.

Attributes:
    AWS_REGION (str): AWS region for resources.
    EMAILS (list): List of email addresses for SNS subscriptions and notifications.
    INITIAL_PASSWORD (str): Initial IAM user password.
    GROUP_NAME (str): Name of IAM group.
    BUCKET_NAME (str): Name of S3 bucket.
    TABLE_NAME (str): Name of DynamoDB table.
    BUCKET_KEYS (list): List of S3 bucket keys to create.

Example:
    To run the setup, use:

        $ python infrastruct.py
"""

import json
import logging
import os
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration - Environment Variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMAILS = os.environ["EMAILS"].split(",")
INITIAL_PASSWORD = os.environ["INITIAL_PASSWORD"]
GROUP_NAME = os.environ["GROUP_NAME"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
BUCKET_KEYS = os.environ["BUCKET_KEYS"].split(",")

# AWS Clients and Resources Initialization
session = boto3.Session(region_name=AWS_REGION)
org_client = session.client("organizations")
iam_client = session.client("iam")
s3_client = session.client("s3")
budgets_client = session.client("budgets")
sns_client = session.client("sns")
cloudwatch_client = session.client("cloudwatch")
sts_client = session.client("sts")
dynamodb_resource = session.resource("dynamodb")

# Get AWS account ID
account_id = sts_client.get_caller_identity()["Account"]

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def create_iam_user(username: str, email: str):
    """Creates an IAM user with console access and a login profile.

    Args:
        username (str): Name of the IAM user to create.
        email (str): Email address associated with the IAM user.

    Raises:
        Exception: If user creation or login profile creation fails.
    """
    try:
        iam_client.get_user(UserName=username)
        logger.info(f"User {username} already exists.")
    except iam_client.exceptions.NoSuchEntityException:
        iam_client.create_user(
            UserName=username,
            Tags=[{"Key": "Environment", "Value": "Management"}, {"Key": "ManagedBy", "Value": "PythonScript"}],
        )
        logger.info(f"Created user {username}.")
        iam_client.create_login_profile(
            UserName=username, Password=INITIAL_PASSWORD, PasswordResetRequired=True
        )
        logger.info(f"Created login profile for {username}.")


def create_or_update_iam_policy(policy_name: str, policy_document: Dict) -> str:
    """Creates or updates an IAM policy and returns its ARN.

    Args:
        policy_name (str): Name of the IAM policy.
        policy_document (Dict): JSON document defining the IAM policy.

    Returns:
        str: ARN of the created or updated IAM policy.

    Raises:
        Exception: If policy creation or update fails.
    """
    policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
    try:
        iam_client.get_policy(PolicyArn=policy_arn)
        logger.info(f"Policy {policy_name} already exists.")
    except iam_client.exceptions.NoSuchEntityException:
        response = iam_client.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document),
            Description=f"Policy {policy_name} created by script.",
        )
        policy_arn = response["Policy"]["Arn"]
        logger.info(f"Created policy {policy_name}.")
    except Exception as e:
        logger.info(f"Error managing policy {policy_name}: {e}")
        raise
    return policy_arn


def attach_policy_to_group(group_name: str, policy_arn: str):
    """Attaches an IAM policy to a specified IAM group.

    Args:
        group_name (str): Name of the IAM group.
        policy_arn (str): ARN of the IAM policy to attach.

    Raises:
        Exception: If attaching the policy to the group fails.
    """
    try:
        response = iam_client.list_attached_group_policies(GroupName=group_name)
        attached_policies = [p["PolicyArn"] for p in response["AttachedPolicies"]]
        if policy_arn not in attached_policies:
            iam_client.attach_group_policy(GroupName=group_name, PolicyArn=policy_arn)
            logger.info(f"Attached policy {policy_arn} to group {group_name}.")
        else:
            logger.info(f"Policy {policy_arn} already attached to group {group_name}.")
    except Exception as e:
        logger.info(f"Error attaching policy: {e}")
        raise


def create_sns_topic_and_subscribe_emails(topic_name: str, emails: List[str]) -> str:
    """Creates an SNS topic and subscribes the provided emails.

    Args:
        topic_name (str): Name of the SNS topic to create.
        emails (List[str]): List of email addresses to subscribe to the topic.

    Returns:
        str: ARN of the created SNS topic.

    Raises:
        Exception: If topic creation or subscription fails.
    """
    topic_arn = None
    response = sns_client.list_topics()
    for t in response["Topics"]:
        if t["TopicArn"].endswith(":" + topic_name):
            topic_arn = t["TopicArn"]
            logger.info(f"SNS topic {topic_name} already exists.")
            break

    if not topic_arn:
        response = sns_client.create_topic(Name=topic_name)
        topic_arn = response["TopicArn"]
        logger.info(f"Created SNS topic {topic_name}.")

    for email in emails:
        subscribed = False
        for sub in sns_client.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]:
            if sub["Endpoint"] == email and sub["Protocol"] == "email":
                logger.info(f"Endpoint {email} already subscribed to {topic_arn}.")
                subscribed = True
                break
        if not subscribed:
            sns_client.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)
            logger.info(f"Subscribed {email} to {topic_arn}.")

    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "cloudwatch.amazonaws.com"},
                "Action": "SNS:Publish",
                "Resource": topic_arn,
            }
        ],
    }
    sns_client.set_topic_attributes(
        TopicArn=topic_arn,
        AttributeName="Policy",
        AttributeValue=json.dumps(policy_document),
    )
    logger.info(f"Set policy for SNS topic {topic_arn}.")
    return topic_arn


# --------------------------------------------------------------------------
# AWS Organizations Setup
# --------------------------------------------------------------------------

try:
    org_client.describe_organization()
    logger.info("Organization already exists.")
except org_client.exceptions.AWSOrganizationsNotInUseException:
    org_client.create_organization(FeatureSet="ALL")
    logger.info("Created new organization.")


# --------------------------------------------------------------------------
# IAM Users and Groups Setup
# --------------------------------------------------------------------------

USERNAMES = [email.split("@")[0] for email in EMAILS]

for username, email in zip(USERNAMES, EMAILS):
    create_iam_user(username, email)

try:
    iam_client.get_group(GroupName=GROUP_NAME)
    logger.info(f"Group {GROUP_NAME} already exists.")
except iam_client.exceptions.NoSuchEntityException:
    iam_client.create_group(GroupName=GROUP_NAME)
    logger.info(f"Created group {GROUP_NAME}.")

response = iam_client.get_group(GroupName=GROUP_NAME)
existing_users = [user["UserName"] for user in response["Users"]]

for username in USERNAMES:
    if username not in existing_users:
        iam_client.add_user_to_group(GroupName=GROUP_NAME, UserName=username)
        logger.info(f"Added {username} to group {GROUP_NAME}.")
    else:
        logger.info(f"{username} is already a member of group {GROUP_NAME}.")

admin_policy_name = "AdministratorAccessPolicy"
admin_policy_document = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}],
}
admin_policy_arn = create_or_update_iam_policy(admin_policy_name, admin_policy_document)
attach_policy_to_group(GROUP_NAME, admin_policy_arn)


# --------------------------------------------------------------------------
# AWS Budgets Setup
# --------------------------------------------------------------------------

budget_name = "SharedFreeTierBudget"

try:
    budgets_client.describe_budget(AccountId=account_id, BudgetName=budget_name)
    logger.info(f"Budget {budget_name} already exists.")
except budgets_client.exceptions.NotFoundException:
    subscribers = [{"SubscriptionType": "EMAIL", "Address": email} for email in EMAILS]
    notifications_with_subscribers = [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": threshold,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": subscribers,
        }
        for threshold in [20, 40, 60, 80, 90]
    ]
    notifications_with_subscribers.append(
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": subscribers,
        }
    )
    budgets_client.create_budget(
        AccountId=account_id,
        Budget={
            "BudgetName": budget_name,
            "BudgetLimit": {"Amount": "1", "Unit": "USD"},
            "BudgetType": "COST",
            "TimeUnit": "MONTHLY",
            "CostFilters": {},
        },
        NotificationsWithSubscribers=notifications_with_subscribers,
    )
    logger.info(f"Created budget {budget_name}.")


# --------------------------------------------------------------------------
# S3 Bucket Setup
# --------------------------------------------------------------------------

try:
    s3_client.head_bucket(Bucket=BUCKET_NAME)
    logger.info(f"Bucket {BUCKET_NAME} already exists.")
except ClientError as e:
    if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
        s3_client.create_bucket(
            Bucket=BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
        )
        logger.info(f"Created bucket {BUCKET_NAME}.")
    else:
        logger.info(f"Error checking bucket {BUCKET_NAME}: {e}")
        raise

if s3_client.get_bucket_versioning(Bucket=BUCKET_NAME).get("Status") != "Enabled":
    s3_client.put_bucket_versioning(
        Bucket=BUCKET_NAME, VersioningConfiguration={"Status": "Enabled"}
    )
    logger.info(f"Enabled versioning on bucket {BUCKET_NAME}.")

try:
    s3_client.get_bucket_encryption(Bucket=BUCKET_NAME)
    logger.info(f"Encryption already enabled on bucket {BUCKET_NAME}.")
except ClientError as e:
    if e.response["Error"]["Code"] == "ServerSideEncryptionConfigurationNotFoundError":
        s3_client.put_bucket_encryption(
            Bucket=BUCKET_NAME,
            ServerSideEncryptionConfiguration={
                "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
            },
        )
        logger.info(f"Enabled encryption on bucket {BUCKET_NAME}.")

s3_client.put_public_access_block(
    Bucket=BUCKET_NAME,
    PublicAccessBlockConfiguration={
        "BlockPublicAcls": True,
        "IgnorePublicAcls": True,
        "BlockPublicPolicy": True,
        "RestrictPublicBuckets": True,
    },
)
logger.info(f"Set public access block on bucket {BUCKET_NAME}.")

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": "s3:*",
            "Effect": "Deny",
            "Principal": "*",
            "Resource": [
                f"arn:aws:s3:::{BUCKET_NAME}/*",
                f"arn:aws:s3:::{BUCKET_NAME}",
            ],
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }
    ],
}
s3_client.put_bucket_policy(Bucket=BUCKET_NAME, Policy=json.dumps(policy))
logger.info(f"Set bucket policy on bucket {BUCKET_NAME}.")

for key_prefix in BUCKET_KEYS:
    key_prefix = key_prefix.strip() + "/"
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key_prefix)
    logger.info(f"Created bucket key: {key_prefix}")


# --------------------------------------------------------------------------
# DynamoDB Table Setup
# --------------------------------------------------------------------------

try:
    table = dynamodb_resource.Table(TABLE_NAME)
    table.load()
    logger.info(f"DynamoDB table {TABLE_NAME} already exists.")
except dynamodb_resource.meta.client.exceptions.ResourceNotFoundException:
    table = dynamodb_resource.create_table(
        TableName=TABLE_NAME,
        KeySchema=[{"AttributeName": "PrimaryKey", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "PrimaryKey", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
        Tags=[{"Key": "Environment", "Value": "Management"}, {"Key": "ManagedBy", "Value": "PythonScript"}],
    )
    table.wait_until_exists()
    logger.info(f"Created DynamoDB table {TABLE_NAME}.")
