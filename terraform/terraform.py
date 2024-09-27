#!/usr/bin/env python3

import argparse
import logging
from typing import Any, Dict, List
import json

import boto3
import hcl2
from botocore.exceptions import ClientError
from python_terraform import IsFlagged, Terraform
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ResourceConfig = Dict[str, Any]
ResourceEntry = Dict[str, str]


RESOURCE_MAPPINGS = {
    "aws_organizations_organization": {
        "client": "organizations",
        "check_method": "describe_organization",
        "create_method": "create_organization",
        "id_param": "Id",
    },
    "aws_iam_user": {
        "client": "iam",
        "check_method": "get_user",
        "create_method": "create_user",
        "id_param": "UserName",
    },
    "aws_iam_group": {
        "client": "iam",
        "check_method": "get_group",
        "create_method": "create_group",
        "id_param": "GroupName",
    },
    "aws_iam_policy": {
        "client": "iam",
        "check_method": "get_policy",
        "create_method": "create_policy",
        "id_param": "PolicyArn",
    },
    "aws_s3_bucket": {
        "client": "s3",
        "check_method": "head_bucket",
        "create_method": "create_bucket",
        "id_param": "Bucket",
    },
    "aws_dynamodb_table": {
        "client": "dynamodb",
        "check_method": "describe_table",
        "create_method": "create_table",
        "id_param": "TableName",
    },
    "aws_budgets_budget": {
        "client": "budgets",
        "check_method": "describe_budget",
        "create_method": "create_budget",
        "id_param": "BudgetName",
    },
    "aws_sns_topic": {
        "client": "sns",
        "check_method": "get_topic_attributes",
        "create_method": "create_topic",
        "id_param": "TopicArn",
    },
    "aws_sns_topic_subscription": {
        "client": "sns",
        "check_method": "get_subscription_attributes",
        "create_method": "subscribe",
        "id_param": "SubscriptionArn",
    },
    "aws_cloudwatch_metric_alarm": {
        "client": "cloudwatch",
        "check_method": "describe_alarms",
        "create_method": "put_metric_alarm",
        "id_param": "AlarmNames",
    },
    "aws_iam_role": {
        "client": "iam",
        "check_method": "get_role",
        "create_method": "create_role",
        "id_param": "RoleName",
    },
    "aws_iam_user_login_profile": {
        "client": "iam",
        "check_method": "get_login_profile",
        "create_method": "create_login_profile",
        "id_param": "UserName",
    },
    "aws_iam_group_policy_attachment": {
        "client": "iam",
        "check_method": "get_group_policy",
        "create_method": "attach_group_policy",
        "id_param": "PolicyArn",
    },
    "aws_s3_bucket_versioning": {
        "client": "s3",
        "check_method": "get_bucket_versioning",
        "create_method": "put_bucket_versioning",
        "id_param": "Bucket",
    },
    "aws_s3_bucket_server_side_encryption_configuration": {
        "client": "s3",
        "check_method": "get_bucket_encryption",
        "create_method": "put_bucket_encryption",
        "id_param": "Bucket",
    },
    "aws_s3_bucket_public_access_block": {
        "client": "s3",
        "check_method": "get_public_access_block",
        "create_method": "put_public_access_block",
        "id_param": "Bucket",
    },
    "aws_dynamodb_resource_policy": {
        "client": "dynamodb",
        "check_method": "get_resource_policy",
        "create_method": "put_resource_policy",
        "id_param": "ResourceArn",
    },
    "aws_sns_topic_policy": {
        "client": "sns",
        "check_method": "get_topic_attributes",
        "create_method": "set_topic_attributes",
        "id_param": "TopicArn",
    },
    "aws_cloudwatch_metric_alarm": {
        "client": "cloudwatch",
        "check_method": "describe_alarms",
        "create_method": "put_metric_alarm",
        "id_param": "AlarmNames",
    },
    "aws_iam_role_policy_attachment": {
        "client": "iam",
        "check_method": "list_attached_role_policies",
        "create_method": "attach_role_policy",
        "id_param": "RoleName",
    },
    "aws_s3_bucket_policy": {
        "client": "s3",
        "check_method": "get_bucket_policy",
        "create_method": "put_bucket_policy",
        "id_param": "Bucket",
    },
}


class TerraformManager:
    def __init__(self):
        self.tf = Terraform(working_dir=".")

    def initialize(self):
        logger.info("Initializing Terraform...")
        code, _, stderr = self.tf.init(
            capture_output=True, no_color=IsFlagged, input=False
        )
        if code != 0:
            raise Exception(f"Terraform initialization failed: {stderr}")
        logger.info("Terraform initialized successfully")

    def is_in_state(self, address: str) -> bool:
        logger.debug(f"Checking if {address} is in Terraform state")
        code, out, _ = self.tf.cmd(
            "state", "list", address, capture_output=True, no_color=IsFlagged
        )
        return code == 0 and bool(out.strip())

    def import_resource(self, address: str, resource_id: str) -> bool:
        logger.info(f"Attempting to import {address} with ID {resource_id}")
        code, _, stderr = self.tf.import_cmd(
            address,
            resource_id,
            capture_output=True,
            no_color=IsFlagged,
            lock=False,
        )
        if code == 0:
            logger.info(f"Successfully imported: {address}")
            return True
        else:
            logger.error(f"Failed to import {address}: {stderr}")
            return False

    def apply(self):
        logger.info("Applying Terraform configuration...")
        code, stdout, stderr = self.tf.apply(
            capture_output=True, skip_plan=True, no_color=IsFlagged
        )
        if code == 0:
            logger.info("Terraform apply completed successfully")
        else:
            logger.error(f"Terraform apply failed: {stderr}")


class ResourceImporter:
    def __init__(self, tf_manager: TerraformManager):
        self.tf_manager = tf_manager

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(ClientError),
    )
    def import_or_create(self, address: str, resource_id: str, resource_type: str) -> None:
        logger.debug(f"Processing resource: {address}")
        if self.tf_manager.is_in_state(address):
            logger.info(f"Resource already in state: {address}")
            return

        mapping = RESOURCE_MAPPINGS.get(resource_type)
        if not mapping:
            logger.warning(f"No mapping for resource type: {resource_type}. Skipping.")
            return

        client = boto3.client(mapping["client"])
        check_exists = getattr(client, mapping["check_method"])

        try:
            # Special case for aws_organizations_organization
            if resource_type == "aws_organizations_organization":
                org = check_exists()
                resource_id = org["Organization"]["Id"]
                logger.info(f"Organization exists: {resource_id}")

            # Special case for aws_sns_topic_policy
            elif resource_type == "aws_sns_topic_policy":
                check_exists(TopicArn=resource_id)
                logger.info(f"SNS Topic Policy exists: {resource_id}")

            # Special case for aws_s3_bucket_policy
            elif resource_type == "aws_s3_bucket_policy":
                check_exists(Bucket=resource_id)
                logger.info(f"S3 Bucket Policy exists: {resource_id}")

            # Special case for aws_iam_group_policy_attachment
            elif resource_type == "aws_iam_group_policy_attachment":
                policy_arn = resource_id
                group_name = address.split(".")[1].split("_")[0]  # Assuming naming convention
                check_exists(GroupName=group_name, PolicyArn=policy_arn)
                logger.info(
                    f"IAM Group Policy Attachment exists: GroupName={group_name}, PolicyArn={policy_arn}"
                )
                resource_id = group_name  # Use group name for import

            # Special case for aws_iam_role_policy_attachment
            elif resource_type == "aws_iam_role_policy_attachment":
                role_name = resource_id
                policy_arn = address.split(".")[1].split("-")[
                    -1
                ]  # Assuming naming convention
                attached_policies = check_exists(RoleName=role_name)
                for policy in attached_policies["AttachedPolicies"]:
                    if policy["PolicyArn"] == policy_arn:
                        logger.info(
                            f"IAM Role Policy Attachment exists: RoleName={role_name}, PolicyArn={policy_arn}"
                        )
                        break
                else:
                    raise ClientError(
                        {
                            "Error": {
                                "Code": "ResourceNotFoundException",
                                "Message": "Policy not attached to role",
                            }
                        },
                        "list_attached_role_policies",
                    )

            # Special case for aws_cloudwatch_metric_alarm
            elif resource_type == "aws_cloudwatch_metric_alarm":
                alarm_name = resource_id[0]  # Assuming resource_id is a list
                check_exists(AlarmNames=[alarm_name])
                logger.info(f"CloudWatch Metric Alarm exists: {alarm_name}")
                resource_id = alarm_name  # Use alarm name for import

            else:
                # Default handling for most resources
                id_param = mapping["id_param"]
                check_exists(**{id_param: resource_id})
                logger.info(f"Resource exists: {address}")

            self.tf_manager.import_resource(address, resource_id)

        except ClientError as e:
            if e.response["Error"]["Code"] in [
                "NoSuchEntity",
                "ResourceNotFoundException",
            ]:
                logger.info(f"Resource not found: {address}")
            else:
                logger.error(f"AWS API error for {address}: {str(e)}")
                raise

    def process_resources(self, resources: List[Dict[str, Any]]) -> None:
        logger.debug(f"Processing resources: {json.dumps(resources, indent=2)}")
        for resource_block in resources:
            for resource_type, resource_data in resource_block.items():
                logger.debug(f"Processing resource type: {resource_type}")

                if resource_type == "aws_iam_user":
                    for resource_name, _ in resource_data.items():
                        address = f"{resource_type}.{resource_name}"
                        resource_id = resource_name  # Username is the resource ID
                        self.import_or_create(address, resource_id, resource_type)
                elif resource_type == "aws_sns_topic_subscription":
                    for resource_name, resource_config in resource_data.items():
                        address = f"{resource_type}.{resource_name}"
                        endpoint = resource_config.get("endpoint")
                        topic_arn = resource_config.get("topic_arn")
                        resource_id = f"{topic_arn}:{endpoint}"
                        self.import_or_create(address, resource_id, resource_type)

                else:
                    for resource_name, _ in resource_data.items():
                        address = f"{resource_type}.{resource_name}"
                        resource_id = resource_name
                        self.import_or_create(address, resource_id, resource_type)


def parse_terraform_file(file_path: str) -> Dict[str, Any]:
    logger.info(f"Parsing Terraform file: {file_path}")
    with open(file_path, "r") as f:
        content = f.read()
        logger.debug(f"File content:\n{content}")
        return hcl2.loads(content)


def extract_resources(tf_config: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Extracting resources from config: {json.dumps(tf_config, indent=2)}")
    resources = tf_config.get("resource", {})
    logger.debug(f"Extracted resources: {json.dumps(resources, indent=2)}")
    return resources


def main(tf_file: str):
    try:
        logger.info(f"Starting import process for file: {tf_file}")
        tf_config = parse_terraform_file(tf_file)
        resources = extract_resources(tf_config)
        print(resources)

        tf_manager = TerraformManager()
        tf_manager.initialize()

        importer = ResourceImporter(tf_manager)
        importer.process_resources(resources)

        # tf_manager.apply()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Detailed traceback:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import AWS resources into Terraform state and apply configuration"
    )
    parser.add_argument(
        "--tf-file",
        default="main.tf",
        help="Path to the Terraform configuration file",
    )
    args = parser.parse_args()
    main(args.tf_file)