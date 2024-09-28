#!/usr/bin/env python3

import argparse
import logging
import warnings
from typing import Any, Dict, List

import boto3
import hcl2
import urllib3
from botocore.exceptions import ClientError
from python_terraform import IsFlagged, Terraform
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from imported libraries
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("hcl2").setLevel(logging.CRITICAL)
logging.getLogger("python_terraform").setLevel(logging.CRITICAL)
logging.getLogger("tenacity").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

ResourceConfig = Dict[str, Any]
ResourceEntry = Dict[str, str]


RESOURCE_MAPPINGS = {
    # Add all the necessary resource mappings
    "aws_organizations_organization": {
        "client": "organizations",
        "check_method": "describe_organization",
        "create_method": "create_organization",
        "id_param": None,  # No ID param needed; special handling
    },
    "aws_iam_user": {
        "client": "iam",
        "check_method": "get_user",
        "create_method": "create_user",
        "id_param": "UserName",
    },
    "aws_iam_user_login_profile": {
        "client": "iam",
        "check_method": "get_login_profile",
        "create_method": "create_login_profile",
        "id_param": "UserName",
    },
    "aws_iam_group": {
        "client": "iam",
        "check_method": "get_group",
        "create_method": "create_group",
        "id_param": "GroupName",
    },
    "aws_iam_group_membership": {
        "client": "iam",
        "check_method": "get_group",
        "create_method": "add_user_to_group",
        "id_param": "GroupName",
    },
    "aws_iam_policy": {
        "client": "iam",
        "check_method": "get_policy",
        "create_method": "create_policy",
        "id_param": "PolicyArn",
    },
    "aws_iam_group_policy_attachment": {
        "client": "iam",
        "check_method": "list_attached_group_policies",
        "create_method": "attach_group_policy",
        "id_param": "GroupName",
    },
    "aws_s3_bucket": {
        "client": "s3",
        "check_method": "head_bucket",
        "create_method": "create_bucket",
        "id_param": "Bucket",
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
    "aws_s3_bucket_policy": {
        "client": "s3",
        "check_method": "get_bucket_policy",
        "create_method": "put_bucket_policy",
        "id_param": "Bucket",
    },
    "aws_dynamodb_table": {
        "client": "dynamodb",
        "check_method": "describe_table",
        "create_method": "create_table",
        "id_param": "TableName",
    },
    "aws_dynamodb_resource_policy": {
        "client": "dynamodb",
        "check_method": "get_resource_policy",
        "create_method": "put_resource_policy",
        "id_param": "ResourceArn",
    },
    "aws_budgets_budget": {
        "client": "budgets",
        "check_method": "describe_budget",
        "create_method": "create_budget",
        "id_param": None,  # Special handling
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
    "aws_iam_role": {
        "client": "iam",
        "check_method": "get_role",
        "create_method": "create_role",
        "id_param": "RoleName",
    },
    "aws_iam_role_policy_attachment": {
        "client": "iam",
        "check_method": "list_attached_role_policies",
        "create_method": "attach_role_policy",
        "id_param": "RoleName",
    },
}


def retry_if_not_access_denied_exception(exception):
    if isinstance(exception, ClientError):
        error_code = exception.response["Error"]["Code"]
        return error_code not in ["AccessDenied", "ValidationException", "InvalidParameter"]
    return True


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
        logger.info(f"Checking if {address} is in Terraform state")
        code, out, err = self.tf.cmd(
            'state', 'list', capture_output=True, no_color=IsFlagged
        )

        if code == 0:
            resources = out.strip().split('\n')
            logger.debug(f"Resources in state: {resources}")
            if address in resources:
                logger.info(f"Resource {address} is already in Terraform state.")
                return True
            else:
                logger.info(f"Resource {address} is not in Terraform state.")
                return False
        else:
            err_str = err.decode('utf-8') if isinstance(err, bytes) else err
            if 'No state file was found' in err_str:
                logger.warning("No state file found. Assuming state is empty.")
                return False
            else:
                logger.error(f"Error running 'terraform state list': {err_str}")
            return False

    def import_resource(self, address: str, resource_id: str) -> bool:
        logger.info(f"Attempting to import {address} with ID {resource_id}")
        code, _, stderr = self.tf.import_cmd(
            address,
            resource_id,
            capture_output=True,
            no_color=IsFlagged,
            lock=False,
        )
        stderr_str = stderr.decode('utf-8') if isinstance(stderr, bytes) else stderr
        if code == 0:
            logger.info(f"Successfully imported: {address}")
            return True
        else:
            if 'Resource already managed by Terraform' in stderr_str:
                logger.info(f"EXISTS: {address}")
                return True
            else:
                logger.warning(f"Failed to import {address}: {stderr_str}")
                return False

    def apply(self):
        logger.info("Applying Terraform configuration...")
        code, _, stderr = self.tf.apply(
            capture_output=True, skip_plan=True, no_color=IsFlagged
        )
        if code == 0:
            logger.info("Terraform apply completed successfully")
        else:
            logger.error(f"Terraform apply failed: {stderr}")

    def state_rm(self, address: str):
        logger.info(f"Removing {address} from Terraform state")
        code, _, stderr = self.tf.cmd(
            "state", "rm", address, capture_output=True, no_color=IsFlagged
        )
        if code != 0:
            logger.error(f"Failed to remove {address} from state: {stderr}")


class ResourceImporter:
    def __init__(self, tf_manager: TerraformManager, tf_config: Dict[str, Any]):
        self.tf_manager = tf_manager
        self.tf_config = tf_config  # Store the parsed Terraform configuration

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(retry_if_not_access_denied_exception),
    )
    def import_or_create(
        self, address: str, resource_id: Any, resource_type: str, resource_config: Dict[str, Any]
    ) -> None:
        if self.tf_manager.is_in_state(address):
            # Resource is already in Terraform state; no need to import
            return

        mapping = RESOURCE_MAPPINGS.get(resource_type)
        if not mapping:
            logger.debug(f"No mapping for resource type: {resource_type}. Skipping.")
            return

        client_name = mapping["client"]
        client = boto3.client(client_name)
        check_exists = getattr(client, mapping["check_method"])

        try:
            # Handle special cases
            if resource_type == "aws_budgets_budget":
                # resource_id is a dict with AccountId and BudgetName
                check_exists(**resource_id)
                logger.info(f"EXISTS: {address}")
                import_id = f"{resource_id['AccountId']}:{resource_id['BudgetName']}"
                self.tf_manager.import_resource(address, import_id)
                return

            elif resource_type == "aws_organizations_organization":
                org = check_exists()
                org_id = org["Organization"]["Id"]
                logger.info(f"EXISTS: {address}")
                self.tf_manager.import_resource(address, org_id)
                return

            elif resource_type == "aws_sns_topic_subscription":
                if resource_id:
                    check_exists(SubscriptionArn=resource_id)
                    logger.info(f"EXISTS: {address}")
                    self.tf_manager.import_resource(address, resource_id)
                else:
                    logger.info(f"EXISTS: {address}")
                return

            elif resource_type == "aws_dynamodb_resource_policy":
                # For DynamoDB resource policies, ensure ResourceArn is correct
                check_exists(ResourceArn=resource_id)
                logger.info(f"EXISTS: {address}")
                self.tf_manager.import_resource(address, resource_id)
                return

            # Default handling
            id_param = mapping["id_param"]
            if id_param:
                check_exists(**{id_param: resource_id})
                logger.info(f"EXISTS: {address}")
                self.tf_manager.import_resource(address, resource_id)
            else:
                check_exists()
                logger.info(f"EXISTS: {address}")
                self.tf_manager.import_resource(address, resource_id)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["NoSuchEntity", "ResourceNotFoundException", "ValidationException", "NotFoundException"]:
                logger.info(f"Resource not found: {address}. It may need to be created.")
            elif error_code == "AccessDenied":
                logger.info(f"EXISTS: {address}")
            elif error_code == "InvalidParameter":
                logger.info(f"AWS API error for {address}: {str(e)}")
            else:
                logger.info(f"AWS API error for {address}: {str(e)}")

    def extract_resource_id(
        self, resource_type: str, resource_name: str, resource_config: Dict[str, Any]
    ) -> Any:
        account_id = boto3.client('sts').get_caller_identity()['Account']
        region = boto3.session.Session().region_name

        if resource_type == "aws_iam_policy":
            policy_name = resource_config.get("name", resource_name)
            return f"arn:aws:iam::{account_id}:policy/{policy_name}"
        elif resource_type == "aws_iam_group_policy_attachment":
            # For importing, the ID is in the format 'group-name/policy-arn'
            group_name = resource_config.get("group")
            policy_arn = resource_config.get("policy_arn")
            return f"{group_name}/{policy_arn}"
        elif resource_type == "aws_s3_bucket":
            return resource_config.get("bucket", resource_name)
        elif resource_type == "aws_cloudwatch_metric_alarm":
            return [resource_config.get("alarm_name", resource_name)]
        elif resource_type == "aws_sns_topic":
            topic_name = resource_config.get("name", resource_name)
            return f"arn:aws:sns:{region}:{account_id}:{topic_name}"
        elif resource_type == "aws_sns_topic_subscription":
            return self.get_subscription_arn(region, account_id, resource_config)
        elif resource_type == "aws_dynamodb_resource_policy":
            table_name = resource_config.get("name", resource_name)
            return f"arn:aws:dynamodb:{region}:{account_id}:table/{table_name}"
        elif resource_type == "aws_budgets_budget":
            budget_name = resource_config.get("name", resource_name)
            return {"AccountId": account_id, "BudgetName": budget_name}
        elif resource_type == "aws_organizations_organization":
            return None  # Will be handled specially
        elif resource_type == "aws_dynamodb_table":
            return resource_config.get("name", resource_name)
        elif resource_type == "aws_iam_user":
            return resource_config.get("name", resource_name)
        elif resource_type == "aws_iam_role":
            return resource_config.get("name", resource_name)
        elif resource_type == "aws_iam_user_login_profile":
            user_name = resource_config.get("user", resource_name)
            if user_name.startswith('${'):
                user_name = self.resolve_reference(user_name)
            return user_name
        elif resource_type == "aws_s3_bucket_policy":
            bucket = resource_config.get("bucket", resource_name)
            if bucket.startswith('${'):
                bucket = self.resolve_reference(bucket)
            return bucket
        # Add more cases as needed
        return resource_config.get("id", resource_name)

    def resolve_reference(self, reference: str) -> str:
        # Implement logic to resolve references like "${aws_iam_user.jagustin.name}"
        if reference.startswith('${') and reference.endswith('}'):
            ref = reference[2:-1]
            parts = ref.split('.')
            if len(parts) >= 3:
                resource_type = parts[0]
                resource_name = parts[1]
                attribute = parts[2]
                resource = self.resource_config_lookup(resource_type, resource_name)
                if resource:
                    return resource.get(attribute)
        logger.info(f"Unable to resolve reference: {reference}")
        return reference

    def resource_config_lookup(self, resource_type: str, resource_name: str) -> Dict[str, Any]:
        for resource_block in self.tf_config.get('resource', []):
            if resource_type in resource_block:
                if resource_name in resource_block[resource_type]:
                    return resource_block[resource_type][resource_name]
        return {}

    def get_subscription_arn(self, region, account_id, resource_config):
        sns_client = boto3.client('sns', region_name=region)
        topic_arn = resource_config.get('topic_arn')
        protocol = resource_config.get('protocol')
        endpoint = resource_config.get('endpoint')

        paginator = sns_client.get_paginator('list_subscriptions')
        for page in paginator.paginate():
            for subscription in page['Subscriptions']:
                if (
                    subscription['TopicArn'] == topic_arn
                    and subscription['Protocol'] == protocol
                    and subscription['Endpoint'] == endpoint
                ):
                    return subscription['SubscriptionArn']
        return None  # Subscription not found

    def handle_iam_user_deletion(self, user_name):
        iam_client = boto3.client('iam')
        try:
            iam_client.delete_login_profile(UserName=user_name)
            logger.info(f"Deleted login profile for user: {user_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                logger.info(f"No login profile found for user: {user_name}")
            else:
                logger.error(f"Error deleting login profile for user {user_name}: {e}")

    def handle_iam_group_deletion(self, group_name):
        iam_client = boto3.client('iam')
        try:
            # Detach policies
            attached_policies = iam_client.list_attached_group_policies(GroupName=group_name)['AttachedPolicies']
            for policy in attached_policies:
                iam_client.detach_group_policy(GroupName=group_name, PolicyArn=policy['PolicyArn'])
                logger.info(f"Detached policy {policy['PolicyArn']} from group {group_name}")

            iam_client.delete_group(GroupName=group_name)
            logger.info(f"Deleted IAM group: {group_name}")
        except ClientError as e:
            logger.error(f"Error deleting IAM group {group_name}: {e}")

    def handle_iam_group_replacement(self, group_name):
        iam_client = boto3.client('iam')
        try:
            # Rename the existing group
            iam_client.update_group(GroupName=group_name, NewGroupName=f"{group_name}_old")
            logger.info(f"Renamed IAM group {group_name} to {group_name}_old")
            # Detach policies from the old group
            self.handle_iam_group_deletion(f"{group_name}_old")
        except ClientError as e:
            logger.error(f"Error replacing IAM group {group_name}: {e}")

    def process_resources(self, resources: List[Dict[str, Any]]) -> None:
        for resource_block in resources:
            for resource_type, resource_data in resource_block.items():
                for resource_name, resource_config in resource_data.items():
                    address = f"{resource_type}.{resource_name}"
                    resource_id = self.extract_resource_id(
                        resource_type, resource_name, resource_config
                    )
                    if resource_id is None:
                        logger.info(f"EXISTS: {address}")
                        continue
                    try:
                        if resource_type == "aws_iam_user" and self.tf_manager.is_in_state(address):
                            self.handle_iam_user_deletion(resource_id)  # Handle any IAM user deletion
                        elif resource_type == "aws_iam_group" and self.tf_manager.is_in_state(address):
                            if any(  # Check if a group with the same name is being created
                                k.startswith(resource_type) and v.get("name") == resource_config.get("name")
                                for k, v in resource_block.items()
                                if k != resource_type or list(v.keys())[0] != resource_name
                            ):
                                self.handle_iam_group_replacement(resource_id)  # Handle IAM group replacement
                            else:
                                self.handle_iam_group_deletion(resource_id)  # Handle IAM group deletion
                        else:
                            self.import_or_create(address, resource_id, resource_type, resource_config)

                    except Exception:
                        logger.info(f"EXISTS: {address}")
                        continue


def parse_terraform_file(file_path: str) -> Dict[str, Any]:
    logger.info(f"Parsing Terraform file: {file_path}")
    with open(file_path, "r") as f:
        content = f.read()
        logger.debug(f"File content:\n{content}")
        return hcl2.loads(content)


def extract_resources(tf_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    resources = tf_config.get("resource", [])
    return resources


def main(tf_file: str):
    try:
        logger.info(f"Starting import process for file: {tf_file}")
        tf_config = parse_terraform_file(tf_file)
        resources = extract_resources(tf_config)
        tf_manager = TerraformManager()
        tf_manager.initialize()
        importer = ResourceImporter(tf_manager, tf_config)
        importer.process_resources(resources)
        tf_manager.apply()

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
