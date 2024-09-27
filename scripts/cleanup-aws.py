#!/usr/bin/env python3
"""
AWS Resource Cleanup Script (Preserving Terraform-managed Resources)

This script automates the process of deleting various AWS resources across multiple regions,
while preserving resources managed by Terraform. It uses boto3 to interact with AWS services
and deletes resources such as EC2 instances, S3 buckets, IAM users, CloudFormation stacks,
Lambda functions, and more.

The script reads the Terraform state file to identify resources that should be preserved.

Usage:
    python cleanup-aws.py

Requirements:
    - Python 3.6+
    - boto3 library
    - Configured AWS credentials with necessary permissions
    - Terraform state file

Note:
    This script will delete resources permanently. Use with caution and ensure your Terraform
    state file is up to date before running this script.
"""
import boto3
from botocore.exceptions import ClientError
import logging
import sys
import concurrent.futures
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global set to store resources managed by Terraform
terraform_resources = set()

def read_terraform_state(state_file_path):
    """
    Read the Terraform state file and extract resource identifiers.

    :param state_file_path: Path to the Terraform state file
    :type state_file_path: str
    """
    global terraform_resources
    try:
        with open(state_file_path, 'r') as f:
            state = json.load(f)

        for resource in state.get('resources', []):
            for instance in resource.get('instances', []):
                attributes = instance.get('attributes', {})
                if 'id' in attributes:
                    terraform_resources.add(attributes['id'])
                if 'arn' in attributes:
                    terraform_resources.add(attributes['arn'])
    except Exception as e:
        logger.error(f"Error reading Terraform state file: {e}")
        sys.exit(1)

def should_preserve(resource_id):
    """
    Check if a resource should be preserved (managed by Terraform).

    :param resource_id: ID or ARN of the resource
    :type resource_id: str
    :return: True if the resource should be preserved, False otherwise
    :rtype: bool
    """
    return resource_id in terraform_resources

def delete_ec2_resources(region):
    """
    Delete EC2 resources in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting EC2 resources in region: {region}")
    ec2 = boto3.resource('ec2', region_name=region)
    client = boto3.client('ec2', region_name=region)

    # Terminate EC2 Instances
    instances = ec2.instances.filter(Filters=[{'Name': 'instance-state-name', 'Values': ['running', 'stopped']}])
    for instance in instances:
        if not should_preserve(instance.id):
            logger.info(f"Terminating EC2 instance: {instance.id}")
            instance.terminate()

    # Delete EBS Volumes
    volumes = ec2.volumes.filter(Filters=[{'Name': 'status', 'Values': ['available']}])
    for volume in volumes:
        if not should_preserve(volume.id):
            try:
                logger.info(f"Deleting EBS volume: {volume.id}")
                volume.delete()
            except ClientError as e:
                logger.error(f"Could not delete volume {volume.id}: {e}")

    # Release Elastic IPs
    addresses = client.describe_addresses()['Addresses']
    for address in addresses:
        if 'AssociationId' not in address:
            allocation_id = address.get('AllocationId')
            public_ip = address.get('PublicIp')
            if not should_preserve(allocation_id):
                logger.info(f"Releasing Elastic IP: {public_ip}")
                try:
                    client.release_address(AllocationId=allocation_id)
                except ClientError as e:
                    logger.error(f"Could not release Elastic IP {public_ip}: {e}")

    # Delete Security Groups
    security_groups = client.describe_security_groups()['SecurityGroups']
    for sg in security_groups:
        sg_id = sg['GroupId']
        sg_name = sg['GroupName']
        if sg_name != 'default' and not should_preserve(sg_id):
            try:
                logger.info(f"Deleting Security Group: {sg_name} ({sg_id})")
                client.delete_security_group(GroupId=sg_id)
            except ClientError as e:
                logger.error(f"Could not delete Security Group {sg_name} ({sg_id}): {e}")

    # Delete Key Pairs
    key_pairs = client.describe_key_pairs()['KeyPairs']
    for key in key_pairs:
        key_name = key['KeyName']
        if not should_preserve(key_name):
            try:
                logger.info(f"Deleting Key Pair: {key_name}")
                client.delete_key_pair(KeyName=key_name)
            except ClientError as e:
                logger.error(f"Could not delete Key Pair {key_name}: {e}")

    # Delete VPCs
    vpcs = ec2.vpcs.all()
    for vpc in vpcs:
        if not should_preserve(vpc.id):
            try:
                delete_vpc_resources(vpc)
            except ClientError as e:
                logger.error(f"Could not delete VPC {vpc.id}: {e}")

def delete_vpc_resources(vpc):
    """
    Delete resources associated with a VPC.

    :param vpc: VPC resource object
    :type vpc: boto3.resources.factory.ec2.Vpc
    """
    # Delete subnets
    for subnet in vpc.subnets.all():
        if not should_preserve(subnet.id):
            logger.info(f"Deleting Subnet: {subnet.id}")
            subnet.delete()

    # Delete route tables
    for rt in vpc.route_tables.all():
        if not rt.associations_attribute or not rt.associations_attribute[0].get('Main', False):
            if not should_preserve(rt.id):
                logger.info(f"Deleting Route Table: {rt.id}")
                rt.delete()

    # Detach and delete internet gateways
    for igw in vpc.internet_gateways.all():
        if not should_preserve(igw.id):
            logger.info(f"Detaching and Deleting Internet Gateway: {igw.id}")
            vpc.detach_internet_gateway(InternetGatewayId=igw.id)
            igw.delete()

    # Delete network ACLs
    for acl in vpc.network_acls.all():
        if not acl.is_default and not should_preserve(acl.id):
            logger.info(f"Deleting Network ACL: {acl.id}")
            acl.delete()

    # Delete VPC
    logger.info(f"Deleting VPC: {vpc.id}")
    vpc.delete()

def delete_s3_buckets(region):
    """
    Delete S3 buckets in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting S3 buckets in region: {region}")
    s3 = boto3.resource('s3', region_name=region)
    for bucket in s3.buckets.all():
        if not should_preserve(bucket.name):
            try:
                logger.info(f"Emptying and deleting bucket: {bucket.name}")
                bucket.object_versions.delete()
                bucket.objects.all().delete()
                bucket.delete()
            except ClientError as e:
                logger.error(f"Could not delete bucket {bucket.name}: {e}")

def delete_iam_users():
    """
    Delete IAM users and their associated resources, except those managed by Terraform.
    """
    logger.info("Deleting IAM users")
    iam = boto3.resource('iam')
    for user in iam.users.all():
        if not should_preserve(user.arn):
            logger.info(f"Deleting IAM user: {user.name}")
            try:
                delete_iam_user_resources(user)
            except ClientError as e:
                logger.error(f"Error deleting user {user.name}: {e}")

def delete_iam_user_resources(user):
    """
    Delete resources associated with an IAM user.

    :param user: IAM user resource object
    :type user: boto3.resources.factory.iam.User
    """
    # Detach policies
    for policy in user.attached_policies.all():
        user.detach_policy(PolicyArn=policy.arn)
    # Delete inline policies
    for policy in user.policies.all():
        policy.delete()
    # Delete access keys
    for key in user.access_keys.all():
        key.delete()
    # Remove from groups
    for group in user.groups.all():
        group.remove_user(UserName=user.name)
    # Delete login profile
    try:
        user.LoginProfile().delete()
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchEntity':
            logger.error(f"Error deleting login profile for {user.name}: {e}")
    # Delete MFA devices
    for mfa_device in user.mfa_devices.all():
        mfa_device.disassociate()
    # Finally, delete the user
    user.delete()

def delete_cloudformation_stacks(region):
    """
    Delete CloudFormation stacks in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting CloudFormation stacks in region: {region}")
    cf = boto3.client('cloudformation', region_name=region)
    try:
        stacks = cf.list_stacks(StackStatusFilter=[
            'CREATE_COMPLETE', 'ROLLBACK_COMPLETE', 'UPDATE_COMPLETE', 'UPDATE_ROLLBACK_COMPLETE'
        ])['StackSummaries']
    except ClientError as e:
        logger.error(f"Error listing CloudFormation stacks: {e}")
        return

    for stack in stacks:
        stack_name = stack['StackName']
        if not should_preserve(stack['StackId']):
            logger.info(f"Deleting CloudFormation stack: {stack_name}")
            try:
                cf.delete_stack(StackName=stack_name)
                waiter = cf.get_waiter('stack_delete_complete')
                waiter.wait(StackName=stack_name)
            except ClientError as e:
                logger.error(f"Error deleting stack {stack_name}: {e}")

def delete_lambda_functions(region):
    """
    Delete Lambda functions in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting Lambda functions in region: {region}")
    lambda_client = boto3.client('lambda', region_name=region)
    try:
        paginator = lambda_client.get_paginator('list_functions')
        for page in paginator.paginate():
            for function in page['Functions']:
                function_name = function['FunctionName']
                if not should_preserve(function['FunctionArn']):
                    logger.info(f"Deleting Lambda function: {function_name}")
                    try:
                        lambda_client.delete_function(FunctionName=function_name)
                    except ClientError as e:
                        logger.error(f"Error deleting function {function_name}: {e}")
    except ClientError as e:
        logger.error(f"Error listing Lambda functions: {e}")

def delete_dynamodb_tables(region):
    """
    Delete DynamoDB tables in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting DynamoDB tables in region: {region}")
    dynamodb = boto3.client('dynamodb', region_name=region)
    try:
        tables = dynamodb.list_tables()['TableNames']
        for table_name in tables:
            if not should_preserve(f"arn:aws:dynamodb:{region}:{boto3.client('sts').get_caller_identity().get('Account')}:table/{table_name}"):
                logger.info(f"Deleting DynamoDB table: {table_name}")
                try:
                    dynamodb.delete_table(TableName=table_name)
                    waiter = dynamodb.get_waiter('table_not_exists')
                    waiter.wait(TableName=table_name)
                except ClientError as e:
                    logger.error(f"Error deleting table {table_name}: {e}")
    except ClientError as e:
        logger.error(f"Error listing DynamoDB tables: {e}")

def delete_rds_instances(region):
    """
    Delete RDS instances in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting RDS instances in region: {region}")
    rds = boto3.client('rds', region_name=region)
    try:
        instances = rds.describe_db_instances()['DBInstances']
        for instance in instances:
            instance_id = instance['DBInstanceIdentifier']
            if not should_preserve(instance['DBInstanceArn']):
                logger.info(f"Deleting RDS instance: {instance_id}")
                try:
                    rds.delete_db_instance(
                        DBInstanceIdentifier=instance_id,
                        SkipFinalSnapshot=True,
                        DeleteAutomatedBackups=True
                    )
                    waiter = rds.get_waiter('db_instance_deleted')
                    waiter.wait(DBInstanceIdentifier=instance_id)
                except ClientError as e:
                    logger.error(f"Error deleting RDS instance {instance_id}: {e}")
    except ClientError as e:
        logger.error(f"Error describing RDS instances: {e}")

def delete_elasticache_clusters(region):
    """
    Delete ElastiCache clusters in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting ElastiCache clusters in region: {region}")
    elasticache = boto3.client('elasticache', region_name=region)
    try:
        clusters = elasticache.describe_cache_clusters()['CacheClusters']
        for cluster in clusters:
            cluster_id = cluster['CacheClusterId']
            if not should_preserve(cluster['ARN']):
                logger.info(f"Deleting ElastiCache cluster: {cluster_id}")
                try:
                    elasticache.delete_cache_cluster(CacheClusterId=cluster_id)
                except ClientError as e:
                    logger.error(f"Error deleting ElastiCache cluster {cluster_id}: {e}")
    except ClientError as e:
        logger.error(f"Error describing ElastiCache clusters: {e}")

def delete_efs_file_systems(region):
    """
    Delete EFS file systems in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting EFS file systems in region: {region}")
    efs = boto3.client('efs', region_name=region)
    try:
        file_systems = efs.describe_file_systems()['FileSystems']
        for fs in file_systems:
            fs_id = fs['FileSystemId']
            if not should_preserve(fs['FileSystemArn']):
                logger.info(f"Deleting EFS file system: {fs_id}")
                try:
                    efs.delete_file_system(FileSystemId=fs_id)
                except ClientError as e:
                    logger.error(f"Error deleting EFS file system {fs_id}: {e}")
    except ClientError as e:
        logger.error(f"Error describing EFS file systems: {e}")

def delete_elbs(region):
    """
    Delete Elastic Load Balancers in the specified region, except those managed by Terraform.

    This function deletes both classic and application/network load balancers in the given region.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting Elastic Load Balancers in region: {region}")
    elb = boto3.client('elb', region_name=region)
    try:
        load_balancers = elb.describe_load_balancers()['LoadBalancerDescriptions']
        for lb in load_balancers:
            lb_name = lb['LoadBalancerName']
            if not should_preserve(lb['LoadBalancerArn']):
                logger.info(f"Deleting ELB: {lb_name}")
                try:
                    elb.delete_load_balancer(LoadBalancerName=lb_name)
                except ClientError as e:
                    logger.error(f"Error deleting ELB {lb_name}: {e}")
    except ClientError as e:
        logger.error(f"Error describing ELBs: {e}")

    elbv2 = boto3.client('elbv2', region_name=region)
    try:
        load_balancers = elbv2.describe_load_balancers()['LoadBalancers']
        for lb in load_balancers:
            lb_arn = lb['LoadBalancerArn']
            if not should_preserve(lb_arn):
                logger.info(f"Deleting ELBv2: {lb_arn}")
                try:
                    elbv2.delete_load_balancer(LoadBalancerArn=lb_arn)
                except ClientError as e:
                    logger.error(f"Error deleting ELBv2 {lb_arn}: {e}")
    except ClientError as e:
        logger.error(f"Error describing ELBv2s: {e}")

def delete_cloudwatch_logs(region):
    """
    Delete CloudWatch log groups in the specified region, except those managed by Terraform.

    :param region: AWS region to target
    :type region: str
    """
    logger.info(f"Deleting CloudWatch logs in region: {region}")
    logs = boto3.client('logs', region_name=region)
    try:
        paginator = logs.get_paginator('describe_log_groups')
        for page in paginator.paginate():
            for group in page['logGroups']:
                group_name = group['logGroupName']
                if not should_preserve(group['arn']):
                    logger.info(f"Deleting log group: {group_name}")
                    try:
                        logs.delete_log_group(logGroupName=group_name)
                    except ClientError as e:
                        logger.error(f"Error deleting log group {group_name}: {e}")
    except ClientError as e:
        logger.error(f"Error describing log groups: {e}")

def delete_resources_in_region(region):
    """
    Delete all supported AWS resources in the specified region, except those managed by Terraform.

    This function calls individual resource deletion functions for the given region.

    :param region: AWS region to target
    :type region: str
    """
    delete_cloudformation_stacks(region)
    delete_ec2_resources(region)
    delete_s3_buckets(region)
    delete_lambda_functions(region)
    delete_dynamodb_tables(region)
    delete_rds_instances(region)
    delete_elasticache_clusters(region)
    delete_efs_file_systems(region)
    delete_elbs(region)
    delete_cloudwatch_logs(region)

def main():
    """
    Main function to orchestrate the AWS resource cleanup process.

    This function defines the regions to clean up and uses a ThreadPoolExecutor
    to delete resources in parallel across all specified regions.
    """
    import os
    print(os.getcwd())

    # Path to your Terraform state file
    terraform_state_path = '../terraform/.terraform/terraform.tfstate'

    # Read Terraform state
    read_terraform_state(terraform_state_path)

    try:
        # Get list of all regions
        ec2 = boto3.client('ec2', region_name='us-east-1')
        regions = [region['RegionName'] for region in ec2.describe_regions()['Regions']]

        # Use ThreadPoolExecutor to delete resources in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(regions)) as executor:
            executor.map(delete_resources_in_region, regions)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
