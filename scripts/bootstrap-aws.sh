#!/usr/bin/env bash
#===============================================================================
# Title           :aws_bootstrap.sh
# Description     :This script bootstraps AWS resources required for Terraform,
#                  such as S3 buckets for state storage and DynamoDB tables for
#                  state locking.
# Version         :1.1
# Usage           :./aws_bootstrap.sh
#===============================================================================
set -e

#---------------------------------------
# Configuration Variables
#---------------------------------------
AWS_REGION='us-east-1'
S3_BUCKET_NAME=${S3_BUCKET_NAME:-"terraform-state-bucket-eeb973f4"}
DYNAMODB_TABLE_NAME=${DYNAMODB_TABLE_NAME:-"terraform-state-lock-eeb973f4"}
CREDENTIALS_FILE="$HOME/.aws/credentials"
CONFIG_FILE="$HOME/.aws/config"

#---------------------------------------
# Function: create_resources
# Description:
#   Creates AWS resources required for Terraform state management.
#   - Writes AWS credentials and config files.
#   - Creates an S3 bucket for Terraform state storage.
#   - Creates a DynamoDB table for Terraform state locking.
#---------------------------------------
create_resources() {
    echo "---- STARTING AWS BOOTSTRAP ----"

    # Get AWS credentials
    if [ -z "$AWS_ACCESS_KEY_ID" ]; then
        echo "Enter your AWS Access Key ID:"
        read -r AWS_ACCESS_KEY_ID
    fi
    if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "Enter your AWS Secret Access Key:"
        read -rs AWS_SECRET_ACCESS_KEY
        echo
    fi

    # Write AWS credentials
    mkdir -p "$(dirname "$CREDENTIALS_FILE")"
    cat << EOF > "$CREDENTIALS_FILE"
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
    echo "STATUS: AWS credentials written successfully to $CREDENTIALS_FILE"

    # Write AWS config
    mkdir -p "$(dirname "$CONFIG_FILE")"
    cat << EOF > "$CONFIG_FILE"
[default]
region = $AWS_REGION
EOF
    echo "STATUS: AWS config written successfully to $CONFIG_FILE"

    # Check AWS authentication
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "ERROR: AWS authentication failed"
        exit 1
    fi
    echo "STATUS: AWS authentication successful"

    # Create S3 bucket
    echo "Creating S3 bucket '$S3_BUCKET_NAME'..."
    if aws s3api create-bucket --bucket "$S3_BUCKET_NAME" --region "$AWS_REGION"; then
        echo "STATUS: S3 bucket '$S3_BUCKET_NAME' created successfully"
    else
        echo "ERROR: Failed to create S3 bucket '$S3_BUCKET_NAME'"
        exit 1
    fi


    # Check if the DynamoDB table already exists
    if aws dynamodb describe-table --table-name terraform-state-lock-eeb973f4 > /dev/null 2>&1; then
    echo "STATUS: DynamoDB table 'terraform-state-lock-eeb973f4' already exists. Skipping creation."
    else
    # Code to create the DynamoDB table (keep this part commented out)
    aws dynamodb create-table \
      --table-name terraform-state-lock-eeb973f4 \
      --attribute-definitions AttributeName=LockID,AttributeType=S \
      --key-schema AttributeName=LockID,KeyType=HASH \
      --provisioned-throughput ReadCapacityUnits=1,WriteCapacityUnits=1
    echo "STATUS: DynamoDB table 'terraform-state-lock-eeb973f4' created successfully"
    fi

    echo "---- AWS BOOTSTRAP COMPLETE ----"
}

#---------------------------------------
# Function: print_status
# Description:
#   Prints the status and configuration of created resources
#---------------------------------------
print_status() {
    echo "---- PRINTING STATUS AND CONFIGURATION ----"

    # Print S3 bucket status
    echo "Checking S3 bucket status..."
    if aws s3api head-bucket --bucket "$S3_BUCKET_NAME" 2>/dev/null; then
        echo "STATUS: S3 bucket '$S3_BUCKET_NAME' exists and is accessible"
        # Print bucket details
        aws s3api get-bucket-location --bucket "$S3_BUCKET_NAME"
    else
        echo "ERROR: S3 bucket '$S3_BUCKET_NAME' is not accessible or does not exist"
    fi

    # Print DynamoDB table status
    echo "Checking DynamoDB table status..."
    if aws dynamodb describe-table --table-name "$DYNAMODB_TABLE_NAME" --region "$AWS_REGION" 2>/dev/null; then
        echo "STATUS: DynamoDB table '$DYNAMODB_TABLE_NAME' exists and is accessible"
    else
        echo "ERROR: DynamoDB table '$DYNAMODB_TABLE_NAME' is not accessible or does not exist"
    fi

    # Print AWS configuration
    echo "AWS Configuration:"
    echo "Region: $AWS_REGION"
    echo "Credentials file: $CREDENTIALS_FILE"
    echo "Config file: $CONFIG_FILE"

    echo "---- STATUS AND CONFIGURATION PRINTOUT COMPLETE ----"
}

#---------------------------------------
# Main Script Logic
#---------------------------------------
# Ensure required AWS CLI tools are installed
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI is not installed. Please install it before running this script."
    exit 1
fi

create_resources
print_status