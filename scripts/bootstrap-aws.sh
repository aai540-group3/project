#!/usr/bin/env bash
#===============================================================================
# Title           :aws_bootstrap.sh
# Description     :This script bootstraps AWS resources required for Terraform,
#                  such as S3 buckets for state storage and DynamoDB tables for
#                  state locking.
# Version         :1.0
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

    # Create DynamoDB table
    echo "Creating DynamoDB table '$DYNAMODB_TABLE_NAME'..."
    if aws dynamodb create-table \
        --table-name "$DYNAMODB_TABLE_NAME" \
        --attribute-definitions AttributeName=LockID,AttributeType=S \
        --key-schema AttributeName=LockID,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region "$AWS_REGION"; then
        echo "STATUS: DynamoDB table '$DYNAMODB_TABLE_NAME' created successfully"
    else
        echo "ERROR: Failed to create DynamoDB table '$DYNAMODB_TABLE_NAME'"
        exit 1
    fi

    echo "---- AWS BOOTSTRAP COMPLETE ----"
    echo "S3 Bucket Name: $S3_BUCKET_NAME"
    echo "DynamoDB Table Name: $DYNAMODB_TABLE_NAME"
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
fi
