#!/bin/bash

set -e

# Generate a random 8-character hash
RANDOM_HASH=$(openssl rand -hex 4)

# Single point of configuration
AWS_REGION='us-east-1'
S3_BUCKET_NAME=${S3_BUCKET_NAME:-"terraform-state-bucket-$RANDOM_HASH"}
DYNAMODB_TABLE_NAME=${DYNAMODB_TABLE_NAME:-"terraform-state-lock-$RANDOM_HASH"}
CREDENTIALS_FILE="$HOME/.aws/credentials"
CONFIG_FILE="$HOME/.aws/config"

# Function to create resources
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
    fi

    # Write AWS credentials
    mkdir -p "$(dirname "$CREDENTIALS_FILE")"
    cat << EOF > "$CREDENTIALS_FILE"
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
    echo "STATUS: AWS CREDENTIALS WRITTEN SUCCESSFULLY TO $CREDENTIALS_FILE"

    # Write AWS config
    mkdir -p "$(dirname "$CONFIG_FILE")"
    cat << EOF > "$CONFIG_FILE"
[default]
region = $AWS_REGION
EOF
    echo "STATUS: AWS CONFIG WRITTEN SUCCESSFULLY TO $CONFIG_FILE"

    # Check AWS authentication
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "ERROR: AWS AUTHENTICATION FAILED"
        exit 1
    fi

    echo "STATUS: AWS AUTHENTICATION SUCCESSFUL"

    # Create S3 bucket
    echo "CREATING S3 BUCKET '$S3_BUCKET_NAME'..."
    if aws s3api create-bucket --bucket "$S3_BUCKET_NAME" --region "$AWS_REGION"; then
        echo "STATUS: S3 BUCKET '$S3_BUCKET_NAME' CREATED SUCCESSFULLY"
    else
        echo "ERROR: FAILED TO CREATE S3 BUCKET '$S3_BUCKET_NAME'"
        exit 1
    fi

    # Create DynamoDB table
    echo "CREATING DYNAMODB TABLE '$DYNAMODB_TABLE_NAME'..."
    if aws dynamodb create-table \
        --table-name "$DYNAMODB_TABLE_NAME" \
        --attribute-definitions AttributeName=LockID,AttributeType=S \
        --key-schema AttributeName=LockID,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region "$AWS_REGION"; then
        echo "STATUS: DYNAMODB TABLE '$DYNAMODB_TABLE_NAME' CREATED SUCCESSFULLY"
    else
        echo "ERROR: FAILED TO CREATE DYNAMODB TABLE '$DYNAMODB_TABLE_NAME'"
        exit 1
    fi

    echo "---- AWS BOOTSTRAP COMPLETE ----"
    echo "S3 Bucket Name: $S3_BUCKET_NAME"
    echo "DynamoDB Table Name: $DYNAMODB_TABLE_NAME"
    
    # Save the resource names to a file for future reference
    echo "S3_BUCKET_NAME=$S3_BUCKET_NAME" > aws_resources.env
    echo "DYNAMODB_TABLE_NAME=$DYNAMODB_TABLE_NAME" >> aws_resources.env
    echo "Resource names saved to aws_resources.env"

    # Save the resource names to a file for Terraform
    cat << EOF > terraform.tfvars
s3_bucket_name = "$S3_BUCKET_NAME"
dynamodb_table_name = "$DYNAMODB_TABLE_NAME"
EOF
    echo "Resource names saved to terraform.tfvars"
}

# Function to undo changes
undo_changes() {
    echo "---- STARTING UNDO PROCESS ----"

    # Load resource names if available
    if [ -f aws_resources.env ]; then
        source aws_resources.env
    fi

    # Remove S3 bucket
    if [ -n "$S3_BUCKET_NAME" ] && aws s3api head-bucket --bucket "$S3_BUCKET_NAME" 2>/dev/null; then
        echo "Deleting S3 bucket: $S3_BUCKET_NAME"
        aws s3 rb "s3://$S3_BUCKET_NAME" --force
    else
        echo "S3 bucket $S3_BUCKET_NAME does not exist or is not accessible"
    fi

    # Remove DynamoDB table
    if [ -n "$DYNAMODB_TABLE_NAME" ] && aws dynamodb describe-table --table-name "$DYNAMODB_TABLE_NAME" &> /dev/null; then
        echo "Deleting DynamoDB table: $DYNAMODB_TABLE_NAME"
        aws dynamodb delete-table --table-name "$DYNAMODB_TABLE_NAME"
    else
        echo "DynamoDB table $DYNAMODB_TABLE_NAME does not exist or is not accessible"
    fi

    # Remove AWS config files
    rm -f "$CREDENTIALS_FILE" "$CONFIG_FILE"
    echo "Removed AWS credential and config files"

    # Remove the resource names file
    rm -f aws_resources.env
    rm -f terraform.tfvars
    echo "Removed aws_resources.env and terraform.tfvars files"

    echo "---- UNDO PROCESS COMPLETE ----"
}

# Main script logic
if [ "$1" = "undo" ]; then
    undo_changes
else
    # Install required tools
    pip install uv --quiet --progress-bar=off
    uv pip install --quiet --system boto3
    create_resources
fi