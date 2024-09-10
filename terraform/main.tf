# ---------------------------------------------------------------------------------------------------------------------
# TERRAFORM CONFIGURATION
# ---------------------------------------------------------------------------------------------------------------------
# This block configures Terraform itself, specifying the required provider versions and backend configuration.

terraform {
  # Specify the required provider and its version
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.66.0"
    }
  }

  # Specify the required Terraform version
  required_version = "= 1.9.5"

  # Configure the S3 backend for storing Terraform state
  backend "s3" {
    bucket         = "terraform-state-bucket-f3b7a9c1"
    dynamodb_table = "terraform-state-lock-e2d8b0a5"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# LOCALS
# ---------------------------------------------------------------------------------------------------------------------
# Define local variables for reuse throughout the configuration

locals {
  # List of email addresses for user creation and notifications
  emails = [
    "jagustin@sandiego.edu",
    "lvo@sandiego.edu",
    "zrobertson@sandiego.edu"
  ]
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS ORGANIZATION
# ---------------------------------------------------------------------------------------------------------------------
# Create an AWS Organization to manage multiple AWS accounts

resource "aws_organizations_organization" "org" {
  feature_set = "ALL"  # Enable all features for the organization
}

# ---------------------------------------------------------------------------------------------------------------------
# IAM USERS AND GROUPS
# ---------------------------------------------------------------------------------------------------------------------
# Create IAM users, a group, and assign permissions

# Create IAM users based on the email addresses
resource "aws_iam_user" "users" {
  count = length(local.emails)
  name  = split("@", local.emails[count.index])[0]  # Use the part before @ as the username

  tags = {
    Email       = local.emails[count.index]
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
}

# Set up login profiles for the IAM users
resource "aws_iam_user_login_profile" "users" {
  count                   = length(local.emails)
  user                    = aws_iam_user.users[count.index].name
  password_reset_required = true

  lifecycle {
    ignore_changes = [password_reset_required]
  }
}

# Create an IAM group for organization users
resource "aws_iam_group" "org_users" {
  name = "OrganizationUsers"
}

# Add all created users to the OrganizationUsers group
resource "aws_iam_group_membership" "org_users" {
  name  = "OrganizationUsersMembership"
  users = aws_iam_user.users[*].name
  group = aws_iam_group.org_users.name
}

# Attach the ReadOnlyAccess policy to the OrganizationUsers group
resource "aws_iam_group_policy_attachment" "read_only_access" {
  group      = aws_iam_group.org_users.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

# ---------------------------------------------------------------------------------------------------------------------
# S3 BUCKET FOR TERRAFORM STATE
# ---------------------------------------------------------------------------------------------------------------------
# Create and configure an S3 bucket to store Terraform state files

resource "aws_s3_bucket" "terraform_state" {
  bucket = "terraform-state-bucket-f3b7a9c1"

  tags = {
    Name        = "Terraform State"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [bucket]
  }
}

# Enable versioning on the S3 bucket
resource "aws_s3_bucket_versioning" "enabled" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption for the S3 bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "default" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket                  = aws_s3_bucket.terraform_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------------------------------------------------
# DYNAMODB TABLE FOR TERRAFORM STATE LOCKING
# ---------------------------------------------------------------------------------------------------------------------
# Create a DynamoDB table for Terraform state locking

resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-state-lock-e2d8b0a5"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name        = "Terraform State Locks"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS BUDGETS
# ---------------------------------------------------------------------------------------------------------------------
# Set up AWS Budgets for cost management

# Create an organization-wide budget
resource "aws_budgets_budget" "organization_wide" {
  name         = "OrganizationWideBudget"
  budget_type  = "COST"
  limit_amount = "0.01"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Define budget notifications
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 10
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = local.emails
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 25
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = local.emails
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }
}

# Create individual budgets for each user
resource "aws_budgets_budget" "individual" {
  count        = length(local.emails)
  name         = "IndividualBudget-${aws_iam_user.users[count.index].name}"
  budget_type  = "COST"
  limit_amount = "0.01"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Define budget notifications for individual users
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 10
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [local.emails[count.index]]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 25
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [local.emails[count.index]]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.emails[count.index]]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [local.emails[count.index]]
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }

  depends_on = [aws_iam_user.users]
}