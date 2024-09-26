variable "s3_bucket_name" {
  description = "Name of the S3 bucket for Terraform state"
  type        = string
}

variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for Terraform state locking"
  type        = string
}

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
}

variable "org_users" {
  description = "List of email addresses for organization users"
  type        = list(string)
}

variable "iam_group_name" {
  description = "Name of the IAM group for organization users"
  type        = string
}

variable "admin_policy_name" {
  description = "Name of the administrator access policy"
  type        = string
}

variable "budget_name" {
  description = "Name of the AWS budget"
  type        = string
}

variable "budget_limit_amount" {
  description = "Limit amount for the AWS budget"
  type        = string
}

variable "budget_limit_unit" {
  description = "Limit unit for the AWS budget"
  type        = string
}

variable "cloudwatch_alarm_name" {
  description = "Name of the CloudWatch alarm for Free Tier usage"
  type        = string
}

variable "cloudwatch_threshold" {
  description = "Threshold for the CloudWatch alarm"
  type        = string
}

variable "free_tier_alerts_topic_name" {
  description = "Name of the SNS topic for Free Tier alerts"
  type        = string
}

variable "mlops_bucket_name" {
  description = "Name of the S3 bucket for MLOps artifacts"
  type        = string
}

variable "github_actions_role_name" {
  description = "Name of the IAM role for GitHub Actions"
  type        = string
}

variable "github_org" {
  description = "Name of the GitHub organization"
  type        = string
}

variable "github_repo" {
  description = "Name of the GitHub repository"
  type        = string
}

variable "aws_account_id" {
  description = "AWS account ID"
  type        = string
}

variable "access_analyzer_policy_name" {
  description = "Name of the Access Analyzer policy"
  type        = string
}

# ---------------------------------------------------------------------------------------------------------------------
# TERRAFORM CONFIGURATION
# ---------------------------------------------------------------------------------------------------------------------
# Terraform is an Infrastructure as Code (IaC) tool that allows you to define and provision infrastructure resources.
# This section specifies the Terraform settings and required providers.

terraform {
  # Define the required providers and their versions
  required_providers {
    aws = {
      source  = "hashicorp/aws" # The source of the AWS provider
      version = "~> 5.66.0"     # The version of the AWS provider to use
    }
    random = {
      source  = "hashicorp/random"
      version = "3.6.2"
    }
  }

  # Specify the required Terraform version
  required_version = ">= 1.5.0" # Use a valid Terraform version

  # Configure the backend for storing Terraform state
  # A backend determines how state is stored. Here, we're using an S3 bucket.
  backend "s3" {
    bucket         = "terraform-state-bucket-eeb973f4"  # The name of the S3 bucket to store state
    dynamodb_table = "terraform-state-lock-eeb973f4"    # DynamoDB table for state locking
    key            = "infrastructure/terraform.tfstate" # The path to the state file in the S3 bucket
    region         = "us-east-1"                        # The AWS region where the S3 bucket is located
    encrypt        = true                               # Enable encryption for the state file
  }
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------------------------------------------------
# LOCALS
# ---------------------------------------------------------------------------------------------------------------------
# Locals are named values that can be used throughout your Terraform configuration.

locals {
  # List of email addresses for the users
  emails = var.org_users
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS ORGANIZATION
# ---------------------------------------------------------------------------------------------------------------------
# AWS Organizations is a service that enables you to centrally manage and govern multiple AWS accounts.

resource "aws_organizations_organization" "org" {
  feature_set = "ALL" # Enable all features of AWS Organizations
}

# ---------------------------------------------------------------------------------------------------------------------
# IDENTITY AND ACCESS MANAGEMENT (IAM) USERS AND GROUPS
# ---------------------------------------------------------------------------------------------------------------------
# IAM is a web service that helps you securely control access to AWS resources.

# Create IAM users
resource "aws_iam_user" "users" {
  count = length(local.emails)                     # Create a user for each email in the list
  name  = split("@", local.emails[count.index])[0] # Use the part before @ as the username

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
  password_reset_required = true # Force users to change password on first login

  lifecycle {
    ignore_changes = [password_reset_required]
  }
}

# Create an IAM group
resource "aws_iam_group" "org_users" {
  name = var.iam_group_name
}

# Add users to the IAM group
resource "aws_iam_group_membership" "org_users" {
  name  = "${var.iam_group_name}Membership"
  users = aws_iam_user.users[*].name
  group = aws_iam_group.org_users.name
}

# Create an IAM policy for administrator access
resource "aws_iam_policy" "administrator_access_policy" {
  name        = var.admin_policy_name
  description = "Policy granting administrator access"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "*",
        Resource = "*"
      }
    ]
  })
}

# Attach the administrator access policy to the IAM group
resource "aws_iam_group_policy_attachment" "administrator_access_policy_attachment" {
  group      = aws_iam_group.org_users.name
  policy_arn = aws_iam_policy.administrator_access_policy.arn
}

# ---------------------------------------------------------------------------------------------------------------------
# SIMPLE STORAGE SERVICE (S3) BUCKET FOR TERRAFORM STATE
# ---------------------------------------------------------------------------------------------------------------------
# Amazon S3 is an object storage service offering industry-leading scalability, data availability, security, and performance.

resource "aws_s3_bucket" "terraform_state" {
  bucket = var.s3_bucket_name

  tags = {
    Name        = "Terraform State"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    # prevent_destroy = true     # Prevent accidental deletion of this bucket
    ignore_changes = [bucket] # Ignore changes to the bucket name
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

# Add a policy to enforce SSL-only access to the S3 bucket
resource "aws_s3_bucket_policy" "terraform_state_policy" {
  bucket = aws_s3_bucket.terraform_state.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "s3:*",
        Effect    = "Deny",
        Principal = "*",
        Resource = [
          "${aws_s3_bucket.terraform_state.arn}/*",
          "${aws_s3_bucket.terraform_state.arn}"
        ],
        Condition = {
          Bool : {
            "aws:SecureTransport" : "false"
          }
        }
      }
    ]
  })
}

# ---------------------------------------------------------------------------------------------------------------------
# DYNAMODB TABLE FOR TERRAFORM STATE LOCKING
# ---------------------------------------------------------------------------------------------------------------------
# Amazon DynamoDB is a key-value and document database that delivers single-digit millisecond performance at any scale.

# DynamoDB table for Terraform state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST" # Pay only for what you use
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S" # String type
  }

  tags = {
    Name        = "Terraform State Locks"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    # prevent_destroy = true   # Prevent accidental deletion of this table
    ignore_changes = [name] # Ignore changes to the table name
  }

  server_side_encryption {
    enabled = true
  }

  point_in_time_recovery {
    enabled = true
  }
}

# Policy to enforce SSL-only access to the DynamoDB table
resource "aws_dynamodb_resource_policy" "terraform_locks_policy" {
  resource_arn = aws_dynamodb_table.terraform_locks.arn
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Deny",
        Principal = "*",
        Action    = "dynamodb:*",
        Resource  = aws_dynamodb_table.terraform_locks.arn,
        Condition = {
          Bool : {
            "aws:SecureTransport" : "false"
          }
        }
      }
    ]
  })
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS BUDGETS
# ---------------------------------------------------------------------------------------------------------------------
# AWS Budgets gives you the ability to set custom budgets that alert you when your costs or usage exceed your budgeted amount.

resource "aws_budgets_budget" "shared_user_budget" {
  name         = var.budget_name
  budget_type  = "COST"
  limit_amount = var.budget_limit_amount
  limit_unit   = var.budget_limit_unit
  time_unit    = "MONTHLY"

  # Notification when actual spend reaches 20% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 20
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  # Notification when actual spend reaches 40% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 40
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  # Notification when actual spend reaches 60% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 60
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  # Notification when actual spend reaches 80% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  # Notification when actual spend reaches 90% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  # Notification when forecasted spend reaches 100% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = local.emails
  }

  lifecycle {
    # prevent_destroy = true   # Prevent accidental deletion of this budget
    ignore_changes = [name] # Ignore changes to the budget name
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# SIMPLE NOTIFICATION SERVICE (SNS) CONFIGURATION
# ---------------------------------------------------------------------------------------------------------------------
# Amazon Simple Notification Service (SNS) is a fully managed messaging service for both application-to-application (A2A)
# and application-to-person (A2P) communication. It enables decoupled microservices, distributed systems, and serverless
# applications to communicate with each other and with users.

# Create an SNS topic for each user
resource "aws_sns_topic" "user_notifications" {
  count = length(local.emails)
  name  = "user-notifications-${split("@", local.emails[count.index])[0]}"

  tags = {
    Name        = "User Notifications"
    Environment = "Management"
    ManagedBy   = "Terraform"
    User        = split("@", local.emails[count.index])[0]
  }
}

# Subscribe each user's email to their respective SNS topic
resource "aws_sns_topic_subscription" "user_email_subscriptions" {
  count     = length(local.emails)
  topic_arn = aws_sns_topic.user_notifications[count.index].arn
  protocol  = "email"
  endpoint  = local.emails[count.index]
}

# Define a policy document that allows CloudWatch to publish to each SNS topic
data "aws_iam_policy_document" "sns_topic_policy" {
  count = length(local.emails)

  statement {
    effect  = "Allow"
    actions = ["SNS:Publish"]

    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }

    resources = [aws_sns_topic.user_notifications[count.index].arn]
  }
}

# Attach the policy to each SNS topic
resource "aws_sns_topic_policy" "default" {
  count  = length(local.emails)
  arn    = aws_sns_topic.user_notifications[count.index].arn
  policy = data.aws_iam_policy_document.sns_topic_policy[count.index].json
}

# ---------------------------------------------------------------------------------------------------------------------
# CLOUDWATCH ALARM FOR FREE TIER USAGE
# ---------------------------------------------------------------------------------------------------------------------
# Amazon CloudWatch is a monitoring and observability service that provides data and actionable insights for AWS resources.

resource "aws_cloudwatch_metric_alarm" "free_tier_usage_alarm" {
  alarm_name          = var.cloudwatch_alarm_name
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "300"
  statistic           = "Maximum"
  threshold           = var.cloudwatch_threshold
  actions_enabled     = true
  alarm_actions       = [aws_sns_topic.free_tier_alerts.arn]

  dimensions = {
    Currency = "USD"
  }
}

# Create an SNS topic for Free Tier alerts
resource "aws_sns_topic" "free_tier_alerts" {
  name = var.free_tier_alerts_topic_name
}

# Subscribe emails to the Free Tier alerts SNS topic
resource "aws_sns_topic_subscription" "free_tier_alerts_email" {
  count     = length(local.emails)
  topic_arn = aws_sns_topic.free_tier_alerts.arn
  protocol  = "email"
  endpoint  = local.emails[count.index]
}

# ---------------------------------------------------------------------------------------------------------------------
# S3 BUCKET FOR MLOPS PIPELINE
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = var.mlops_bucket_name

  tags = {
    Name        = "MLOps Artifacts"
    Environment = "Development"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    # prevent_destroy = true     # Prevent accidental deletion of this bucket
    ignore_changes = [bucket] # Ignore changes to the bucket name
  }
}

# Enable versioning on the S3 bucket
resource "aws_s3_bucket_versioning" "mlops_versioning" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption for the S3 bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_encryption" {
  bucket = aws_s3_bucket.mlops_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "mlops_public_access" {
  bucket                  = aws_s3_bucket.mlops_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Add a policy to enforce SSL-only access to the S3 bucket
resource "aws_s3_bucket_policy" "mlops_bucket_policy" {
  bucket = aws_s3_bucket.mlops_artifacts.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "s3:*",
        Effect    = "Deny",
        Principal = "*",
        Resource = [
          "${aws_s3_bucket.mlops_artifacts.arn}/*",
          "${aws_s3_bucket.mlops_artifacts.arn}"
        ],
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# ---------------------------------------------------------------------------------------------------------------------
# IAM ROLE FOR GITHUB ACTIONS POLICY VALIDATION
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_iam_role" "github_actions_policy_validator" {
  name = var.github_actions_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = "arn:aws:iam::${var.aws_account_id}:oidc-provider/token.actions.githubusercontent.com"
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com",
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_org}/${var.github_repo}:ref:refs/heads/main"
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "access_analyzer_policy" {
  name        = var.access_analyzer_policy_name
  path        = "/"
  description = "IAM policy for Access Analyzer"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "access-analyzer:*",
          "iam:GetRole",
          "iam:ListRoles",
          "organizations:DescribeAccount",
          "organizations:DescribeOrganization",
          "organizations:ListAccounts"
        ]
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "access_analyzer_attachment" {
  role       = aws_iam_role.github_actions_policy_validator.name
  policy_arn = aws_iam_policy.access_analyzer_policy.arn
}