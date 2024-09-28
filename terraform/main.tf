# ---------------------------------------------------------------------------------------------------------------------
# TERRAFORM CONFIGURATION
# ---------------------------------------------------------------------------------------------------------------------
# Terraform is an Infrastructure as Code (IaC) tool that allows you to define and provision infrastructure resources.
# This section specifies the Terraform settings and required providers.

terraform {
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
  region = "us-east-1"
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS ORGANIZATION
# ---------------------------------------------------------------------------------------------------------------------
# AWS Organizations is a service that enables you to centrally manage and govern multiple AWS accounts.

resource "aws_organizations_organization" "org" {
  feature_set = "ALL"
}

# ---------------------------------------------------------------------------------------------------------------------
# IDENTITY AND ACCESS MANAGEMENT (IAM) USERS AND GROUPS
# ---------------------------------------------------------------------------------------------------------------------
# IAM is a web service that helps you securely control access to AWS resources.

# Create IAM user - jagustin
resource "aws_iam_user" "jagustin" {
  name = "jagustin"
  tags = {
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
}

# Create IAM user - lvo
resource "aws_iam_user" "lvo" {
  name = "lvo"
  tags = {
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
}

# Create IAM user - zrobertson
resource "aws_iam_user" "zrobertson" {
  name = "zrobertson"
  tags = {
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
}

# Set up login profiles for the IAM users
resource "aws_iam_user_login_profile" "jagustin" {
  user                    = "jagustin"
  password_reset_required = true
  lifecycle {
    ignore_changes = [password_reset_required]
  }
}

resource "aws_iam_user_login_profile" "lvo" {
  user                    = "lvo"
  password_reset_required = true
  lifecycle {
    ignore_changes = [password_reset_required]
  }
}

resource "aws_iam_user_login_profile" "zrobertson" {
  user                    = "zrobertson"
  password_reset_required = true
  lifecycle {
    ignore_changes = [password_reset_required]
  }
}

# Create an IAM group
resource "aws_iam_group" "organization_users" {
  name = "OrganizationUsers"
}

# Add users to the IAM group
resource "aws_iam_group_membership" "organization_users" {
  name  = "OrganizationUsersMembership"
  users = ["jagustin", "lvo", "zrobertson"]
  group = "OrganizationUsers"
}

# Create an IAM policy for administrator access
resource "aws_iam_policy" "administrator_access_policy" {
  name        = "AdministratorAccessPolicy"
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

# Create an IAM policy for S3 bucket versioning access
resource "aws_iam_policy" "s3_versioning_access_policy" {
  name        = "S3VersioningAccessPolicy"
  description = "Policy granting access to S3 bucket versioning"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect   = "Allow",
        Action   = "s3:GetBucketVersioning",
        Resource = "arn:aws:s3:::terraform-state-bucket-eeb973f4"
      }
    ]
  })
}

resource "aws_iam_group_policy_attachment" "administrator_access_policy_attachment" {
  group      = "OrganizationUsers"
  policy_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/AdministratorAccessPolicy"
}

resource "aws_iam_group_policy_attachment" "s3_versioning_access_policy_attachment" {
  group      = "OrganizationUsers"
  policy_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/S3VersioningAccessPolicy"
}

# ---------------------------------------------------------------------------------------------------------------------
# SIMPLE STORAGE SERVICE (S3) BUCKET FOR TERRAFORM STATE
# ---------------------------------------------------------------------------------------------------------------------
# Amazon S3 is an object storage service offering industry-leading scalability, data availability, security, and performance.

resource "aws_s3_bucket" "terraform_state" {
  bucket = "terraform-state-bucket-eeb973f4"

  tags = {
    Name        = "Terraform State"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    prevent_destroy = true     # Prevent accidental deletion of this bucket
    ignore_changes  = [bucket] # Ignore changes to the bucket name
  }
}

# Enable versioning on the S3 bucket
resource "aws_s3_bucket_versioning" "enabled" {
  bucket = "terraform-state-bucket-eeb973f4"
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption for the S3 bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "default" {
  bucket = "terraform-state-bucket-eeb973f4"
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket                  = "terraform-state-bucket-eeb973f4"
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Add a policy to enforce SSL-only access to the S3 bucket
resource "aws_s3_bucket_policy" "terraform_state_policy" {
  bucket = "terraform-state-bucket-eeb973f4"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "s3:*",
        Effect    = "Deny",
        Principal = "*",
        Resource = [
          "arn:aws:s3:::terraform-state-bucket-eeb973f4/*",
          "arn:aws:s3:::terraform-state-bucket-eeb973f4"
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
  name         = "terraform-state-lock-eeb973f4"
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
    prevent_destroy = true   # Prevent accidental deletion of this table
    ignore_changes  = [name] # Ignore changes to the table name
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
  resource_arn = "arn:aws:dynamodb:us-east-1:${data.aws_caller_identity.current.account_id}:table/terraform-state-lock-eeb973f4"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Deny",
        Principal = "*",
        Action    = "dynamodb:*",
        Resource  = "arn:aws:dynamodb:us-east-1:${data.aws_caller_identity.current.account_id}:table/terraform-state-lock-eeb973f4",
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
  name         = "SharedFreeTierBudget"
  budget_type  = "COST"
  limit_amount = "1"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Notification when actual spend reaches 20% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 20
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["jagustin@sandiego.edu", "lvo@sandiego.edu", "zrobertson@sandiego.edu"]
  }

  # Notification when actual spend reaches 40% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 40
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["jagustin@sandiego.edu", "lvo@sandiego.edu", "zrobertson@sandiego.edu"]
  }

  # Notification when actual spend reaches 60% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 60
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["jagustin@sandiego.edu", "lvo@sandiego.edu", "zrobertson@sandiego.edu"]
  }

  # Notification when actual spend reaches 80% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["jagustin@sandiego.edu", "lvo@sandiego.edu", "zrobertson@sandiego.edu"]
  }

  # Notification when actual spend reaches 90% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["jagustin@sandiego.edu", "lvo@sandiego.edu", "zrobertson@sandiego.edu"]
  }

  # Notification when forecasted spend reaches 100% of the budget
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["jagustin@sandiego.edu", "lvo@sandiego.edu", "zrobertson@sandiego.edu"]
  }
  lifecycle {
    prevent_destroy = true   # Prevent accidental deletion of this budget
    ignore_changes  = [name] # Ignore changes to the budget name
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# SIMPLE NOTIFICATION SERVICE (SNS) CONFIGURATION
# ---------------------------------------------------------------------------------------------------------------------
# Amazon Simple Notification Service (SNS) is a fully managed messaging service for both application-to-application (A2A)
# and application-to-person (A2P) communication. It enables decoupled microservices, distributed systems, and serverless
# applications to communicate with each other and with users.

# Create an SNS topic for jagustin
resource "aws_sns_topic" "jagustin_notifications" {
  name = "user-notifications-jagustin"
  tags = {
    Name        = "User Notifications"
    Environment = "Management"
    ManagedBy   = "Terraform"
    User        = "jagustin"
  }
}

# Create an SNS topic for lvo
resource "aws_sns_topic" "lvo_notifications" {
  name = "user-notifications-lvo"
  tags = {
    Name        = "User Notifications"
    Environment = "Management"
    ManagedBy   = "Terraform"
    User        = "lvo"
  }
}

# Create an SNS topic for zrobertson
resource "aws_sns_topic" "zrobertson_notifications" {
  name = "user-notifications-zrobertson"
  tags = {
    Name        = "User Notifications"
    Environment = "Management"
    ManagedBy   = "Terraform"
    User        = "zrobertson"
  }
}

# Subscribe jagustin's email to their respective SNS topic
resource "aws_sns_topic_subscription" "jagustin_email_subscriptions" {
  topic_arn = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-jagustin"
  protocol  = "email"
  endpoint  = "jagustin@sandiego.edu"
}

# Subscribe lvo's email to their respective SNS topic
resource "aws_sns_topic_subscription" "lvo_email_subscriptions" {
  topic_arn = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-lvo"
  protocol  = "email"
  endpoint  = "lvo@sandiego.edu"
}

# Subscribe zrobertson's email to their respective SNS topic
resource "aws_sns_topic_subscription" "zrobertson_email_subscriptions" {
  topic_arn = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-zrobertson"
  protocol  = "email"
  endpoint  = "zrobertson@sandiego.edu"
}

# Define a policy document that allows CloudWatch to publish to each SNS topic
data "aws_iam_policy_document" "sns_topic_policy_jagustin" {
  statement {
    effect  = "Allow"
    actions = ["SNS:Publish"]
    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }
    resources = ["arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-jagustin"]
  }
}

data "aws_iam_policy_document" "sns_topic_policy_lvo" {
  statement {
    effect  = "Allow"
    actions = ["SNS:Publish"]
    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }
    resources = ["arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-lvo"]
  }
}

data "aws_iam_policy_document" "sns_topic_policy_zrobertson" {
  statement {
    effect  = "Allow"
    actions = ["SNS:Publish"]
    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }
    resources = ["arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-zrobertson"]
  }
}

resource "aws_sns_topic_policy" "jagustin_default" {
  arn    = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-jagustin"
  policy = data.aws_iam_policy_document.sns_topic_policy_jagustin.json
}

resource "aws_sns_topic_policy" "lvo_default" {
  arn    = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-lvo"
  policy = data.aws_iam_policy_document.sns_topic_policy_lvo.json
}

resource "aws_sns_topic_policy" "zrobertson_default" {
  arn    = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:user-notifications-zrobertson"
  policy = data.aws_iam_policy_document.sns_topic_policy_zrobertson.json
}

# ---------------------------------------------------------------------------------------------------------------------
# CLOUDWATCH ALARM FOR FREE TIER USAGE
# ---------------------------------------------------------------------------------------------------------------------
# Amazon CloudWatch is a monitoring and observability service that provides data and actionable insights for AWS resources.
resource "aws_cloudwatch_metric_alarm" "free_tier_usage_alarm" {
  alarm_name          = "FreeTierUsageAlarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "300"
  statistic           = "Maximum"
  threshold           = "1.00"
  actions_enabled     = true
  alarm_actions       = ["arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:free-tier-alerts"]
  dimensions = {
    Currency = "USD"
  }
}

# Create an SNS topic for Free Tier alerts
resource "aws_sns_topic" "free_tier_alerts" {
  name = "free-tier-alerts"
}


resource "aws_cloudwatch_log_group" "mlops_pipeline_logs" {
  name = "/aws/mlops/pipeline"

  tags = {
    Name        = "MLOps Pipeline Logs"
    Environment = "Development"
    ManagedBy   = "Terraform"
  }
}

# Subscribe jagustin's email to the Free Tier alerts SNS topic
resource "aws_sns_topic_subscription" "free_tier_alerts_email_jagustin" {
  topic_arn = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:free-tier-alerts"
  protocol  = "email"
  endpoint  = "jagustin@sandiego.edu"
}

# Subscribe lvo's email to the Free Tier alerts SNS topic
resource "aws_sns_topic_subscription" "free_tier_alerts_email_lvo" {
  topic_arn = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:free-tier-alerts"
  protocol  = "email"
  endpoint  = "lvo@sandiego.edu"
}

# Subscribe zrobertson's email to the Free Tier alerts SNS topic
resource "aws_sns_topic_subscription" "free_tier_alerts_email_zrobertson" {
  topic_arn = "arn:aws:sns:us-east-1:${data.aws_caller_identity.current.account_id}:free-tier-alerts"
  protocol  = "email"
  endpoint  = "zrobertson@sandiego.edu"
}

# ---------------------------------------------------------------------------------------------------------------------
# S3 BUCKET FOR MLOPS PIPELINE
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = "mlops-artifacts-aai540-group3"
  tags = {
    Name        = "MLOps Artifacts"
    Environment = "Development"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes = [bucket]
  }
}

# Enable versioning on the S3 bucket
resource "aws_s3_bucket_versioning" "mlops_versioning" {
  bucket = "mlops-artifacts-aai540-group3"
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption for the S3 bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_encryption" {
  bucket = "mlops-artifacts-aai540-group3"
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "mlops_public_access" {
  bucket                  = "mlops-artifacts-aai540-group3"
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Add a policy to enforce SSL-only access to the S3 bucket
resource "aws_s3_bucket_policy" "mlops_bucket_policy" {
  bucket = "mlops-artifacts-aai540-group3"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "s3:*",
        Effect    = "Deny",
        Principal = "*",
        Resource = [
          "arn:aws:s3:::mlops-artifacts-aai540-group3/*",
          "arn:aws:s3:::mlops-artifacts-aai540-group3"
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
  name = "GitHub-Actions-PolicyValidator"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = "arn:aws:iam::864899865811:oidc-provider/token.actions.githubusercontent.com"
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com",
            "token.actions.githubusercontent.com:sub" = "repo:aai540-group3/project:ref:refs/heads/main"
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "access_analyzer_policy" {
  name        = "AccessAnalyzerPolicy"
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
  role       = "GitHub-Actions-PolicyValidator"
  policy_arn = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/AccessAnalyzerPolicy"
}

data "aws_caller_identity" "current" {}