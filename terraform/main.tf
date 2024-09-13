variable "s3_bucket_name" {
  description = "Name of the S3 bucket for Terraform state"
  type        = string
}

variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for Terraform state locking"
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
    asana = {
      source  = "davidji99/asana"
      version = "0.1.2"
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

provider "asana" {}

# ---------------------------------------------------------------------------------------------------------------------
# LOCALS
# ---------------------------------------------------------------------------------------------------------------------
# Locals are named values that can be used throughout your Terraform configuration.

locals {
  # List of email addresses for the users
  emails = [
    "jagustin@sandiego.edu",
    "lvo@sandiego.edu",
    "zrobertson@sandiego.edu"
  ]
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
  name = "OrganizationUsers"
}

# Add users to the IAM group
resource "aws_iam_group_membership" "org_users" {
  name  = "OrganizationUsersMembership"
  users = aws_iam_user.users[*].name
  group = aws_iam_group.org_users.name
}

# Create an IAM policy for common AWS services with instance type restrictions
resource "aws_iam_policy" "common_services_policy" {
  name        = "CommonServicesPolicy"
  description = "Policy allowing access to common AWS services"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "apigateway:*",
          "cloudwatch:*",
          "codebuild:*",
          "codecommit:*",
          "codepipeline:*",
          "dynamodb:*",
          "ec2:Describe*",
          "ecr:*",
          "iam:PassRole",
          "lambda:*",
          "logs:*",
          "s3:*",
          "sagemaker:*",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances",
          "ec2:StartInstances",
          "ec2:StopInstances",
          "ec2:TerminateInstances",
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:InstanceType" = [
              "t2.micro",
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "sns:GetTopicAttributes",
          "sns:ListSubscriptionsByTopic",
          "sns:ListTopics",
          "sns:Subscribe",
          "sns:Unsubscribe",
        ]
        Resource = aws_sns_topic.user_notifications[*].arn
      },
      # Allow stopping and terminating instances based on tags
      {
        Effect = "Allow"
        Action = [
          "ec2:StopInstances",
          "ec2:TerminateInstances"
        ]
        Resource = "arn:aws:ec2:*:*:instance/*"
        Condition = {
          StringEquals = {
            "ec2:ResourceTag/AutoShutdown" : "true"
          }
        }
      }
    ]
  })

  # Ensure that the SNS topics are created before referencing them in this policy
  depends_on = [aws_sns_topic.user_notifications]
}

# Attach the policy to the IAM group
resource "aws_iam_group_policy_attachment" "common_services_policy_attachment" {
  group      = aws_iam_group.org_users.name
  policy_arn = aws_iam_policy.common_services_policy.arn
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
  alarm_name          = "FreeTierUsageAlarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "300"
  statistic           = "Maximum"
  threshold           = "1.00"
  actions_enabled     = true
  alarm_actions       = [aws_sns_topic.free_tier_alerts.arn]

  dimensions = {
    Currency = "USD"
  }
}

# Create an SNS topic for Free Tier alerts
resource "aws_sns_topic" "free_tier_alerts" {
  name = "free-tier-alerts"
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
  bucket = "mlops-artifacts-aai540-group3"

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
# EC2 INSTANCE FOR MLOPS PIPELINE
# ---------------------------------------------------------------------------------------------------------------------

# Create a VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name        = "MLOps-VPC"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}

# Create an Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "MLOps-IGW"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}

# Create subnets
resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true

  tags = {
    Name        = "MLOps-Subnet-Main"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}

# Create a route table
resource "aws_route_table" "main" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "MLOps-RouteTable"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}

# Associate the route table with the main subnet
resource "aws_route_table_association" "main" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.main.id
}

# Create a NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.main.id

  tags = {
    Name        = "MLOps-NATGateway"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}

# Create a security group
resource "aws_security_group" "mlops_sg" {
  name        = "mlops_sg"
  description = "Security group for MLOps pipeline"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["140.82.112.0/20", "185.199.108.0/22", "192.30.252.0/22"] # GitHub Codespaces IP ranges
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "MLOps-SecurityGroup"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}

# Create an EC2 instance
resource "aws_instance" "mlops_instance" {
  ami                    = "ami-0182f373e66f89c85" # Amazon Linux 2023 AMI 2023.5.20240903.0 x86_64 HVM kernel-6.1
  instance_type          = "t2.micro"
  subnet_id              = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]

  user_data = <<-EOF
              #!/bin/bash
              echo "ssh-rsa YOUR_PUBLIC_KEY" >> /home/ec2-user/.ssh/authorized_keys
              chmod 600 /home/ec2-user/.ssh/authorized_keys
              chown ec2-user:ec2-user /home/ec2-user/.ssh/authorized_keys
              EOF

  tags = {
    Name        = "MLOps-Instance"
    Project     = "MLOps-Pipeline"
    Environment = "Development"
  }
}
