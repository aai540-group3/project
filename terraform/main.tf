# ---------------------------------------------------------------------------------------------------------------------
# TERRAFORM CONFIGURATION
# ---------------------------------------------------------------------------------------------------------------------
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.66.0"
    }
  }

  required_version = "= 1.9.5"

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
locals {
  emails = [
    "jagustin@sandiego.edu",
    "lvo@sandiego.edu",
    "zrobertson@sandiego.edu"
  ]
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS ORGANIZATION
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_organizations_organization" "org" {
  feature_set = "ALL"
}

# ---------------------------------------------------------------------------------------------------------------------
# IAM USERS AND GROUPS
# ---------------------------------------------------------------------------------------------------------------------
resource "aws_iam_user" "users" {
  count = length(local.emails)
  name  = split("@", local.emails[count.index])[0]

  tags = {
    Email       = local.emails[count.index]
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_user_login_profile" "users" {
  count                   = length(local.emails)
  user                    = aws_iam_user.users[count.index].name
  password_reset_required = true

  lifecycle {
    ignore_changes = [password_reset_required]
  }
}

resource "aws_iam_group" "org_users" {
  name = "OrganizationUsers"
}

resource "aws_iam_group_membership" "org_users" {
  name  = "OrganizationUsersMembership"
  users = aws_iam_user.users[*].name
  group = aws_iam_group.org_users.name
}

resource "aws_iam_policy" "free_tier_policy" {
  name        = "FreeTierEligiblePolicy"
  description = "Policy allowing access to free-tier eligible services"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:*",
          "dynamodb:*",
          "lambda:*",
          "ec2:Describe*",
          "rds:Describe*",
          "cloudwatch:*",
          "sns:*",
          "sqs:*",
          "glacier:*"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:RunInstances",
          "ec2:StartInstances",
          "ec2:StopInstances",
          "ec2:TerminateInstances"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "ec2:InstanceType" = [
              "t2.micro",
              "t3.micro"
            ]
          }
        }
      }
    ]
  })
}

resource "aws_iam_group_policy_attachment" "free_tier_policy_attachment" {
  group      = aws_iam_group.org_users.name
  policy_arn = aws_iam_policy.free_tier_policy.arn
}

# ---------------------------------------------------------------------------------------------------------------------
# S3 BUCKET FOR TERRAFORM STATE
# ---------------------------------------------------------------------------------------------------------------------
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

resource "aws_s3_bucket_versioning" "enabled" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "default" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

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
resource "aws_budgets_budget" "shared_user_budget" {
  name         = "SharedFreeTierBudget"
  budget_type  = "COST"
  limit_amount = "1"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = local.emails
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = local.emails
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }
}