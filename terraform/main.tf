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
# AWS BUDGETS
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_budgets_budget" "organization_wide" {
  name         = "OrganizationWideBudget"
  budget_type  = "COST"
  limit_amount = "1.00"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

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

resource "aws_budgets_budget" "individual" {
  count        = length(local.emails)
  name         = "IndividualBudget-${local.emails[count.index]}"
  budget_type  = "COST"
  limit_amount = "1.00"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

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
}