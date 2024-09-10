# ---------------------------------------------------------------------------------------------------------------------
# LOCALS
# ---------------------------------------------------------------------------------------------------------------------

locals {
  # Split the comma-separated email list into an array
  emails = split(",", var.email_list)
  
  # Set the total budget amount for the organization
  total_budget_amount = "1.00"
  
  # Calculate individual budget amount by dividing total budget by number of emails plus one (for org-wide budget)
  individual_budget_amount = format("%.2f", tonumber(local.total_budget_amount) / (length(local.emails) + 1))
}

# ---------------------------------------------------------------------------------------------------------------------
# S3 BUCKET FOR TERRAFORM STATE
# ---------------------------------------------------------------------------------------------------------------------

# Create an S3 bucket to store Terraform state
resource "aws_s3_bucket" "terraform_state" {
  bucket = var.state_bucket_name
  
  tags = {
    Name        = "Terraform State"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
  
  # Prevent accidental deletion and ignore changes to the bucket name
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [bucket]
  }
}

# Enable versioning for the S3 bucket
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
  name         = var.dynamodb_table_name
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
  
  # Prevent accidental deletion and ignore changes to the table name
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS ORGANIZATION ACCOUNTS
# ---------------------------------------------------------------------------------------------------------------------

# Create AWS accounts for each email in the list
resource "aws_organizations_account" "user_accounts" {
  count                      = length(local.emails)
  email                      = local.emails[count.index]
  name                       = "User Account ${count.index + 1}"
  role_name                  = "OrganizationAccountAccessRole"
  close_on_deletion          = false
  iam_user_access_to_billing = "ALLOW"

  # Prevent Terraform from removing the account from your organization
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name, email]
  }

  # Add tags for better organization
  tags = {
    ManagedBy = "Terraform"
    Email     = local.emails[count.index]
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS BUDGETS
# ---------------------------------------------------------------------------------------------------------------------

# Create an organization-wide AWS budget
resource "aws_budgets_budget" "organization_wide" {
  name         = "OrganizationWideBudget"
  budget_type  = "COST"
  limit_amount = local.total_budget_amount
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
}

# Individual AWS budgets for each email address
resource "aws_budgets_budget" "individual" {
  count        = length(local.emails)
  name         = "IndividualBudget-${count.index}"
  budget_type  = "COST"
  limit_amount = local.individual_budget_amount
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
}
