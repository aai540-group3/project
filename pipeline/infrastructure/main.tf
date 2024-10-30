provider "aws" {
  region = var.aws_region
}

# S3 Bucket
resource "aws_s3_bucket" "ml_bucket" {
  bucket = var.bucket_name
  acl    = "private"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  lifecycle_rule {
    enabled = true
    prefix  = "logs/"

    expiration {
      days = 90
    }
  }
}

# DynamoDB Table
resource "aws_dynamodb_table" "feature_store" {
  name           = var.table_name
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "id"
  stream_enabled = true

  attribute {
    name = "id"
    type = "S"
  }

  ttl {
    enabled        = true
    attribute_name = "expiration_time"
  }
}

# IAM Role
resource "aws_iam_role" "ml_pipeline_role" {
  name = "ml_pipeline_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ml_pipeline" {
  name              = "/ml/pipeline"
  retention_in_days = 30
}

# SNS Topic
resource "aws_sns_topic" "alerts" {
  name = "ml-pipeline-alerts"
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "S3 bucket name"
  type        = string
}

variable "table_name" {
  description = "DynamoDB table name"
  type        = string
}

# Outputs
output "bucket_name" {
  value = aws_s3_bucket.ml_bucket.id
}

output "table_name" {
  value = aws_dynamodb_table.feature_store.id
}

output "role_arn" {
  value = aws_iam_role.ml_pipeline_role.arn
}