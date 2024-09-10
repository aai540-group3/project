variable "aws_region" {
  description = "The AWS region to create resources in"
  type        = string
  sensitive   = true
}

variable "tf_state_key" {
  description = "The path and filename for the state file within the bucket"
  type        = string
  sensitive   = true
}

variable "state_bucket_name" {
  description = "Name of the S3 bucket for storing Terraform state"
  type        = string
  sensitive   = true
}

variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for Terraform state locking"
  type        = string
  sensitive   = true
}

variable "email_list" {
  description = "Comma-separated list of email addresses for budget notifications"
  type        = string
  sensitive   = true
}
