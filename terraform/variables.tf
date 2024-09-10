variable "aws_region" {
  description = "The AWS region to create resources in"
  type        = string
}

variable "tf_state_key" {
  description = "The path and filename for the state file within the bucket"
  type        = string
}

variable "state_bucket_name" {
  description = "Name of the S3 bucket for storing Terraform state"
  type        = string
}

variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for Terraform state locking"
  type        = string
}

variable "email_list" {
  type        = string
  description = "Comma-separated list of email addresses"
}
