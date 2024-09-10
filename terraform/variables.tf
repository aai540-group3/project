variable "aws_region" {
  description = "The AWS region to create resources in"
  type        = string
}

variable "state_bucket_name" {
  description = "The name of the S3 bucket for Terraform state"
  type        = string
}

variable "dynamodb_table_name" {
  description = "The name of the DynamoDB table for Terraform state locking"
  type        = string
}

variable "tf_state_key" {
  description = "The path and filename for the state file within the bucket"
  type        = string
}