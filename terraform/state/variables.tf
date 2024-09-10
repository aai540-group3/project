variable "tf_state_key" {
  description = "The path and filename for the state file within the bucket"
  type        = string
  sensitive   = true
}

variable "state_bucket_name" {
  description = "Name of the S3 bucket for storing Terraform state"
  type        = string
}

variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for Terraform state locking"
  type        = string
}