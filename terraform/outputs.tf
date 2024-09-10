output "s3_bucket_name" {
  value       = data.aws_s3_bucket.terraform_state.id != null ? data.aws_s3_bucket.terraform_state.id : aws_s3_bucket.terraform_state[0].id
  description = "The name of the S3 bucket"
}

output "dynamodb_table_name" {
  value       = data.aws_dynamodb_table.terraform_locks.id != null ? data.aws_dynamodb_table.terraform_locks.id : aws_dynamodb_table.terraform_locks[0].id
  description = "The name of the DynamoDB table"
}