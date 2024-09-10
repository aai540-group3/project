data "aws_s3_bucket" "terraform_state" {
  bucket = var.state_bucket_name
}

data "aws_dynamodb_table" "terraform_locks" {
  name = var.dynamodb_table_name
}

resource "aws_s3_bucket" "terraform_state" {
  count  = data.aws_s3_bucket.terraform_state.id == null ? 1 : 0
  bucket = var.state_bucket_name

  tags = {
    Name        = "Terraform State"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  count  = data.aws_s3_bucket.terraform_state.id == null ? 1 : 0
  bucket = aws_s3_bucket.terraform_state[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  count  = data.aws_s3_bucket.terraform_state.id == null ? 1 : 0
  bucket = aws_s3_bucket.terraform_state[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  count                   = data.aws_s3_bucket.terraform_state.id == null ? 1 : 0
  bucket                  = aws_s3_bucket.terraform_state[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "terraform_locks" {
  count        = data.aws_dynamodb_table.terraform_locks.id == null ? 1 : 0
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [name]
  }

  tags = {
    Name        = "Terraform State Locks"
    Environment = "Management"
    ManagedBy   = "Terraform"
  }
}