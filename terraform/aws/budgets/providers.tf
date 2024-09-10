terraform {
  required_version = ">= 1.5.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.66.0"
    }
  }

  backend "s3" {
    bucket         = var.state_bucket_name
    key            = var.tf_state_key
    region         = var.aws_region
    dynamodb_table = var.dynamodb_table_name
  }
}

provider "aws" {
  region = "us-west-2"
}