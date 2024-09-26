# S3 and DynamoDB names for Terraform state
s3_bucket_name      = "terraform-state-bucket-eeb973f4"
dynamodb_table_name = "terraform-state-lock-eeb973f4"

# AWS region
aws_region = "us-east-1"

# Organization users
org_users = [
  "jagustin@sandiego.edu",
  "lvo@sandiego.edu",
  "zrobertson@sandiego.edu"
]

# IAM group name
iam_group_name = "OrganizationUsers"

# Administrator policy name
admin_policy_name = "AdministratorAccessPolicy"

# Budget configuration
budget_name         = "SharedFreeTierBudget"
budget_limit_amount = "1"
budget_limit_unit   = "USD"

# CloudWatch alarm configuration
cloudwatch_alarm_name = "FreeTierUsageAlarm"
cloudwatch_threshold  = "1.00"

# SNS topic names
free_tier_alerts_topic_name = "free-tier-alerts"

# S3 bucket for MLOps
mlops_bucket_name = "mlops-artifacts-aai540-group3"

# GitHub Actions IAM role
github_actions_role_name = "GitHub-Actions-PolicyValidator"
github_org               = "aai540-group3"
github_repo              = "project"
aws_account_id           = "864899865811"

# Access Analyzer policy name
access_analyzer_policy_name = "AccessAnalyzerPolicy"