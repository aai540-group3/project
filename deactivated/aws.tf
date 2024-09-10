# ---------------------------------------------------------------------------------------------------------------------
# LOCALS
# ---------------------------------------------------------------------------------------------------------------------

locals {
  emails                   = split(",", var.email_list)
  total_budget_amount      = "1.00"
  individual_budget_amount = format("%.2f", tonumber(local.total_budget_amount) / (length(local.emails) + 1))
}

# ---------------------------------------------------------------------------------------------------------------------
# DATA SOURCES
# ---------------------------------------------------------------------------------------------------------------------

data "aws_organizations_organization" "org" {}

# ---------------------------------------------------------------------------------------------------------------------
# AWS ORGANIZATION ACCOUNTS
# ---------------------------------------------------------------------------------------------------------------------

resource "aws_organizations_account" "user_accounts" {
  for_each = toset(local.emails)

  email     = each.key
  name      = "User Account for ${each.key}"
  role_name = "OrganizationAccountAccessRole"

  lifecycle {
    ignore_changes  = [name, email, role_name]
    prevent_destroy = true
  }

  tags = {
    ManagedBy = "Terraform"
    Email     = each.key
  }
}

# ---------------------------------------------------------------------------------------------------------------------
# AWS BUDGETS
# ---------------------------------------------------------------------------------------------------------------------

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

resource "aws_budgets_budget" "individual" {
  for_each     = aws_organizations_account.user_accounts
  name         = "IndividualBudget-${each.value.id}"
  account_id   = each.value.id
  budget_type  = "COST"
  limit_amount = local.individual_budget_amount
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [each.key]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [each.key]
  }
}
