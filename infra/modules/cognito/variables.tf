variable "cognito_domain_prefix" {
  description = "Cognito Hosted UI domain prefix (must be unique within the region)"
  type        = string
}

variable "cognito_callback_urls" {
  description = "Allowed callback URLs for the Cognito app client"
  type        = list(string)
}

variable "cognito_logout_urls" {
  description = "Allowed sign-out URLs for the Cognito app client"
  type        = list(string)
}

variable "cognito_user_pool_name" {
    description = "Name of the Cognito User Pool"
    type        = string
}

variable "aws_region" {
  description = "AWS region to deploy resources into"
  type        = string
  default     = "ap-southeast-1"
}
