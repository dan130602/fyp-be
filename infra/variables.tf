variable "aws_region" {
  description = "AWS region to deploy resources into"
  type        = string
  default     = "ap-southeast-1"
}

variable "project" {
  description = "Short name used for naming resources"
  type        = string
  default     = "oitnb"
}

variable "env" {
  description = "Environment name (e.g., dev, stg, prod)"
  type        = string
  default     = "dev"
}

# -----------------------
# S3
# -----------------------

variable "ingestion_audio_bucket_name" {
  type = string
}

variable "frontend_bucket_name" {
  type = string
}

variable "frontend_origins" {
  type = list(string)
}

variable "presign_role_arn" {
  type    = string
  default = null
}

variable "expiration_in_days" {
  type = number
}

# -----------------------
# Cognito
# -----------------------

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
