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

variable "ingestion_audio_output_bucket_name" {
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
# Transcribe Lambda
# -----------------------

variable "transcribe_lambda_function_name" {
  description = "Name of the Lambda that starts Transcribe jobs on new S3 uploads"
  type        = string
  default     = "audio-transcribe-trigger"
}

variable "transcribe_ecr_repository_name" {
  description = "ECR repository holding the Transcribe Lambda's Docker image"
  type        = string
  default     = "audio-transcribe-lambda"
}

variable "transcribe_lambda_image_tag" {
  description = "Image tag in ECR to deploy — push it before the first apply"
  type        = string
  default     = "latest"
}

variable "github_repository" {
  description = "owner/repo allowed to push to the transcribe ECR repo via GitHub Actions OIDC"
  type        = string
  default     = "dan130602/fyp-be"
}

# -----------------------
# Cognito
# -----------------------

variable "cognito_domain_prefix" {
  description = "Cognito Hosted UI domain prefix (must be unique within the region)"
  type        = string
  default     = "ap-southeast-1jyrmtiwyi"
}

variable "cognito_callback_urls" {
  description = "Allowed callback URLs for the Cognito app client"
  type        = list(string)
  default     = ["https://d84l1y8p4kdic.cloudfront.net"]
}

variable "cognito_logout_urls" {
  description = "Allowed sign-out URLs for the Cognito app client"
  type        = list(string)
  default     = ["https://d84l1y8p4kdic.cloudfront.net"]
}

variable "cognito_user_pool_name" {
  description = "Name of the Cognito User Pool"
  type        = string
  default     = "User pool - 4vhqku"
}
