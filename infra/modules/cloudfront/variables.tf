variable "name" {
  description = "Name prefix for CloudFront-related resources (e.g. the OAC)"
  type        = string
  default     = "frontend"
}

variable "frontend_bucket_id" {
  description = "ID of the S3 bucket holding the built frontend"
  type        = string
}

variable "frontend_bucket_arn" {
  description = "ARN of the S3 bucket holding the built frontend"
  type        = string
}

variable "frontend_bucket_regional_domain_name" {
  description = "Regional domain name of the S3 bucket - used as the CloudFront origin"
  type        = string
}

variable "price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_100"
}
