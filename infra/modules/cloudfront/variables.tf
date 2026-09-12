variable "domain_name" {
  description = "Custom domain the frontend is served on (e.g. app.example.com) - also used as the CloudFront alias and ACM cert CN"
  type        = string
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
