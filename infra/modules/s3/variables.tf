variable "frontend_bucket_name" {
  description = "S3 bucket for frontend"
  type        = string
}

variable "ingestion_audio_bucket_name" {
  description = "S3 bucket for incoming audio"
  type        = string
}

variable "ingestion_audio_output_bucket_name" {
  description = "S3 bucket for transcribed audio"
  type        = string
}

variable "frontend_origins" {
  description = "Frontend origins allowed to upload directly to S3 via browser (CORS)"
  type        = list(string)
  default     = ["http://localhost:8081"]
}

variable "presign_role_arn" {
  description = "(Optional) IAM role ARN that is allowed to PUT to uploads/* (used in bucket policy"
  type        = string
  default     = null
}

variable "expiration_in_days" {
  description = "Expiry ofS S3 files in days"
  type        = number
}
