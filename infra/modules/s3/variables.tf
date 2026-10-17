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

variable "ingestion_audio_batch_bucket_name" {
  description = "S3 bucket for audio queued for the nightly batch run (no S3 event trigger)"
  type        = string
  default = "ingestion-audio-batch"
}

variable "batch_expiration_in_days" {
  description = "Safety-net expiry for objects left in the batch bucket (normally deleted after processing)"
  type        = number
  default     = 7
}
