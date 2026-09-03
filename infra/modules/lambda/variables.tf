variable "lambda_function_name" {
  description = "Name of the Lambda function that starts Transcribe jobs"
  type        = string
}

variable "ecr_repository_url" {
  description = "URL of the ECR repository holding the Lambda's Docker image (from the ecr module)"
  type        = string
}

variable "image_tag" {
  description = "Tag of the image in ECR to deploy (push it before the first apply)"
  type        = string
  default     = "latest"
}

variable "timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 120
}

variable "memory_size" {
  description = "Lambda memory in MB"
  type        = number
  default     = 512
}

variable "transcribe_language_code" {
  description = "Language code passed to StartTranscriptionJob"
  type        = string
  default     = "en-US"
}

variable "input_bucket_id" {
  description = "ID of the S3 bucket that receives uploaded audio and triggers this Lambda"
  type        = string
}

variable "input_bucket_arn" {
  description = "ARN of the input audio bucket"
  type        = string
}

variable "input_bucket_prefix" {
  description = "Key prefix to filter S3 event notifications on"
  type        = string
  default     = "uploads/"
}

variable "output_bucket_id" {
  description = "ID of the S3 bucket Transcribe writes results to"
  type        = string
}

variable "output_bucket_arn" {
  description = "ARN of the output bucket"
  type        = string
}

variable "output_bucket_name" {
  description = "Name of the output bucket (passed to the Lambda as an env var)"
  type        = string
}
