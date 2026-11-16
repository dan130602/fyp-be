variable "name_prefix" {
  description = "Prefix for the resources in this module"
  type        = string
  default     = "batch-processing"
}

# -- image (shared with the immediate trigger Lambda) --------------------------

variable "ecr_repository_url" {
  description = "URL of the ECR repository holding the Lambda image (from the ecr module)"
  type        = string
}

variable "ecr_repository_name" {
  description = "Name of the ECR repository, used to resolve image_tag to a digest"
  type        = string
}

variable "image_tag" {
  description = "Tag of the image in ECR to deploy"
  type        = string
  default     = "latest"
}

# -- wiring --------------------------------------------------------------------

variable "batch_bucket_arn" {
  type = string
}

variable "batch_bucket_name" {
  type = string
}

variable "output_bucket_arn" {
  description = "Transcribe output bucket ARN"
  type        = string
}

variable "output_bucket_name" {
  description = "Transcribe output bucket name"
  type        = string
}

variable "queue_arn" {
  description = "ARN of the batch SQS queue"
  type        = string
}

variable "queue_url" {
  description = "URL of the batch SQS queue"
  type        = string
}

variable "max_receive_count" {
  description = "The batch queue's redrive maxReceiveCount (must match the sqs module call)"
  type        = number
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  description = "Private subnets for the Lambdas (the RDS security group admits their CIDRs)"
  type        = list(string)
}

variable "db_host" {
  type = string
}

variable "db_port" {
  type    = number
  default = 5432
}

variable "db_name" {
  type = string
}

variable "db_secret_arn" {
  description = "Secrets Manager secret holding {\"username\", \"password\"} for the database"
  type        = string
}

variable "transcribe_language_code" {
  type    = string
  default = "en-US"
}

variable "transcribe_language_options" {
  type    = string
  default = ""
}

variable "transcribe_language_id_mode" {
  type    = string
  default = "multiple"
}

variable "transcribe_max_speaker_labels" {
  type = number
}

# -- schedule ------------------------------------------------------------------

variable "schedule_expression" {
  description = "EventBridge Scheduler cron expression - default is 20:00 Monday-Friday"
  type        = string
  default     = "cron(0 20 ? * MON-FRI *)"
}

variable "schedule_timezone" {
  description = "IANA timezone the schedule expression is evaluated in (set explicitly - Scheduler defaults to UTC)"
  type        = string
  default     = "Asia/Singapore"
}

variable "schedule_enabled" {
  type    = bool
  default = true
}

# -- rate limiting -------------------------------------------------------------
# Throughput = worker_max_concurrency x worker_batch_size messages in flight. Each message only
# STARTS a Transcribe job (the job itself runs asynchronously), so the real downstreams these
# protect are the DB (one connection per concurrent invocation) and Transcribe's StartTranscriptionJob
# API / concurrent-job quota. Defaults are deliberately small.

variable "worker_max_concurrency" {
  description = "Event source mapping MaximumConcurrency: max parallel worker invocations from the batch queue (2-1000)"
  type        = number
  default     = 5

  validation {
    condition     = var.worker_max_concurrency >= 2 && var.worker_max_concurrency <= 1000
    error_message = "worker_max_concurrency must be between 2 and 1000 (AWS limits for SQS event source mappings)."
  }
}

variable "worker_batch_size" {
  description = "Messages per worker invocation"
  type        = number
  default     = 5

  validation {
    condition     = var.worker_batch_size >= 1 && var.worker_batch_size <= 10
    error_message = "worker_batch_size must be between 1 and 10 (no batching window is configured)."
  }
}

variable "worker_reserved_concurrency" {
  description = "Reserved concurrency for the worker function - keep >= worker_max_concurrency"
  type        = number
  default     = 5

  validation {
    condition     = var.worker_reserved_concurrency >= var.worker_max_concurrency
    error_message = "worker_reserved_concurrency must be >= worker_max_concurrency, or the event source mapping is throttled."
  }
}

variable "worker_timeout" {
  description = "Worker Lambda timeout in seconds (the batch queue visibility timeout is derived from this)"
  type        = number
  default     = 60
}

variable "worker_memory_size" {
  type    = number
  default = 512
}

variable "enqueuer_timeout" {
  description = "Enqueuer Lambda timeout in seconds (max 900)"
  type        = number
  default     = 900
}
