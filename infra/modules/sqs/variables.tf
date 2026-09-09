variable "queue_name" {
  description = "Name of the SQS queue (its DLQ is named \"<queue_name>-dlq\")"
  type        = string
}

variable "visibility_timeout_seconds" {
  description = "How long a message is hidden after a consumer receives it, before it's redelivered if not deleted"
  type        = number
  default     = 60
}

variable "message_retention_seconds" {
  type    = number
  default = 345600 # 4 days
}

variable "max_receive_count" {
  description = "Times a message can be received before it's routed to the DLQ instead of looping forever"
  type        = number
  default     = 5
}
