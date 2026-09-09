variable "rule_name" {
  type = string
}

variable "event_pattern" {
  description = "JSON event pattern (use jsonencode(...) at the call site)"
  type        = string
}

variable "target_arn" {
  description = "ARN of the resource this rule delivers matching events to"
  type        = string
}
