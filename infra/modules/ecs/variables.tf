variable "name" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "container_image" {
  description = "Full image ref, e.g. <ecr_repo_url>:latest"
  type        = string
}

variable "container_port" {
  type    = number
  default = 8080
}

variable "cpu" {
  type    = number
  default = 512
}

variable "memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "environment" {
  description = "Plain (non-secret) container env vars"
  type        = map(string)
  default     = {}
}

variable "secrets" {
  description = "Container env vars sourced from Secrets Manager: env var name => valueFrom (a secret ARN, optionally suffixed with :jsonKey::)"
  type        = map(string)
  default     = {}
}

variable "secret_arns_for_execution_role" {
  description = "Distinct Secrets Manager secret ARNs the execution role needs secretsmanager:GetSecretValue on (base ARNs, without :jsonKey:: suffixes)"
  type        = list(string)
  default     = []
}

variable "alb_security_group_id" {
  description = "Security group of the ALB allowed to reach the task on container_port. Leave null to run without a load balancer."
  type        = string
  default     = null
}

variable "target_group_arn" {
  description = "ALB target group to register tasks into. Leave null to run without a load balancer."
  type        = string
  default     = null
}

variable "task_role_policy_json" {
  description = "IAM policy JSON granting the running container's task role its AWS permissions"
  type        = string
}

variable "log_retention_days" {
  type    = number
  default = 14
}
