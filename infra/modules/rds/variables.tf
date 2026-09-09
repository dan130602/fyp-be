variable "name" {
  description = "Name prefix for RDS resources"
  type        = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  description = "Subnet IDs for the DB subnet group (private subnets)"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to connect to Postgres on port 5432"
  type        = list(string)
}

variable "database_name" {
  type    = string
  default = "ingestion"
}

variable "master_username" {
  type    = string
  default = "postgres"
}

variable "engine_version" {
  description = "Aurora PostgreSQL engine version — check `aws rds describe-db-engine-versions --engine aurora-postgresql` if this version isn't available in your region"
  type        = string
  default     = "16.6"
}

variable "min_capacity" {
  description = "Minimum Aurora Serverless v2 capacity, in ACUs"
  type        = number
  default     = 0.5
}

variable "max_capacity" {
  description = "Maximum Aurora Serverless v2 capacity, in ACUs"
  type        = number
  default     = 2
}

variable "backup_retention_period" {
  type    = number
  default = 1
}

variable "deletion_protection" {
  type    = bool
  default = false
}

variable "skip_final_snapshot" {
  type    = bool
  default = true
}
