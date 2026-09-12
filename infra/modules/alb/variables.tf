variable "name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "container_port" {
  type    = number
  default = 8080
}

variable "health_check_path" {
  type    = string
  default = "/api/health"
}

variable "domain_name" {
  description = "Domain the ALB is served on over HTTPS (e.g. api.example.com) - used as the ACM cert CN"
  type        = string
}
