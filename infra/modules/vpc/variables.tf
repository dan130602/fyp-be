variable "name" {
  description = "Name prefix for VPC resources"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "enable_dns_support" {
  description = "Enable Amazon-provided DNS resolver"
  type        = bool
  default     = true
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC"
  type        = bool
  default     = true
}

variable "public_subnets" {
  description = "Map of public subnet definitions"
  type = map(object({
    cidr_block = string
    az         = string
  }))
  default = {}
}

variable "private_subnets" {
  description = "Map of private subnet definitions"
  type = map(object({
    cidr_block = string
    az         = string
  }))
  default = {}
}

variable "create_internet_gateway" {
  description = "Whether to create an internet gateway"
  type        = bool
  default     = true
}

variable "create_s3_gateway_endpoint" {
  description = "Whether to create an S3 gateway endpoint"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Common tags applied to resources"
  type        = map(string)
  default     = {}
}

variable "create_nat_gateway" {
  type = bool
  default = false
}