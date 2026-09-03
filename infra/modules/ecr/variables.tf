variable "repository_name" {
  description = "Name of the ECR repository"
  type        = string
}

variable "image_tag_mutability" {
  description = "Whether image tags can be overwritten (MUTABLE or IMMUTABLE)"
  type        = string
  default     = "MUTABLE"
}

variable "github_repository" {
  description = "owner/repo allowed to assume the push role via OIDC, e.g. dan130602/fyp-be"
  type        = string
}

variable "github_allowed_refs" {
  description = "Git refs allowed to assume the push role, e.g. refs/heads/main"
  type        = list(string)
  default     = ["refs/heads/main"]
}
