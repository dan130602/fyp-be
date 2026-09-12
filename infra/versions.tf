terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.39"
    }
  }

  # For remote running with Terraform Cloud
  backend "remote" {
    organization = "fyp-2026-dan"

    workspaces {
      name = "fyp-be"
    }
  }

  # For local running
  # backend "local" {
  #   path = "terraform.tfstate"
  # }
}

provider "aws" {
  region = var.aws_region
}