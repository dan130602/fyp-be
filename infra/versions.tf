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

# CloudFront only accepts ACM certs from us-east-1, regardless of the region
# above - used solely by modules/cloudfront for the cert.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}