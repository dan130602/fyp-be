
# module "vpc" {
#   source   = "./modules/vpc"
#   name     = "${var.project}-vpc"
#   vpc_cidr = var.vpc_cidr
#
#   # only 1 AZ in ap-southeast-1 for now
#   public_subnets = {
#     public_a = {
#       cidr_block = var.public_subnet_cidr
#       az         = "ap-southeast-1a"
#     }
#     public_b = {
#       cidr_block = "10.0.16.0/20"
#       az         = "ap-southeast-1b"
#     }
#   }
#
#   private_subnets = {
#     private_a = {
#       cidr_block = var.private_subnet_cidr
#       az         = "ap-southeast-1a"
#     }
#     private_b = {
#       cidr_block = "10.0.144.0/20"
#       az         = "ap-southeast-1b"
#     }
#   }
#   create_internet_gateway    = true
#   create_s3_gateway_endpoint = true
#   create_nat_gateway         = true
#
# }

module "s3" {
  source = "./modules/s3"

  ingestion_audio_bucket_name  = var.ingestion_audio_bucket_name
  frontend_bucket_name         = var.frontend_bucket_name
  frontend_origins             = var.cognito_callback_urls

  expiration_in_days = var.expiration_in_days

}
# module "cognito" {
#   source = "./modules/cognito"
#
#   aws_region             = var.aws_region
#   cognito_callback_urls  = var.cognito_callback_urls
#   cognito_domain_prefix  = var.cognito_domain_prefix
#   cognito_logout_urls    = var.cognito_logout_urls
#   cognito_user_pool_name = var.cognito_user_pool_name
# }
