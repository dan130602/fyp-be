
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

  ingestion_audio_bucket_name        = var.ingestion_audio_bucket_name
  ingestion_audio_output_bucket_name = var.ingestion_audio_output_bucket_name
  frontend_bucket_name               = var.frontend_bucket_name
  frontend_origins                   = ["http://localhost:8081"]

  expiration_in_days = var.expiration_in_days

}

module "ecr" {
  source = "./modules/ecr"

  repository_name   = var.transcribe_ecr_repository_name
  github_repository = var.github_repository
}

module "lambda" {
  source = "./modules/lambda"

  lambda_function_name = var.transcribe_lambda_function_name
  ecr_repository_url   = module.ecr.repository_url
  ecr_repository_name  = module.ecr.repository_name
  image_tag            = var.transcribe_lambda_image_tag

  input_bucket_id  = module.s3.ingestion_audio_bucket_id
  input_bucket_arn = module.s3.ingestion_audio_bucket_arn

  output_bucket_id   = module.s3.ingestion_audio_output_bucket_id
  output_bucket_arn  = module.s3.ingestion_audio_output_bucket_arn
  output_bucket_name = module.s3.ingestion_audio_output_bucket_name
}
module "transcribe_status_queue" {
  source = "./modules/sqs"

  queue_name = "transcribe-status-updates"
}

module "transcribe_status_events" {
  source = "./modules/eventbridge"

  rule_name = "transcribe-job-state-change"
  event_pattern = jsonencode({
    source      = ["aws.transcribe"]
    detail-type = ["Transcribe Job State Change"]
    detail = {
      TranscriptionJobStatus = ["COMPLETED", "FAILED"]
    }
  })
  target_arn = module.transcribe_status_queue.queue_arn
}

# Glue between the two modules above - needs ARNs from both, so it can't live inside
# either without creating a module dependency cycle (events needs the queue's ARN as
# its target; this policy needs the rule's ARN to scope who may send to the queue).
data "aws_iam_policy_document" "transcribe_status_queue_policy" {
  statement {
    sid    = "AllowEventBridgeSend"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    actions   = ["sqs:SendMessage"]
    resources = [module.transcribe_status_queue.queue_arn]

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [module.transcribe_status_events.rule_arn]
    }
  }
}

resource "aws_sqs_queue_policy" "transcribe_status" {
  queue_url = module.transcribe_status_queue.queue_url
  policy    = data.aws_iam_policy_document.transcribe_status_queue_policy.json
}

module "cognito" {
  source = "./modules/cognito"

  aws_region             = var.aws_region
  cognito_callback_urls  = var.cognito_callback_urls
  cognito_domain_prefix  = var.cognito_domain_prefix
  cognito_logout_urls    = var.cognito_logout_urls
  cognito_user_pool_name = var.cognito_user_pool_name
}
