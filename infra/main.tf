
data "aws_caller_identity" "current" {}

locals {
  media_processing_name = "media-processing-service"
}

module "vpc" {
  source   = "./modules/vpc"
  name     = "${var.project}-vpc"
  vpc_cidr = var.vpc_cidr

  public_subnets  = var.public_subnet_cidrs
  private_subnets = var.private_subnet_cidrs

  create_internet_gateway    = true
  create_s3_gateway_endpoint = true
  create_nat_gateway         = true
}

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
  transcribe_max_speaker_labels = var.transcribe_max_speaker_labels

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

# -----------------------
# Media Processing Service (ECS Fargate)
# -----------------------

module "media_processing_ecr" {
  source = "./modules/ecr"

  repository_name   = var.media_processing_ecr_repository_name
  github_repository = var.github_repository
}

# Not read directly by Spring (Secrets Manager values aren't native datasource
# config) — passed into the task as the DEFAULT_USER_PASSWORD secret instead.
resource "aws_secretsmanager_secret" "default_user_password" {
  name = "${local.media_processing_name}/default-user-password"
}

resource "aws_secretsmanager_secret_version" "default_user_password" {
  secret_id     = aws_secretsmanager_secret.default_user_password.id
  secret_string = var.default_user_password
}

module "media_processing_alb" {
  source = "./modules/alb"

  name              = local.media_processing_name
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = values(module.vpc.public_subnet_ids)
  container_port    = 8080
  health_check_path = "/api/health"
}

module "media_processing_rds" {
  source = "./modules/rds"

  name       = local.media_processing_name
  vpc_id     = module.vpc.vpc_id
  subnet_ids = values(module.vpc.private_subnet_ids)

  # Tasks live in the private subnets, so scoping by CIDR (rather than the ECS
  # tasks SG) avoids an rds <-> ecs module dependency cycle.
  allowed_cidr_blocks = [for s in var.private_subnet_cidrs : s.cidr_block]

  database_name   = var.db_name
  master_username = var.db_master_username
  min_capacity    = var.rds_min_capacity
  max_capacity    = var.rds_max_capacity
}

data "aws_iam_policy_document" "media_processing_task" {
  statement {
    sid    = "MediaAssetStorage"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${module.s3.ingestion_audio_bucket_arn}/*"]
  }

  statement {
    sid       = "PlaybackReadAccess"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${module.s3.ingestion_audio_output_bucket_arn}/*"]
  }

  statement {
    sid    = "PollTranscribeStatus"
    effect = "Allow"
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = [module.transcribe_status_queue.queue_arn]
  }

  statement {
    sid    = "ManageCognitoUsers"
    effect = "Allow"
    actions = [
      "cognito-idp:AdminCreateUser",
      "cognito-idp:AdminSetUserPassword",
      "cognito-idp:AdminGetUser",
      "cognito-idp:AdminDeleteUser",
    ]
    resources = [module.cognito.cognito_user_pool_arn]
  }
}

module "media_processing_ecs" {
  source = "./modules/ecs"

  name               = local.media_processing_name
  aws_region         = var.aws_region
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = values(module.vpc.private_subnet_ids)

  container_image = "${module.media_processing_ecr.repository_url}:${var.media_processing_image_tag}"
  container_port  = 8080
  cpu             = var.media_processing_cpu
  memory          = var.media_processing_memory
  desired_count   = var.media_processing_desired_count

  alb_security_group_id = module.media_processing_alb.alb_security_group_id
  target_group_arn      = module.media_processing_alb.target_group_arn

  task_role_policy_json = data.aws_iam_policy_document.media_processing_task.json

  environment = {
    SPRING_DATASOURCE_URL       = "jdbc:postgresql://${module.media_processing_rds.cluster_endpoint}:${module.media_processing_rds.port}/${module.media_processing_rds.database_name}"
    AWS_REGION                  = var.aws_region
    S3_BUCKET                   = module.s3.ingestion_audio_bucket_name
    TRANSCRIBE_OUTPUT_BUCKET    = module.s3.ingestion_audio_output_bucket_name
    TRANSCRIBE_STATUS_QUEUE_URL = module.transcribe_status_queue.queue_url
    AWS_COGNITO_REGION          = var.aws_region
    AWS_COGNITO_USER_POOL_ID    = module.cognito.cognito_user_pool_id
    AWS_COGNITO_APP_CLIENT_ID   = module.cognito.cognito_user_pool_client_id
    AWS_COGNITO_ISSUER_URI      = module.cognito.cognito_issuer_uri
    CORS_ALLOWED_ORIGINS        = var.cors_allowed_origins
    BOOTSTRAP_ADMIN_EMAILS      = var.bootstrap_admin_emails
  }

  secrets = {
    DB_USERNAME           = "${module.media_processing_rds.master_user_secret_arn}:username::"
    DB_PASSWORD           = "${module.media_processing_rds.master_user_secret_arn}:password::"
    DEFAULT_USER_PASSWORD = aws_secretsmanager_secret.default_user_password.arn
  }

  secret_arns_for_execution_role = [
    module.media_processing_rds.master_user_secret_arn,
    aws_secretsmanager_secret.default_user_password.arn,
  ]

  depends_on = [module.media_processing_alb]
}

# Lets the same OIDC role CI uses to push images also roll the ECS service
# after a successful push (see .github/workflows/media-processing-deploy.yml).
data "aws_iam_policy_document" "media_processing_ecs_deploy" {
  statement {
    effect = "Allow"
    actions = [
      "ecs:UpdateService",
      "ecs:DescribeServices",
    ]
    resources = [
      "arn:aws:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:service/${module.media_processing_ecs.cluster_name}/${module.media_processing_ecs.service_name}",
    ]
  }
}

resource "aws_iam_role_policy" "media_processing_ecs_deploy" {
  name   = "${var.media_processing_ecr_repository_name}-ecs-deploy"
  role   = module.media_processing_ecr.github_actions_role_name
  policy = data.aws_iam_policy_document.media_processing_ecs_deploy.json
}
