
data "aws_caller_identity" "current" {}

locals {
  media_processing_name     = "media-processing-service"
  ai_reasoning_service_name = "ai-reasoning-service"
  ai_reasoning_service_port = 2024

  # Built from plain strings rather than module outputs, so media-processing-service's env
  # doesn't depend on the ai-reasoning-service module (which itself depends on media-processing-service's SG).
  ai_reasoning_service_base_url = "http://${local.ai_reasoning_service_name}.${var.service_discovery_namespace}:${local.ai_reasoning_service_port}"
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
  frontend_origins                   = split(",", var.cors_allowed_origins)

  expiration_in_days = var.expiration_in_days

}

module "cloudfront" {
  source = "./modules/cloudfront"
  providers = {
    aws           = aws
    aws.us_east_1 = aws.us_east_1
  }

  domain_name                          = var.frontend_domain_name
  frontend_bucket_id                   = module.s3.frontend_bucket_id
  frontend_bucket_arn                  = module.s3.frontend_bucket_arn
  frontend_bucket_regional_domain_name = module.s3.frontend_bucket_regional_domain_name
}

module "ecr" {
  source = "./modules/ecr"

  repository_name   = var.transcribe_ecr_repository_name
  github_repository = var.github_repository
}

module "lambda" {
  source = "./modules/lambda"

  lambda_function_name          = var.transcribe_lambda_function_name
  ecr_repository_url            = module.ecr.repository_url
  ecr_repository_name           = module.ecr.repository_name
  image_tag                     = var.transcribe_lambda_image_tag
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

# Separate from transcribe_status_queue above - that one carries the raw Transcribe/Call
# Analytics completion event for THIS service's own poller; this one carries the payload
# (transcript pointer + checklist) that a downstream service consumes once we're done with it.
# Only media-processing-service itself sends here (see TranscriptHandoffService), so unlike
# transcribe_status_queue there's no EventBridge-facing queue policy needed.
# Consumed by media-processing-service's ComplianceAnalysisService, which holds each message
# for the whole synchronous ai-reasoning-service /runs/wait call (one LLM call per checklist item) - the
# visibility timeout has to outlast that, or SQS redelivers it mid-run.
module "transcript_handoff_queue" {
  source = "./modules/sqs"

  queue_name                 = "transcript-handoff"
  visibility_timeout_seconds = 900
  max_receive_count          = 3
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

# Creates empty Secrets Manager placeholders for this service — the RDS
# master credentials and the app's seed/default user password. No values are
# generated by Terraform; you fill them in by hand via the console (see the
# module's outputs for exactly what JSON/string each one expects). The RDS
# module below reads the credentials secret back in via a data source.
module "media_processing_secrets" {
  source = "./modules/secrets"

  name_prefix = local.media_processing_name
}

module "media_processing_alb" {
  source = "./modules/alb"

  name              = local.media_processing_name
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = values(module.vpc.public_subnet_ids)
  container_port    = 8080
  health_check_path = "/api/health"
  domain_name       = var.api_domain_name
}

# Reads back the {"username", "password"} you entered by hand into the
# rds-credentials secret — Terraform never generates or sees this value
# except to pass it straight through to the cluster below.
data "aws_secretsmanager_secret_version" "media_processing_rds_credentials" {
  secret_id  = module.media_processing_secrets.rds_credentials_secret_arn
  depends_on = [module.media_processing_secrets]
}

locals {
  media_processing_rds_credentials = jsondecode(data.aws_secretsmanager_secret_version.media_processing_rds_credentials.secret_string)
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
  master_username = local.media_processing_rds_credentials.username
  master_password = local.media_processing_rds_credentials.password
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
    sid       = "PublishTranscriptHandoff"
    effect    = "Allow"
    actions   = ["sqs:SendMessage"]
    resources = [module.transcript_handoff_queue.queue_arn]
  }

  statement {
    sid    = "ConsumeTranscriptHandoff"
    effect = "Allow"
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:ChangeMessageVisibility",
    ]
    resources = [module.transcript_handoff_queue.queue_arn]
  }

  statement {
    sid    = "ManageCognitoUsers"
    effect = "Allow"
    actions = [
      "cognito-idp:AdminCreateUser",
      "cognito-idp:AdminSetUserPassword",
      "cognito-idp:AdminGetUser",
      "cognito-idp:AdminDeleteUser",
      "cognito-idp:AdminAddUserToGroup",
      "cognito-idp:AdminRemoveUserFromGroup",
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
    SPRING_DATASOURCE_URL        = "jdbc:postgresql://${module.media_processing_rds.cluster_endpoint}:${module.media_processing_rds.port}/${module.media_processing_rds.database_name}"
    AWS_REGION                   = var.aws_region
    S3_BUCKET                    = module.s3.ingestion_audio_bucket_name
    TRANSCRIBE_OUTPUT_BUCKET     = module.s3.ingestion_audio_output_bucket_name
    TRANSCRIBE_STATUS_QUEUE_URL  = module.transcribe_status_queue.queue_url
    TRANSCRIPT_HANDOFF_QUEUE_URL = module.transcript_handoff_queue.queue_url
    CHECKLIST_JSON               = var.checklist_json
    AWS_COGNITO_REGION           = var.aws_region
    AWS_COGNITO_USER_POOL_ID     = module.cognito.cognito_user_pool_id
    AWS_COGNITO_APP_CLIENT_ID    = "${module.cognito.cognito_user_pool_client_id},${var.frontend_spa_cognito_client_id}"
    AWS_COGNITO_ISSUER_URI       = module.cognito.cognito_issuer_uri
    CORS_ALLOWED_ORIGINS         = var.cors_allowed_origins
    BOOTSTRAP_ADMIN_EMAILS       = var.bootstrap_admin_emails
    AI_BASE_URL                  = local.ai_reasoning_service_base_url
  }

  secrets = {
    DB_USERNAME           = "${module.media_processing_secrets.rds_credentials_secret_arn}:username::"
    DB_PASSWORD           = "${module.media_processing_secrets.rds_credentials_secret_arn}:password::"
    DEFAULT_USER_PASSWORD = module.media_processing_secrets.default_user_password_secret_arn
  }

  secret_arns_for_execution_role = [
    module.media_processing_secrets.rds_credentials_secret_arn,
    module.media_processing_secrets.default_user_password_secret_arn,
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

# -----------------------
# ai-reasoning-service (LangGraph compliance pipeline, ECS Fargate)
# -----------------------

# Private DNS for service-to-service calls inside the VPC - ai-reasoning-service registers here as
# ai-reasoning-service.<namespace>, and Cloud Map keeps that A record pointed at whatever IP its current
# task has (Fargate tasks get a new one on every restart/deploy).
resource "aws_service_discovery_private_dns_namespace" "internal" {
  name = var.service_discovery_namespace
  vpc  = module.vpc.vpc_id
}

module "ai_reasoning_service_ecr" {
  source = "./modules/ecr"

  repository_name     = var.ai_reasoning_service_ecr_repository_name
  github_repository   = var.ai_reasoning_service_github_repository
  github_allowed_refs = var.ai_reasoning_service_github_allowed_refs
}

# Empty placeholder, same as modules/secrets - fill the raw API key in by hand via the console
# BEFORE the first apply that creates the service, or the task fails to start pulling it.
resource "aws_secretsmanager_secret" "ai_reasoning_service_langsmith_api_key" {
  name        = "${local.ai_reasoning_service_name}/langsmith-api-key"
  description = "LangSmith API key for ai-reasoning-service tracing (raw string)"
}

# Bedrock credentials come from this task role - no AWS_BEARER_TOKEN_BEDROCK needed in ECS.
# Foundation-model ARNs are region-wildcarded because the apac.* cross-region inference
# profiles route requests to other APAC regions.
data "aws_iam_policy_document" "ai_reasoning_service_task" {
  statement {
    sid    = "InvokeBedrockModels"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:${var.aws_region}:${data.aws_caller_identity.current.account_id}:inference-profile/*",
    ]
  }
}

module "ai_reasoning_service_ecs" {
  source = "./modules/ecs"

  name               = local.ai_reasoning_service_name
  aws_region         = var.aws_region
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = values(module.vpc.private_subnet_ids)

  container_image = "${module.ai_reasoning_service_ecr.repository_url}:${var.ai_reasoning_service_image_tag}"
  container_port  = local.ai_reasoning_service_port
  cpu             = var.ai_reasoning_service_cpu
  memory          = var.ai_reasoning_service_memory
  desired_count   = var.ai_reasoning_service_desired_count

  # No ALB - only media-processing-service may call it, over Cloud Map DNS.
  ingress_security_group_ids     = [module.media_processing_ecs.security_group_id]
  service_discovery_namespace_id = aws_service_discovery_private_dns_namespace.internal.id

  # python:3.12-slim has no wget/curl. /ok is the langgraph server's liveness endpoint.
  health_check_command      = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${local.ai_reasoning_service_port}/ok', timeout=4)\" || exit 1"]
  health_check_start_period = 180

  task_role_policy_json = data.aws_iam_policy_document.ai_reasoning_service_task.json

  environment = {
    AWS_REGION        = var.aws_region
    LANGSMITH_TRACING = tostring(var.ai_reasoning_service_langsmith_tracing)
    LANGSMITH_PROJECT = var.ai_reasoning_service_langsmith_project
  }

  secrets = {
    LANGSMITH_API_KEY = aws_secretsmanager_secret.ai_reasoning_service_langsmith_api_key.arn
  }

  secret_arns_for_execution_role = [
    aws_secretsmanager_secret.ai_reasoning_service_langsmith_api_key.arn,
  ]
}

# Lets ai-reasoning-service's CI role roll the ECS service after pushing a new image
# (see ai-reasoning-service's .github/workflows/backend-ecr-push.yml).
data "aws_iam_policy_document" "ai_reasoning_service_ecs_deploy" {
  statement {
    effect = "Allow"
    actions = [
      "ecs:UpdateService",
      "ecs:DescribeServices",
    ]
    resources = [
      "arn:aws:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:service/${module.ai_reasoning_service_ecs.cluster_name}/${module.ai_reasoning_service_ecs.service_name}",
    ]
  }
}

resource "aws_iam_role_policy" "ai_reasoning_service_ecs_deploy" {
  name   = "${var.ai_reasoning_service_ecr_repository_name}-ecs-deploy"
  role   = module.ai_reasoning_service_ecr.github_actions_role_name
  policy = data.aws_iam_policy_document.ai_reasoning_service_ecs_deploy.json
}
