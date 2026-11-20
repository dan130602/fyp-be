# variable "aws_region" {
#   description = "AWS region to deploy resources into"
#   type        = string
#   default     = "ap-southeast-1"
# }
#
# variable "project" {
#   description = "Short name used for naming resources"
#   type        = string
#   default     = "oitnb"
# }
#
# variable "env" {
#   description = "Environment name (e.g., dev, stg, prod)"
#   type        = string
#   default     = "dev"
# }
#
# # -----------------------
# # S3
# # -----------------------
#
# variable "ingestion_audio_bucket_name" {
#   type = string
# }
#
# variable "ingestion_audio_output_bucket_name" {
#   type = string
# }
#
# variable "ingestion_audio_batch_bucket_name" {
#   description = "S3 bucket for audio queued for the nightly batch run - globally unique, no S3 event trigger"
#   type        = string
#   default     = "ingestion-audio-batch"
# }
#
# variable "batch_expiration_in_days" {
#   description = "Safety-net expiry for objects left in the batch bucket (normally deleted after processing)"
#   type        = number
#   default     = 7
# }
#
# variable "frontend_bucket_name" {
#   type = string
# }
#
# variable "frontend_origins" {
#   type = list(string)
# }
#
# variable "frontend_spa_cognito_client_id" {
#   description = "App client ID for the browser SPA frontend - a public client (no secret) created manually in the Cognito console, not managed by Terraform, since aws_cognito_user_pool_client.web in modules/cognito generates a confidential client unsuitable for a browser. Combined with that managed client's ID when configuring the backend's accepted client_id list."
#   type        = string
#   default     = "6jde2h1pb3cqjuvcn6gbqvqbdu"
# }
#
# variable "api_domain_name" {
#   description = "Domain the media-processing-service ALB is served on over HTTPS. Route 53 records for it and for ACM validation are created manually, not by Terraform."
#   type        = string
#   default     = "api.clearcall-fyp-smu.com"
# }
#
# variable "frontend_domain_name" {
#   description = "Custom domain the frontend is served on via CloudFront. Route 53 records for it and for ACM validation are created manually, not by Terraform."
#   type        = string
#   default     = "clearcall-fyp-smu.com"
# }
#
# variable "presign_role_arn" {
#   type    = string
#   default = null
# }
#
# variable "expiration_in_days" {
#   type = number
# }
#
# # -----------------------
# # Transcribe Lambda
# # -----------------------
#
# variable "transcribe_lambda_function_name" {
#   description = "Name of the Lambda that starts Transcribe jobs on new S3 uploads"
#   type        = string
#   default     = "audio-transcribe-trigger"
# }
#
# variable "transcribe_ecr_repository_name" {
#   description = "ECR repository holding the Transcribe Lambda's Docker image"
#   type        = string
#   default     = "audio-transcribe-lambda"
# }
#
# variable "transcribe_lambda_image_tag" {
#   description = "Image tag in ECR to deploy — push it before the first apply"
#   type        = string
#   default     = "latest"
# }
#
# variable "github_repository" {
#   description = "owner/repo allowed to push to the transcribe ECR repo via GitHub Actions OIDC"
#   type        = string
#   default     = "dan130602/fyp-be"
# }
# variable "transcribe_language_options" {
#   description = "Comma-separated language codes for Transcribe language identification; empty = fixed language"
#   type        = string
#   default     = "en-US,zh-CN"
# }
#
# variable "transcribe_language_id_mode" {
#   description = "Transcribe language identification mode: multiple or single"
#   type        = string
#   default     = "multiple"
# }
#
# variable "transcribe_max_speaker_labels" {
#   description = "Maximum number of speaker labels to detect in a transcription job"
#   type        = number
#   default     = 2
# }
#
# # -----------------------
# # Networking
# # -----------------------
#
# variable "vpc_cidr" {
#   type    = string
#   default = "10.0.0.0/16"
# }
#
# variable "public_subnet_cidrs" {
#   type = map(object({
#     cidr_block = string
#     az         = string
#   }))
#   default = {
#     public_a = { cidr_block = "10.0.0.0/20", az = "ap-southeast-1a" }
#     public_b = { cidr_block = "10.0.16.0/20", az = "ap-southeast-1b" }
#   }
# }
#
# variable "private_subnet_cidrs" {
#   type = map(object({
#     cidr_block = string
#     az         = string
#   }))
#   default = {
#     private_a = { cidr_block = "10.0.128.0/20", az = "ap-southeast-1a" }
#     private_b = { cidr_block = "10.0.144.0/20", az = "ap-southeast-1b" }
#   }
# }
#
# # -----------------------
# # Media Processing Service (ECS Fargate)
# # -----------------------
#
# variable "media_processing_ecr_repository_name" {
#   type    = string
#   default = "media-processing-service"
# }
#
# variable "media_processing_image_tag" {
#   description = "Image tag the ECS task definition points at — CI pushes both this tag and the git SHA on every build"
#   type        = string
#   default     = "latest"
# }
#
# variable "media_processing_cpu" {
#   description = "Fargate task vCPU units (1024 = 1 vCPU)"
#   type        = number
#   default     = 512
# }
#
# variable "media_processing_memory" {
#   description = "Fargate task memory, in MiB"
#   type        = number
#   default     = 1024
# }
#
# variable "media_processing_desired_count" {
#   type    = number
#   default = 1
# }
#
# variable "cors_allowed_origins" {
#   type    = string
#   default = "http://localhost:5173,https://clearcall-fyp-smu.com"
# }
#
# variable "bootstrap_admin_emails" {
#   type    = string
#   default = ""
# }
#
# variable "checklist_json" {
#   description = "Fixed compliance checklist forwarded to the next service with every completed transcript - override via tfvars, not per-upload"
#   type        = string
#   default     = "{}"
# }
#
# # -----------------------
# # RDS (Aurora PostgreSQL Serverless v2)
# # -----------------------
#
# variable "db_name" {
#   type    = string
#   default = "ingestion"
# }
#
# variable "db_master_username" {
#   type    = string
#   default = "postgres"
# }
#
# variable "rds_min_capacity" {
#   description = "Minimum Aurora Serverless v2 capacity, in ACUs"
#   type        = number
#   default     = 0.5
# }
#
# variable "rds_max_capacity" {
#   description = "Maximum Aurora Serverless v2 capacity, in ACUs"
#   type        = number
#   default     = 2
# }
#
#
# # -----------------------
# # Cognito
# # -----------------------
#
# variable "cognito_domain_prefix" {
#   description = "Cognito Hosted UI domain prefix (must be unique within the region)"
#   type        = string
#   default     = "ap-southeast-1jyrmtiwyi"
# }
#
# variable "cognito_callback_urls" {
#   description = "Allowed callback URLs for the Cognito app client"
#   type        = list(string)
#   default     = ["https://d84l1y8p4kdic.cloudfront.net", "https://clearcall-fyp-smu.com"]
# }
#
# variable "cognito_logout_urls" {
#   description = "Allowed sign-out URLs for the Cognito app client"
#   type        = list(string)
#   default     = ["https://d84l1y8p4kdic.cloudfront.net", "https://clearcall-fyp-smu.com"]
# }
#
# variable "cognito_user_pool_name" {
#   description = "Name of the Cognito User Pool"
#   type        = string
#   default     = "User pool - 4vhqku"
# }
#
# # -----------------------
# # ai-reasoning-service (ECS Fargate)
# # -----------------------
#
# variable "service_discovery_namespace" {
#   description = "Cloud Map private DNS namespace for service-to-service calls; ai-reasoning-service is reachable at ai-reasoning-service.<this>"
#   type        = string
#   default     = "clearcall.local"
# }
#
# variable "ai_reasoning_service_ecr_repository_name" {
#   type    = string
#   default = "ai-reasoning-service"
# }
#
# variable "ai_reasoning_service_github_repository" {
#   description = "owner/repo allowed to push to the ai-reasoning-service ECR repo via GitHub Actions OIDC"
#   type        = string
#   default     = "gnlyrad/fyp-ai"
# }
#
# variable "ai_reasoning_service_github_allowed_refs" {
#   description = "Branches of ai_reasoning_service_github_repository allowed to assume the push/deploy role"
#   type        = list(string)
#   default     = ["refs/heads/main", "refs/heads/langgraph-backend"]
# }
#
# variable "ai_reasoning_service_image_tag" {
#   type    = string
#   default = "latest"
# }
#
# variable "ai_reasoning_service_cpu" {
#   description = "Fargate task vCPU units (1024 = 1 vCPU)"
#   type        = number
#   default     = 1024
# }
#
# variable "ai_reasoning_service_memory" {
#   description = "Fargate task memory, in MiB - holds the FAISS index plus the langgraph server"
#   type        = number
#   default     = 3072
# }
#
# variable "ai_reasoning_service_desired_count" {
#   description = "Keep at 1 - langgraph dev keeps run state in memory"
#   type        = number
#   default     = 1
# }
#
# variable "ai_reasoning_service_langsmith_tracing" {
#   type    = bool
#   default = true
# }
#
# variable "ai_reasoning_service_langsmith_project" {
#   type    = string
#   default = "ai-reasoning-service"
# }
#
# # -----------------------
# # Batch ("process later") processing
# # -----------------------
#
# variable "batch_schedule_expression" {
#   description = "EventBridge Scheduler cron for the nightly enqueue - default 20:00 Monday-Friday"
#   type        = string
#   default     = "cron(0 20 ? * MON-FRI *)"
# }
#
# variable "batch_schedule_timezone" {
#   description = "IANA timezone the batch schedule is evaluated in"
#   type        = string
#   default     = "Asia/Singapore"
# }
#
# variable "batch_schedule_enabled" {
#   description = "Set false to pause the nightly run without destroying it"
#   type        = bool
#   default     = true
# }
#
# variable "batch_worker_max_concurrency" {
#   description = "Max parallel batch worker invocations (SQS event source mapping MaximumConcurrency, 2-1000). Each holds one DB connection."
#   type        = number
#   default     = 5
# }
#
# variable "batch_worker_batch_size" {
#   description = "Messages per batch worker invocation (1-10)"
#   type        = number
#   default     = 5
# }
#
# variable "batch_worker_reserved_concurrency" {
#   description = "Reserved concurrency for the batch worker - keep >= batch_worker_max_concurrency"
#   type        = number
#   default     = 5
# }
#
# variable "batch_worker_timeout" {
#   description = "Batch worker Lambda timeout in seconds; the batch queue visibility timeout is 6x this"
#   type        = number
#   default     = 60
# }
#
# variable "batch_max_receive_count" {
#   description = "Attempts per batch message before it is moved to the batch DLQ and its record marked FAILED"
#   type        = number
#   default     = 5
# }
