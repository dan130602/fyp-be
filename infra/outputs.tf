output "transcribe_ecr_repository_url" {
  description = "docker push target for the transcribe Lambda image"
  value       = module.ecr.repository_url
}

output "transcribe_ecr_repository_name" {
  value = module.ecr.repository_name
}

output "transcribe_ecr_github_actions_role_arn" {
  description = "Set this as the TRANSCRIBE_ECR_PUSH_ROLE_ARN repo variable in GitHub Actions"
  value       = module.ecr.github_actions_role_arn
}

# output "transcribe_lambda_function_name" {
#   value = module.lambda.lambda_function_name
# }

output "transcribe_lambda_dlq_url" {
  description = "Failed/throttled async invocation records of the trigger Lambda land here"
  value       = module.lambda.dlq_url
}

output "transcribe_status_queue_url" {
  description = "Set as TRANSCRIBE_STATUS_QUEUE_URL for the Spring Boot backend's status poller"
  value       = module.transcribe_status_queue.queue_url
}

output "transcribe_status_queue_arn" {
  value = module.transcribe_status_queue.queue_arn
}

output "transcribe_status_dlq_url" {
  description = "Status messages that failed processing max_receive_count times land here"
  value       = module.transcribe_status_queue.dlq_url
}

output "transcribe_status_event_rule_arn" {
  value = module.transcribe_status_events.rule_arn
}

output "transcript_handoff_queue_url" {
  description = "The next service reads its {mediaAssetId, transcriptUri, checklist} payload from here - set as TRANSCRIPT_HANDOFF_QUEUE_URL on media-processing-service, and consume it directly in whatever reads this from the other side"
  value       = module.transcript_handoff_queue.queue_url
}

output "transcript_handoff_queue_arn" {
  value = module.transcript_handoff_queue.queue_arn
}

output "transcript_handoff_dlq_url" {
  value = module.transcript_handoff_queue.dlq_url
}

# -----------------------
# Media Processing Service (ECS Fargate)
# -----------------------

output "media_processing_ecr_repository_url" {
  value = module.media_processing_ecr.repository_url
}

output "media_processing_ecr_github_actions_role_arn" {
  description = "Set this as the MEDIA_PROCESSING_ECR_PUSH_ROLE_ARN repo variable in GitHub Actions"
  value       = module.media_processing_ecr.github_actions_role_arn
}

output "media_processing_alb_dns_name" {
  description = "Public entrypoint for the media-processing-service"
  value       = module.media_processing_alb.alb_dns_name
}

output "media_processing_alb_acm_certificate_arn" {
  value = module.media_processing_alb.acm_certificate_arn
}

output "media_processing_alb_acm_certificate_status" {
  description = "Stays PENDING_VALIDATION until the validation CNAME below is created and AWS confirms it"
  value       = module.media_processing_alb.acm_certificate_status
}

output "media_processing_alb_acm_domain_validation_options" {
  description = "Create these as CNAME records manually in Route 53 to validate the cert (name -> value)"
  value       = module.media_processing_alb.acm_domain_validation_options
}

output "media_processing_ecs_cluster_name" {
  description = "Set this as the ECS_CLUSTER_NAME repo variable in GitHub Actions"
  value       = module.media_processing_ecs.cluster_name
}

output "media_processing_ecs_service_name" {
  description = "Set this as the ECS_SERVICE_NAME repo variable in GitHub Actions"
  value       = module.media_processing_ecs.service_name
}

output "media_processing_ecs_log_group_name" {
  description = "CloudWatch log group holding container stdout/stderr"
  value       = module.media_processing_ecs.log_group_name
}

output "media_processing_rds_credentials_secret_arn" {
  description = "{\"username\": \"...\", \"password\": \"...\"} — edit this in Secrets Manager to rotate the master password (then re-apply)"
  value       = module.media_processing_secrets.rds_credentials_secret_arn
}

output "media_processing_default_user_password_secret_arn" {
  description = "Raw seed password string for Cognito-provisioned users — edit this in Secrets Manager to change it"
  value       = module.media_processing_secrets.default_user_password_secret_arn
}

output "media_processing_rds_endpoint" {
  description = "Writer endpoint — also what SPRING_DATASOURCE_URL is built from"
  value       = module.media_processing_rds.cluster_endpoint
}

# -----------------------
# Frontend (S3 + CloudFront + ACM)
# -----------------------

output "frontend_acm_certificate_arn" {
  value = module.cloudfront.acm_certificate_arn
}

output "frontend_acm_certificate_status" {
  description = "Stays PENDING_VALIDATION until the domain is live, Route 53 is delegated, and the validation CNAME below is created"
  value       = module.cloudfront.acm_certificate_status
}

output "frontend_acm_domain_validation_options" {
  description = "Create these as CNAME records manually in Route 53 to validate the cert (name -> value)"
  value       = module.cloudfront.acm_domain_validation_options
}

output "frontend_cloudfront_distribution_id" {
  value = module.cloudfront.distribution_id
}

output "frontend_cloudfront_domain_name" {
  description = "Point a Route 53 ALIAS record at this (use frontend_cloudfront_hosted_zone_id as the alias target zone)"
  value       = module.cloudfront.distribution_domain_name
}

output "frontend_cloudfront_hosted_zone_id" {
  value = module.cloudfront.distribution_hosted_zone_id
}

# -----------------------
# ai-reasoning-service (ECS Fargate)
# -----------------------

output "ai_reasoning_service_ecr_repository_name" {
  description = "Set this as the BACKEND_ECR_REPOSITORY_NAME repo variable in the ai-reasoning-service GitHub repo"
  value       = module.ai_reasoning_service_ecr.repository_name
}

output "ai_reasoning_service_ecr_repository_url" {
  value = module.ai_reasoning_service_ecr.repository_url
}

output "ai_reasoning_service_ecr_github_actions_role_arn" {
  description = "Set this as the BACKEND_ECR_PUSH_ROLE_ARN repo variable in the ai-reasoning-service GitHub repo"
  value       = module.ai_reasoning_service_ecr.github_actions_role_arn
}

output "ai_reasoning_service_ecs_cluster_name" {
  description = "Set this as the ECS_CLUSTER_NAME repo variable in the ai-reasoning-service GitHub repo"
  value       = module.ai_reasoning_service_ecs.cluster_name
}

output "ai_reasoning_service_ecs_service_name" {
  description = "Set this as the ECS_SERVICE_NAME repo variable in the ai-reasoning-service GitHub repo"
  value       = module.ai_reasoning_service_ecs.service_name
}

output "ai_reasoning_service_ecs_log_group_name" {
  value = module.ai_reasoning_service_ecs.log_group_name
}

output "ai_reasoning_service_base_url" {
  description = "What media-processing-service calls (AI_BASE_URL) - resolvable only inside the VPC"
  value       = local.ai_reasoning_service_base_url
}

output "ai_reasoning_service_langsmith_api_key_secret_arn" {
  description = "Raw LangSmith API key string - fill in via Secrets Manager before the service first starts"
  value       = aws_secretsmanager_secret.ai_reasoning_service_langsmith_api_key.arn
}

output "batch_bucket_name" {
  description = "Presigned URLs for \"process later\" uploads point here; set as S3_BATCH_BUCKET on media-processing-service"
  value       = module.s3.ingestion_audio_batch_bucket_name
}

output "batch_queue_url" {
  value = module.batch_queue.queue_url
}

output "batch_dlq_url" {
  description = "Batch messages that failed max_receive_count times land here (their records are marked FAILED)"
  value       = module.batch_queue.dlq_url
}

output "batch_enqueuer_function_name" {
  description = "Invoke manually to run the nightly enqueue on demand (payload {\"dry_run\": true} to preview)"
  value       = module.batch_processing.enqueuer_function_name
}
