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

output "media_processing_rds_endpoint" {
  value = module.media_processing_rds.cluster_endpoint
}

output "media_processing_rds_secret_arn" {
  description = "Managed master credentials — in the RDS console, open Query Editor on this cluster and connect with this secret to run SQL directly"
  value       = module.media_processing_rds.master_user_secret_arn
}
