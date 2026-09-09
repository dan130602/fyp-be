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
