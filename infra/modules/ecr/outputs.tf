output "repository_url" {
  description = "Push Docker images here, e.g. docker push <repository_url>:latest"
  value       = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  value = aws_ecr_repository.this.arn
}

output "repository_name" {
  value = aws_ecr_repository.this.name
}

output "github_actions_role_arn" {
  description = "Set as the role-to-assume input for aws-actions/configure-aws-credentials in CI"
  value       = aws_iam_role.github_actions_push.arn
}

output "github_actions_role_name" {
  description = "Lets callers attach extra inline policies (e.g. ecs:UpdateService) to the same CI role"
  value       = aws_iam_role.github_actions_push.name
}
