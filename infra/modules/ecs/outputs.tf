output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "service_name" {
  value = aws_ecs_service.this.name
}

output "task_execution_role_arn" {
  value = aws_iam_role.execution.arn
}

output "task_role_arn" {
  value = aws_iam_role.task.arn
}

output "task_role_name" {
  value = aws_iam_role.task.name
}

output "security_group_id" {
  value = aws_security_group.tasks.id
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.this.name
}

output "service_discovery_name" {
  description = "Hostname prefix registered in Cloud Map (null when service discovery is off)"
  value       = one(aws_service_discovery_service.this[*].name)
}
