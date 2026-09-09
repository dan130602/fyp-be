output "cluster_endpoint" {
  description = "Writer endpoint — use for SPRING_DATASOURCE_URL"
  value       = aws_rds_cluster.this.endpoint
}

output "cluster_reader_endpoint" {
  value = aws_rds_cluster.this.reader_endpoint
}

output "cluster_arn" {
  value = aws_rds_cluster.this.arn
}

output "cluster_id" {
  value = aws_rds_cluster.this.cluster_identifier
}

output "database_name" {
  value = aws_rds_cluster.this.database_name
}

output "port" {
  value = aws_rds_cluster.this.port
}

output "security_group_id" {
  value = aws_security_group.this.id
}
