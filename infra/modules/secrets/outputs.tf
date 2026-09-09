output "rds_credentials_secret_arn" {
  description = "Fill this in via the console with {\"username\": \"...\", \"password\": \"...\"} — also what you connect with in the RDS console's Query Editor"
  value       = aws_secretsmanager_secret.rds_credentials.arn
}

output "default_user_password_secret_arn" {
  description = "Fill this in via the console with the raw seed password string"
  value       = aws_secretsmanager_secret.default_user_password.arn
}
