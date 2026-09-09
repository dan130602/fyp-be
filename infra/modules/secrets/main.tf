resource "aws_secretsmanager_secret" "rds_credentials" {
  name = "${var.name_prefix}/rds-credentials"
}

resource "aws_secretsmanager_secret" "default_user_password" {
  name = "${var.name_prefix}/default-user-password"
  description = "The password Cognito assigns to every app-provisioned user"
}
