
output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = aws_cognito_user_pool.main.id
}

output "cognito_user_pool_arn" {
  description = "Cognito User Pool ARN"
  value       = aws_cognito_user_pool.main.arn
}

output "cognito_user_pool_client_id" {
  description = "Cognito App Client ID (for frontend)"
  value       = aws_cognito_user_pool_client.web.id
}

output "cognito_domain_prefix" {
  description = "Cognito Hosted UI domain prefix"
  value       = aws_cognito_user_pool_domain.domain.domain
}

output "cognito_issuer_uri" {
  description = "Issuer URI used by Spring Security to validate JWTs"
  value       = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.main.id}"
}

output "cognito_userinfo_url" {
  description = "OIDC userInfo endpoint URL used by backend to fetch email/username"
  value       = "https://${aws_cognito_user_pool_domain.domain.domain}.auth.${var.aws_region}.amazoncognito.com/oauth2/userInfo"
}