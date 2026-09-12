output "distribution_id" {
  value = aws_cloudfront_distribution.frontend.id
}

output "distribution_arn" {
  value = aws_cloudfront_distribution.frontend.arn
}

output "distribution_domain_name" {
  description = "Serve/access the frontend at this CloudFront domain (e.g. set as CORS_ALLOWED_ORIGINS / Cognito callback URLs)"
  value       = aws_cloudfront_distribution.frontend.domain_name
}
