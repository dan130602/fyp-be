# Uncomment alongside the distribution resource in main.tf once the cert is ISSUED.

# output "distribution_id" {
#   value = aws_cloudfront_distribution.frontend.id
# }
#
# output "distribution_arn" {
#   value = aws_cloudfront_distribution.frontend.arn
# }
#
# output "distribution_domain_name" {
#   description = "Point your Route 53 alias record (or CNAME) at this"
#   value       = aws_cloudfront_distribution.frontend.domain_name
# }
#
# output "distribution_hosted_zone_id" {
#   description = "CloudFront's fixed hosted zone ID - needed for a Route 53 ALIAS record"
#   value       = aws_cloudfront_distribution.frontend.hosted_zone_id
# }

output "acm_certificate_arn" {
  value = aws_acm_certificate.frontend.arn
}

output "acm_certificate_status" {
  description = "Stays PENDING_VALIDATION until the domain is registered, Route 53 is delegated as its nameserver, and the DNS validation record below is created and AWS confirms it"
  value       = aws_acm_certificate.frontend.status
}

output "acm_domain_validation_options" {
  description = "Create this as a CNAME record manually in Route 53 to validate the certificate: name -> value"
  value = {
    for dvo in aws_acm_certificate.frontend.domain_validation_options :
    dvo.resource_record_name => dvo.resource_record_value
  }
}
