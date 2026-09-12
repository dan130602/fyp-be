output "alb_dns_name" {
  value = aws_lb.this.dns_name
}

output "alb_security_group_id" {
  value = aws_security_group.alb.id
}

output "target_group_arn" {
  value = aws_lb_target_group.this.arn
}

output "listener_arn" {
  value = aws_lb_listener.http.arn
}

output "acm_certificate_arn" {
  value = aws_acm_certificate.this.arn
}

output "acm_certificate_status" {
  description = "Stays PENDING_VALIDATION until the validation CNAME below is created and AWS confirms it"
  value       = aws_acm_certificate.this.status
}

output "acm_domain_validation_options" {
  description = "Create this as a CNAME record manually in Route 53 to validate the certificate: name -> value"
  value = {
    for dvo in aws_acm_certificate.this.domain_validation_options :
    dvo.resource_record_name => dvo.resource_record_value
  }
}

# Uncomment alongside the https listener resource in main.tf once the cert is ISSUED.

# output "https_listener_arn" {
#   value = aws_lb_listener.https.arn
# }
