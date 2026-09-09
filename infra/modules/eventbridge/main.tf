resource "aws_cloudwatch_event_rule" "this" {
  name          = var.rule_name
  event_pattern = var.event_pattern
}

resource "aws_cloudwatch_event_target" "this" {
  rule = aws_cloudwatch_event_rule.this.name
  arn  = var.target_arn
}
