# Nightly "process later" path:
#   EventBridge Scheduler -> enqueuer Lambda -> batch SQS queue -> worker Lambda -> Transcribe
#
# Both Lambdas run the SAME container image as the immediate trigger Lambda (modules/lambda); they
# differ only in handler (image_config.command) and configuration. The immediate Lambda is not
# touched. These two need the DB (status transitions), so they sit in the private subnets - the
# RDS security group already admits the whole private-subnet CIDR range.

data "aws_caller_identity" "current" {}

data "aws_ecr_image" "this" {
  repository_name = var.ecr_repository_name
  image_tag       = var.image_tag
}

locals {
  image_uri   = "${var.ecr_repository_url}@${data.aws_ecr_image.this.image_digest}"
  worker_name = "${var.name_prefix}-worker"
  enq_name    = "${var.name_prefix}-enqueuer"

  db_env = {
    DB_HOST       = var.db_host
    DB_PORT       = tostring(var.db_port)
    DB_NAME       = var.db_name
    DB_SECRET_ARN = var.db_secret_arn
  }
}

resource "aws_security_group" "lambda" {
  name        = "${var.name_prefix}-lambda-sg"
  description = "Batch enqueuer/worker Lambdas: outbound only (Postgres, SQS, S3, Transcribe)"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# -----------------------
# Worker (batch queue consumer)
# -----------------------

resource "aws_iam_role" "worker" {
  name = "${local.worker_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Logs + the ENI permissions a VPC-attached function needs
resource "aws_iam_role_policy_attachment" "worker_vpc_execution" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

data "aws_iam_policy_document" "worker" {
  statement {
    sid       = "ReadBatchAudio"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${var.batch_bucket_arn}/*"]
  }

  statement {
    sid    = "WriteTranscriptOutput"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      var.output_bucket_arn,
      "${var.output_bucket_arn}/*",
    ]
  }

  statement {
    sid    = "RunTranscribeJobs"
    effect = "Allow"
    actions = [
      "transcribe:StartTranscriptionJob",
      "transcribe:GetTranscriptionJob",
    ]
    resources = ["*"]
  }

  # Required by the SQS event source mapping, which polls with this execution role
  statement {
    sid    = "ConsumeBatchQueue"
    effect = "Allow"
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = [var.queue_arn]
  }

  statement {
    sid       = "ReadDbCredentials"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.db_secret_arn]
  }
}

resource "aws_iam_role_policy" "worker" {
  name   = "${local.worker_name}-policy"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker.json
}

resource "aws_lambda_function" "worker" {
  function_name = local.worker_name
  role          = aws_iam_role.worker.arn

  package_type = "Image"
  image_uri    = local.image_uri
  image_config {
    command = ["app.handler.handler"]
  }

  timeout     = var.worker_timeout
  memory_size = var.worker_memory_size

  # Hard ceiling on parallel invocations. Keep >= worker_max_concurrency: if it is lower, the
  # event source mapping gets throttled, and each throttled delivery still counts toward the
  # queue's max receive count (risking premature DLQ).
  reserved_concurrent_executions = var.worker_reserved_concurrency

  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = merge(local.db_env, {
      TRANSCRIBE_OUTPUT_BUCKET      = var.output_bucket_name
      TRANSCRIBE_LANGUAGE_CODE      = var.transcribe_language_code
      TRANSCRIBE_MAX_SPEAKER_LABELS = tostring(var.transcribe_max_speaker_labels)
      # Same value as the queue's redrive maxReceiveCount - the worker marks a record FAILED on
      # its last allowed attempt, just before SQS moves it to the DLQ.
      BATCH_MAX_RECEIVE_COUNT = tostring(var.max_receive_count)
    })
  }

  depends_on = [
    aws_iam_role_policy_attachment.worker_vpc_execution,
    aws_iam_role_policy.worker,
  ]
}

# The rate limiter. Throughput through the batch queue is bounded by:
#   maximum_concurrency (parallel invocations) x batch_size (messages per invocation)
resource "aws_lambda_event_source_mapping" "worker" {
  event_source_arn = var.queue_arn
  function_name    = aws_lambda_function.worker.arn

  batch_size                         = var.worker_batch_size
  maximum_batching_window_in_seconds = 0

  # Lets the handler fail single records instead of the whole batch
  function_response_types = ["ReportBatchItemFailures"]

  scaling_config {
    maximum_concurrency = var.worker_max_concurrency
  }
}

# -----------------------
# Enqueuer (scheduled)
# -----------------------

resource "aws_iam_role" "enqueuer" {
  name = "${local.enq_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "enqueuer_vpc_execution" {
  role       = aws_iam_role.enqueuer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

# List only - the enqueuer never reads or deletes objects
data "aws_iam_policy_document" "enqueuer" {
  statement {
    sid       = "ListBatchBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [var.batch_bucket_arn]
  }

  statement {
    sid       = "EnqueueBatchWork"
    effect    = "Allow"
    actions   = ["sqs:SendMessage"]
    resources = [var.queue_arn]
  }

  statement {
    sid       = "ReadDbCredentials"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.db_secret_arn]
  }
}

resource "aws_iam_role_policy" "enqueuer" {
  name   = "${local.enq_name}-policy"
  role   = aws_iam_role.enqueuer.id
  policy = data.aws_iam_policy_document.enqueuer.json
}

resource "aws_lambda_function" "enqueuer" {
  function_name = local.enq_name
  role          = aws_iam_role.enqueuer.arn

  package_type = "Image"
  image_uri    = local.image_uri
  image_config {
    command = ["app.enqueuer.handler"]
  }

  # One DB round trip + one SQS send per object, so this bounds how large a night it can enqueue
  timeout     = var.enqueuer_timeout
  memory_size = 256

  # Overlapping runs are safe (rows are claimed atomically) but pointless
  reserved_concurrent_executions = 1

  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = merge(local.db_env, {
      BATCH_BUCKET    = var.batch_bucket_name
      BATCH_QUEUE_URL = var.queue_url
    })
  }

  depends_on = [
    aws_iam_role_policy_attachment.enqueuer_vpc_execution,
    aws_iam_role_policy.enqueuer,
  ]
}

# -----------------------
# EventBridge Scheduler
# -----------------------

resource "aws_iam_role" "scheduler" {
  name = "${var.name_prefix}-scheduler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "scheduler.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
      }
    }]
  })
}

resource "aws_iam_role_policy" "scheduler" {
  name = "${var.name_prefix}-scheduler-policy"
  role = aws_iam_role.scheduler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["lambda:InvokeFunction"]
      Resource = [aws_lambda_function.enqueuer.arn]
    }]
  })
}

resource "aws_scheduler_schedule" "nightly" {
  name  = "${var.name_prefix}-nightly"
  state = var.schedule_enabled ? "ENABLED" : "DISABLED"

  schedule_expression          = var.schedule_expression
  schedule_expression_timezone = var.schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.enqueuer.arn
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({})

    # The enqueuer is idempotent, so retrying a failed run is safe
    retry_policy {
      maximum_retry_attempts       = 2
      maximum_event_age_in_seconds = 3600
    }
  }
}
