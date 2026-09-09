data "aws_caller_identity" "current" {}

# Resolved at plan time so `terraform apply` picks up whatever image CI sent recently
data "aws_ecr_image" "this" {
  repository_name = var.ecr_repository_name
  image_tag       = var.image_tag
}

resource "aws_iam_role" "lambda" {
  name = "${var.lambda_function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_sqs_queue" "dlq" {
  name                      = "${var.lambda_function_name}-dlq"
  message_retention_seconds = var.dlq_message_retention_seconds
}

# Transcribe assumes this to read the input audio and write the transcript - work it does
# minutes after the Lambda that started the job has exited and its credentials are gone,
# so it cannot borrow the function's identity. StartCallAnalyticsJob requires it; plain
# StartTranscriptionJob did not, which is why this role is new.
resource "aws_iam_role" "transcribe_data_access" {
  name = "${var.lambda_function_name}-transcribe-data-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "transcribe.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

data "aws_iam_policy_document" "transcribe_data_access" {
  statement {
    sid       = "ReadInputAudio"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${var.input_bucket_arn}/*"]
  }

  statement {
    sid     = "WriteTranscriptOutput"
    effect  = "Allow"
    actions = ["s3:PutObject"]

    resources = ["${var.output_bucket_arn}/*"]
  }

  # ListBucket acts on the bucket, so it takes the bare ARN - object actions above take
  # /*. Mixing the two up is the usual way this role looks right and fails at run time.
  statement {
    sid     = "ListBuckets"
    effect  = "Allow"
    actions = ["s3:ListBucket"]

    resources = [
      var.input_bucket_arn,
      var.output_bucket_arn,
    ]
  }
}

resource "aws_iam_role_policy" "transcribe_data_access" {
  name   = "${var.lambda_function_name}-transcribe-data-access-policy"
  role   = aws_iam_role.transcribe_data_access.id
  policy = data.aws_iam_policy_document.transcribe_data_access.json
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid     = "ReadInputAudio"
    effect  = "Allow"
    actions = ["s3:GetObject"]

    resources = ["${var.input_bucket_arn}/*"]
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
    sid       = "RunCallAnalyticsJobs"
    effect    = "Allow"
    actions   = ["transcribe:StartCallAnalyticsJob"]
    resources = ["*"]
  }

  # Naming a role in StartCallAnalyticsJob is itself privileged - without this scope the
  # function could hand Transcribe any role in the account and have it act as that role.
  statement {
    sid       = "PassTranscribeDataAccessRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.transcribe_data_access.arn]
  }

  # Exercised by the Lambda service (not handler.py) when delivering an async
  # on-failure destination record, using this function's own execution role.
  statement {
    sid       = "SendToDlq"
    effect    = "Allow"
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.dlq.arn]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${var.lambda_function_name}-policy"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

resource "aws_lambda_function" "this" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda.arn

  package_type = "Image"
  image_uri    = "${var.ecr_repository_url}@${data.aws_ecr_image.this.image_digest}"

  timeout                        = var.timeout
  memory_size                    = var.memory_size
  reserved_concurrent_executions = var.reserved_concurrent_executions

  environment {
    variables = {
      TRANSCRIBE_OUTPUT_BUCKET        = var.output_bucket_name
      TRANSCRIBE_DATA_ACCESS_ROLE_ARN = aws_iam_role.transcribe_data_access.arn
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic_execution,
    aws_iam_role_policy.lambda,
    aws_iam_role_policy.transcribe_data_access,
  ]
}

# S3 invokes this Lambda asynchronously - a throttled (over reserved_concurrent_executions)
# invocation that exhausts its retries is delivered here instead of silently vanishing.
resource "aws_lambda_function_event_invoke_config" "this" {
  function_name          = aws_lambda_function.this.function_name
  maximum_retry_attempts = var.async_max_retry_attempts

  destination_config {
    on_failure {
      destination = aws_sqs_queue.dlq.arn
    }
  }
}

resource "aws_lambda_permission" "allow_s3_invoke" {
  statement_id   = "AllowS3Invoke"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.this.function_name
  principal      = "s3.amazonaws.com"
  source_arn     = var.input_bucket_arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "input_bucket_trigger" {
  bucket = var.input_bucket_id

  lambda_function {
    lambda_function_arn = aws_lambda_function.this.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = var.input_bucket_prefix
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}
