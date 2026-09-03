data "aws_caller_identity" "current" {}

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

# Transcribe uses the caller's (this Lambda's) IAM credentials to read the
# input media and write results, rather than a service-linked role — so the
# execution role itself needs S3 access on both buckets.
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
    sid    = "RunTranscribeJobs"
    effect = "Allow"
    actions = [
      "transcribe:StartTranscriptionJob",
      "transcribe:GetTranscriptionJob",
    ]

    resources = ["*"]
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
  image_uri    = "${var.ecr_repository_url}:${var.image_tag}"

  timeout     = var.timeout
  memory_size = var.memory_size

  environment {
    variables = {
      TRANSCRIBE_OUTPUT_BUCKET = var.output_bucket_name
      TRANSCRIBE_LANGUAGE_CODE = var.transcribe_language_code
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_basic_execution,
    aws_iam_role_policy.lambda,
  ]
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
