locals {
  secured_buckets = {
    ingestion_audio    = aws_s3_bucket.ingestion_audio
  }
}

# resource "aws_s3_bucket" "frontend" {
#   bucket = var.frontend_bucket_name
# }

resource "aws_s3_bucket" "ingestion_audio" {
  bucket = var.ingestion_audio_bucket_name
}


resource "aws_s3_bucket_public_access_block" "secured_buckets_config" {
  for_each                = local.secured_buckets
  bucket                  = each.value.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# Allow browser uploads via presigned URL (CORS)
resource "aws_s3_bucket_cors_configuration" "this" {
  for_each = local.secured_buckets
  bucket   = each.value.id

  cors_rule {
    allowed_origins = var.frontend_origins
    allowed_methods = ["PUT", "GET", "HEAD"]
    allowed_headers = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# Optional: bucket policy to restrict writes to uploads/* to the presign role only
data "aws_iam_policy_document" "audio_ingestion_bucket_policy" {
  count = var.presign_role_arn == null ? 0 : 1

  statement {
    sid    = "AllowPresignedUploadsFromBackendRoleToUploadsPrefixOnly"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = [var.presign_role_arn]
    }

    actions = [
      "s3:PutObject",
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts"
    ]

    resources = [
      "${aws_s3_bucket.ingestion_audio.arn}/uploads/*"
    ]
  }

  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.ingestion_audio.arn,
      "${aws_s3_bucket.ingestion_audio.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}
resource "aws_s3_bucket_policy" "videos" {
  count  = var.presign_role_arn == null ? 0 : 1
  bucket = aws_s3_bucket.ingestion_audio.id
  policy = data.aws_iam_policy_document.audio_ingestion_bucket_policy[0].json
}

resource "aws_s3_bucket_lifecycle_configuration" "secured_buckets_lifecycle" {
  for_each = {
    for k, v in local.secured_buckets : k => v
    if !contains(["processed", "livestream"], k)
  }
  bucket = each.value.id

  rule {
    expiration {
      days = var.expiration_in_days
    }

    filter {
      prefix = "uploads/"
    }
    id     = "7 day expiry for uploaded videos"
    status = "Enabled"
  }
}
