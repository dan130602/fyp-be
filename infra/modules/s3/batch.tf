# "Process later" bucket. Deliberately has NO aws_s3_bucket_notification - uploads here must not
# trigger the existing upload-complete flow (the Transcribe trigger Lambda is wired only to
# ingestion_audio in modules/lambda). Objects are picked up by the nightly batch enqueuer instead.
# Keys are "{media_asset_id}/{filename}" so each object maps back to its DB row.

resource "aws_s3_bucket" "ingestion_audio_batch" {
  bucket = var.ingestion_audio_batch_bucket_name
}

resource "aws_s3_bucket_public_access_block" "ingestion_audio_batch" {
  bucket                  = aws_s3_bucket.ingestion_audio_batch.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ingestion_audio_batch" {
  bucket = aws_s3_bucket.ingestion_audio_batch.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Browser uploads via presigned URL
resource "aws_s3_bucket_cors_configuration" "ingestion_audio_batch" {
  bucket = aws_s3_bucket.ingestion_audio_batch.id

  cors_rule {
    allowed_origins = var.frontend_origins
    allowed_methods = ["PUT", "GET", "HEAD"]
    allowed_headers = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# Safety net only: the backend deletes each object after its Transcribe job completes. This
# expires anything that was never processed (or whose delete failed) so nothing lingers forever.
resource "aws_s3_bucket_lifecycle_configuration" "ingestion_audio_batch" {
  bucket = aws_s3_bucket.ingestion_audio_batch.id

  rule {
    id     = "expire-unprocessed-batch-uploads"
    status = "Enabled"

    filter {}

    expiration {
      days = var.batch_expiration_in_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "ingestion_audio_batch" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.ingestion_audio_batch.arn,
      "${aws_s3_bucket.ingestion_audio_batch.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "ingestion_audio_batch" {
  bucket = aws_s3_bucket.ingestion_audio_batch.id
  policy = data.aws_iam_policy_document.ingestion_audio_batch.json

  depends_on = [aws_s3_bucket_public_access_block.ingestion_audio_batch]
}
