output "ingestion_audio_bucket_id" {
  description = "ID of the audio ingestion (input) bucket"
  value       = aws_s3_bucket.ingestion_audio.id
}

output "ingestion_audio_bucket_arn" {
  description = "ARN of the audio ingestion (input) bucket"
  value       = aws_s3_bucket.ingestion_audio.arn
}

output "ingestion_audio_bucket_name" {
  description = "Name of the audio ingestion (input) bucket"
  value       = aws_s3_bucket.ingestion_audio.bucket
}

output "ingestion_audio_output_bucket_id" {
  description = "ID of the transcribed audio (output) bucket"
  value       = aws_s3_bucket.ingestion_audio_output.id
}

output "ingestion_audio_output_bucket_arn" {
  description = "ARN of the transcribed audio (output) bucket"
  value       = aws_s3_bucket.ingestion_audio_output.arn
}

output "ingestion_audio_output_bucket_name" {
  description = "Name of the transcribed audio (output) bucket"
  value       = aws_s3_bucket.ingestion_audio_output.bucket
}
