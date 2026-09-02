output "videos_bucket_arn"{
  value = aws_s3_bucket.videos.arn
}

output "videos_bucket_name" {
  description = "S3 bucket name for video uploads"
  value       = aws_s3_bucket.videos.bucket
}

output "processed_videos_arn"{
  value = aws_s3_bucket.processed_videos.arn
}

output "processed_videos_name" {
  description = "S3 bucket name for video uploads"
  value       = aws_s3_bucket.processed_videos.bucket
}

output "processed_videos_bucket_id" {
  value = aws_s3_bucket.processed_videos.id
}

output "videos_bucket_id" {
  value = aws_s3_bucket.videos.id
}

output "live_video_bucket_arn" {
  value = aws_s3_bucket.livestream.arn
}

output "live_video_bucket_name" {
  value = aws_s3_bucket.livestream.bucket
}

output "frontend_bucket_regional_domain_name" {
  value = aws_s3_bucket.frontend.bucket_regional_domain_name
}