# # ---------- Global ----------
# aws_region = "ap-southeast-1"
# project    = "oitnb"
# env        = "fsd"
#
# # ---------- Certificates ----------
# acm_certificate_arn_fe_cloudfront = "arn:aws:acm:us-east-1:851725490631:certificate/900d6285-faab-4124-99ac-cad7d9edcecc"
# alb_acm_certificate_arn           = "arn:aws:acm:ap-southeast-1:851725490631:certificate/e321781c-6eb3-4fed-b298-4a10f94464e6"
#
# # ---------- Cognito ----------
# cognito_domain_prefix   = "ap-southeast-13q5n9zcqg"
# cognito_user_pool_name  = "User pool - kutyt2"
#
# cognito_callback_urls = [
#   "https://dpga1vyb9gjzn.cloudfront.net/",
#   "http://localhost:8081",
#   "https://cloudgridapp.com/",
#   "https://cloudgridapp.com"
# ]
#
# cognito_logout_urls = [
#   "https://dpga1vyb9gjzn.cloudfront.net/",
#   "http://localhost:8081",
#   "https://cloudgridapp.com/",
#   "https://cloudgridapp.com"
# ]
#
# # ---------- S3 ----------
# videos_bucket_name           = "oitnb-videos-dev"
# frontend_bucket_name         = "fsd-frontend-bucket"
# processed_videos_bucket_name = "fsd-processed-2026"
# livestream_bucket_name       = "fsd-live-2026"
# expiration_in_days           = 7
#
# frontend_origins = [
#   "http://localhost:8081",
#   "https://cloudgridapp.com/"
# ]
#
# presign_role_arn = "arn:aws:iam::851725490631:role/VideoServicePresignRole"
#
# # ---------- SQS ----------
# video_processing_queue        = "video_processing_queue"
# video_processing_status_queue = "video_processing_status_queue"
# livestream_input_queue        = "livestream_queue"
# livestream_status_queue       = "livestream_status_queue"
#
# # ---------- DynamoDB / Tables ----------
# comments_table_name      = "comments-table"
# live_comments_table_name = "live-comments-table"
#
# # ---------- Database ----------
# db_name               = "video_db"
# db_username           = "postgres"
# db_password           = "Ilovefsd2026"
# secrets_manager_arn   = "arn:aws:secretsmanager:ap-southeast-1:851725490631:secret:db_password-CT2ecM"
# internal_service_token = "x8Qm1vN4pR2kT7zJc9Lh5sDf0Wb3YgUa_Mn6eKq2Pd4"
#
# # ---------- Networking ----------
# vpc_cidr            = "10.0.0.0/16"
# public_subnet_cidr  = "10.0.0.0/20"
# private_subnet_cidr = "10.0.128.0/20"