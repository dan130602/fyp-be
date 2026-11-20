# # Adopt the DEVELOPER group created in the verified ClearCall user pool.
# # The next Terraform Cloud apply imports it into the existing workspace state.
# import {
#   to = module.cognito.aws_cognito_user_group.roles["DEVELOPER"]
#   id = "ap-southeast-1_JyRMtIWYI/DEVELOPER"
# }
