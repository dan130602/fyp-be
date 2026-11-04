# Prepared developer-role backend release

Prepared from backend main commit d8bce10 on 4 Nov 2026. Includes the existing
management dashboard work, DEVELOPER support and the read-only production
observability endpoint. No existing SQL migrations or review workflow changed.

Validation: 161 backend tests passed, zero failures/errors/skips.

## Current deployment state

At preparation time the ECS cluster media-processing-service-cluster, service
media-processing-service, used task definition media-processing-service:14 with
desiredCount=0 and runningCount=0. The owner explicitly requested that the backend
remain stopped. No ECS update or user-role mutation was performed.

The DEVELOPER group exists in Cognito pool ap-southeast-1_JyRMtIWYI. The selected
existing account was last observed in COMPLIANCE_OFFICER. Cognito login and the
application database role are distinct; group membership alone is insufficient.

## Deployment when the owner authorizes startup

1. Fetch main again and check for changes since d8bce10 before releasing.
2. Build the existing media-processing-service/Dockerfile from this branch, tag
   the image with the release commit and push that immutable tag to the existing
   media-processing-service ECR repository. Do not overwrite latest.
3. Record the then-current ECS task definition and image digest for rollback.
   Register a new revision copied from that definition, changing only the backend
   image to the release digest. Preserve environment, secrets, IAM and networking.
4. Update the existing service to that revision and the explicitly approved task
   count. Wait for ECS stability and verify backend health before role assignment.
5. An existing ADMIN or SUPER_ADMIN signs into ClearCall and resolves the selected
   account UUID using GET /api/users. Confirm its identity and current role.
6. Call PATCH /api/users/{userId}/role with JSON {"role":"DEVELOPER"} using that
   administrator session. The administrator cannot change their own role. Check
   both the database-backed response and Cognito membership after the operation;
   the existing service writes the database before Cognito and is not atomic.
7. Sign into the target account and verify GET /api/users/me reports DEVELOPER and
   GET /api/observability?days=7 returns production data. Developer access does not
   grant officer review or role-administration permissions.

Do not put tokens or passwords in this document or shell history. Use the normal
application administrator session; do not use bootstrap-admin configuration to
promote the selected account.

If rolling back after role assignment, first use the supported admin endpoint to
restore the account to its recorded original role and verify Cognito membership.
Only then roll back to a backend version that lacks the DEVELOPER enum value.

Terraform import declaration is prepared separately; no full infrastructure
apply is required to use the already-created group. Review its plan before any
future Terraform apply. This release does not merge or push to main.
