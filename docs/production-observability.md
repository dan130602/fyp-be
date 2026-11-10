# Production observability

## Live RDS reporting from a local backend (5 November 2026)

The default reader still uses the application's JDBC datasource. For local
development against the private Aurora cluster, an opt-in RDS Data API reader
reuses the same endpoint and ProductionMetrics calculations. Configure only
the backend (never the frontend):

```text
APP_OBSERVABILITY_SOURCE=rds-data
APP_OBSERVABILITY_RDS_CLUSTERARN=<Aurora cluster ARN>
APP_OBSERVABILITY_RDS_SECRETARN=<existing database secret ARN>
APP_OBSERVABILITY_RDS_DATABASE=ingestion
```

The backend uses its AWS SDK credential chain. AWS access must permit Data API
transactions/queries for that cluster and access to the configured secret. No
secret value is copied into source or returned to the browser. An expired AWS
session produces an error, not fallback metrics from the local database.

This option switches only the reporting reader. Local application sign-in still
uses the existing local user record. The reader additionally checks the verified
Cognito subject against an ACTIVE reporting role in RDS on every request. Local
roles alone cannot grant access to live results. It does not provision accounts,
change roles, run migrations, invoke AI, or start queue consumers in RDS.

Each read uses one repeatable-read, read-only Data API transaction, which is
rolled back after reading. Results and override history are paginated in that
snapshot. The SQL selects verdicts and telemetry, excluding evidence text,
transcripts, checklist wording and officer comments. The existing history IDs
are preserved. JDBC remains the deployment default; no cloud rollout is needed
to use this local mode.

Live verification found 23 saved outputs overall, 18 in the 30-day cohort,
one finalized review and zero override rows. The latest output was 26 October,
so the 7-day view was empty. Select **Last 30 days** to inspect existing records.
The reviewed metrics currently represent only one reviewed call. Missing cost
and latency metadata remains unavailable. These are observations as of the
verification time, not seeded dashboard values.

The local Docker reporting setup uses UTC, matching the deployed database's
zone-less timestamps. Keep that timezone when running this mode elsewhere.
Starting Aurora resumes normal AWS database charges; ECS can remain stopped.
Refresh the AWS session and restart the local container when credentials expire.

The older deployment notes below describe previous verification stages.

The Developer -> Production dashboard reads live application records through
`GET /api/observability?days=7`. Reporting allows DEVELOPER, ADMIN, SUPER_ADMIN and MANAGER. Management and officer screens and authentication
are unchanged. Existing accounts are not automatically reassigned.

## Developer role

DEVELOPER is a backend role stored in t_user.role and mirrored to the Cognito
user-pool group DEVELOPER. It can read production observability but has no user
administration or officer-review permission. The existing ADMIN, SUPER_ADMIN
and MANAGER reporting permissions remain in place.

The JWT authenticates identity; the database role determines backend access.
Creating a Cognito group or manually adding a member to it does not change the
internal role. Existing users keep their current role. First-time provisioning
keeps its existing default and bootstrap rules.

Deployment order:

1. The empty DEVELOPER group has been created and verified in the live ClearCall
   pool ap-southeast-1_JyRMtIWYI. It has no IAM role or user memberships.
   The existing Terraform Cloud workspace fyp-2026-dan/fyp-be must adopt it
   using infra/imports-developer.tf. Review and apply that import plan; the
   address is module.cognito.aws_cognito_user_group.roles["DEVELOPER"].
   The Cognito module's default role list already includes DEVELOPER. The import
   declaration prevents the next infrastructure run trying to create it again.
   No Terraform plan or apply has been run from this machine.
2. Deploy the backend role and observability endpoint. Group creation must come
   first because existing Cognito role synchronization visits all known groups.
3. An existing ADMIN or SUPER_ADMIN assigns the intended account through the
   existing PATCH /api/users/{userId}/role endpoint with body:

       {"role":"DEVELOPER"}

   That path updates the internal role and synchronizes Cognito membership.
   Use the normal administrator token; this is not self-service and the request
   must target another account. No user has been reassigned by this code change.
4. Deploy the matching Developer panel and open /developer?tab=production after
   signing in. The panel accepts the role returned by /api/users/me. Global
   role mapping, login routing, Management and officer screens are unchanged.

The existing role column is VARCHAR(30); adding the enum value needs no schema
migration. The application does not create Cognito groups at runtime or get any
new IAM policy. Cognito group definition is managed by Terraform.

AWS reference: https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-user-groups.html

## Existing data sources

- `t_langgraph_output`: the current saved AI output for each checklist.
- `t_checklist`: configured questions, call identity and final-review timestamp.
- `t_checklist_decision`: append-only officer overrides with their existing UUIDs,
  the AI verdict at the time, the officer verdict and decision timestamp.

This replaces the earlier local proposal for analysis snapshots and a telemetry
outbox. The new endpoint uses SELECT statements in a read-only, repeatable-read
transaction. It adds no migration, changes no existing entity/repository/review
code, and makes no AI calls. The former V14, write hooks and optional Langfuse
exporter have been parked outside the repositories under the workspace's
`tmp/observability-readonly-20261102/previous-write-instrumentation` folder.
Those earlier changes were never applied to production.

## Officer-approved reference

For each question in a finalized review, the newest decision row is the active
override. A NULL decision is a revert to the AI answer. If there is no active
override, the officer has accepted that AI answer. Unreviewed calls contribute
to decision coverage, but not FP/FN, recall, precision or review agreement.

FAIL (a breach) is the positive class:

| AI | Final officer reference | Matrix |
| --- | --- | --- |
| PASS | PASS | TN |
| PASS | FAIL | FN |
| FAIL | PASS | FP |
| FAIL | FAIL | TP |

- Decision coverage = PASS/FAIL predictions / all configured Stage 2 checks.
- Reviewed breach recall = TP / (TP + FN), using binary AI/officer pairs.
- Reviewed breach precision = TP / (TP + FP).
- Reviewed breach detection = TP / all officer FAIL cases, including unclear,
  N/A and failed AI predictions. This prevents unclear answers hiding misses.
- Missing results, duplicate item results and runtime prediction failures are
  separate from a correctly detected compliance breach.
- Comment-only edits and repeat submissions do not inflate current FP/FN.
  Reverting removes the active override and preserves the original history.
- The event feed retains original decision IDs and shows changes in the window;
  its history is distinct from the current confusion matrix.

The existing schema has no immutable AI revision links. A review predating the
saved output is excluded, as is an active override whose saved AI verdict
conflicts with that output. Conflicting latest decisions at the same timestamp
are excluded rather than assigned an arbitrary order. Timestamp compatibility
is a safeguard, not proof of immutable historical identity. Exact reconstruction
of overwritten AI results is unavailable with the current schema.

Officer decisions are the production reference requested by the team. These
metrics measure agreement with finalized reviews, not independent benchmark
accuracy. A finalized review with no override rows is valid approval; no-review
is not approval. Empty denominators return null, never automatic 100%.

## Windows, cost and timing

The 1-90 day activity window includes calls whose latest output or finalized
review occurred in the window. Older active overrides are still loaded so a
recent review does not lose an earlier decision.

Costs and timing cover the latest saved results in that cohort, including older
calls reviewed recently. They are not total production spend within the window:
`t_langgraph_output` overwrites previous attempts, which cannot be recovered.

When an output contains additive AI `observability` metadata, display its reported
chat-cost estimate, graph duration, stage timings and token usage. Unknown or
unpriced usage remains unavailable. Graph duration excludes queue time, output
writer, upload, STT and backend network overhead. Never infer latency by
subtracting database created/updated timestamps. Cost excludes embeddings, STT
and hosting, and remains an estimate rather than a provider invoice.

The existing backend already saves the raw AI JSON, so additive AI telemetry
needs no backend schema or write-workflow change. Deploy the AI instrumentation
for future timing/cost data; the decision dashboard works without it. Optional
telemetry makes no extra model requests. No Langfuse setup is required.

## Deployment and AWS access

Deploy the backend reporting endpoint and matching frontend. Existing Flyway
migrations V1-V13 are unchanged; this feature adds no migration. Deploy AI
instrumentation only to collect metadata on future normal analyses. No benchmark
rerun is required. The current backend and frontend GitHub workflows deploy
from main, so the maintainer must coordinate review and deployment.

The backend reuses its existing RDS DataSource. The infrastructure change declares the DEVELOPER Cognito group and imports the
already-created group. It changes no IAM policy, security group, database schema
or application account. The browser
fetches authenticated aggregates, never RDS credentials or direct SQL access.

The infrastructure enables Aurora's Data API for administrative tooling. For
an optional direct verification, the team must configure an approved AWS session
and a database principal with SELECT-only access to the three tables. An AWS
identity alone is not a database connection. Do not paste cloud or database
credentials into chat or the frontend. This machine had no usable
AWS CLI credentials in the earlier check, so the live database was not queried.
AWS access is now available through the official CLI browser sign-in under the
dedicated local profile clearcall-dev-role-setup. Only the Cognito group was
created; no production database query or account reassignment was performed.

On 2 Nov 2026, the deployed API was healthy, but an authenticated request to this
new reporting route returned 404. That deployment still needs the endpoint.
The test account was an active compliance officer and did not have reporting
access. No production role, schema or data was changed during verification.

## Validation and limits

Local tests exercise read mapping, finalized approval without overrides, FP/FN,
unclear and failed predictions, old overrides, reverts, stale reviews, ambiguous
timestamps, honest missing telemetry and HTTP authorization. They do not access
production, AWS, Cognito, Gemini or a live database. The dashboard polls every
minute and provides a visual override feed; it sends no email or pager alerts.

The endpoint aggregates the selected cohort in memory. Larger production volumes
will need SQL aggregation and pagination. That scaling work is separate from the
current read-only implementation.

## Completed cloud group setup

The DEVELOPER group is verified in the user pool used by the live public
frontend. Group creation did not assign any users, attach an IAM role or change
database contents. The backend and frontend role changes remain local and need
deployment. After backend rollout, an existing administrator can assign a
particular account through the normal role-management API described above.

The standard installed AWS CLI was too old for console sign-in. A vendor-signed
AWS CLI 2.39.3 copy was extracted under workspace tmp/aws-cli-console-login.
The existing global CLI was not replaced. The new CLI uses the dedicated
clearcall-dev-role-setup profile and AWS's normal credential cache outside the
repositories; no credentials are stored in application source or these notes.
