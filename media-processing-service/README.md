# media-processing-service

Spring Boot backend for ClearCall: it takes uploaded advisor call recordings, tracks them through Amazon Transcribe and the fyp-ai compliance pipeline, and serves the results to compliance officers, managers and admins.

API documentation:

- [docs/api.html](docs/api.html): **start here.** The full endpoint reference as one self-contained page that works offline (Redoc). Just open it in a browser.
- [docs/API.md](docs/API.md) / [docs/API.pdf](docs/API.pdf): readable reference with workflow rules, ownership and dashboard definitions.
- [docs/openapi.json](docs/openapi.json): OpenAPI 3.1 spec, generated from the code. Import it into Swagger UI, Redoc, Postman or a client generator.
- [docs/postman_collection.json](docs/postman_collection.json): Postman collection built from the spec. See [API docs](#api-docs) below.

## What it does

1. **Upload.** The frontend asks for presigned S3 URLs (`POST /api/uploads/presigned-urls`) and uploads the audio straight to S3. The service never handles audio bytes itself. Each file gets a `t_media_asset` row and a `t_checklist` row when the URL is issued.
2. **Transcription.** S3 events start Amazon Transcribe in the trigger Lambda (`lambda/transcribe-trigger`), not here. This service finds out a job finished by polling an SQS queue that EventBridge fills with "Transcribe Job State Change" events (`TranscribeStatusPollingService`).
3. **Handoff.** When a job completes, the asset is marked `COMPLETED` and a `{mediaAssetId, transcriptUri, checklistId}` message goes onto the transcript-handoff queue (`TranscriptHandoffService`).
4. **AI analysis.** A second poller (`ComplianceAnalysisService`) reads that queue, loads the transcript from S3, calls fyp-ai's LangGraph server (`POST /runs/wait`, assistant `compliance`) and stores the result in `t_langgraph_output`.
5. **Review.** A compliance officer opens the analysis, overrides any AI verdicts they disagree with, and submits. Every decision is kept in `t_checklist_decision` as an audit log.
6. **Reporting.** The management dashboard (`/api/dashboard/*`) and the developer observability view (`/api/observability`) compute their figures from that stored data.

```
Browser ──presigned PUT──▶ S3 (main bucket) ──S3 event──▶ transcribe-trigger Lambda ──▶ Amazon Transcribe
   │                                                                                       │
   │ REST + Cognito JWT                                                  EventBridge ─▶ SQS: transcribe-status
   ▼                                                                                       │
media-processing-service ◀──────────────────── TranscribeStatusPollingService ◀────────────┘
   │  └─ TranscriptHandoffService ─▶ SQS: transcript-handoff ─▶ ComplianceAnalysisService ─▶ fyp-ai (/runs/wait)
   ▼
PostgreSQL (Flyway-managed)
```

### Immediate vs batch uploads

| Mode | Bucket | Object key | Status flow |
|---|---|---|---|
| `immediate` (default) | `S3_BUCKET` (has the S3 trigger) | `uploads/{userId}/{uuid}-{filename}` | `PENDING → COMPLETED \| FAILED` |
| `batch` | `S3_BATCH_BUCKET` (no trigger; a nightly enqueuer Lambda picks files up) | `{assetId}/{filename}` | `PENDING → QUEUED → PROCESSING → COMPLETED \| FAILED` |

`QUEUED` and `PROCESSING` are written by the batch Lambdas, not by this service (`models/MediaAssetStatus.java`). Batch objects are deleted from the batch bucket once the job completes (`MediaAssetService.applyTranscribeStatusUpdate`). Batch mode is refused with a 400 if `S3_BATCH_BUCKET` is blank.

The Transcribe job name is derived from the object key (`PresignedUrlService.jobNameForObjectKey`). It must stay byte-identical to `_job_name_for_key` in `lambda/transcribe-trigger/app/handler.py`, because the job name is the only link between a Transcribe job and its database row.

## Tech stack

- Java 25, Spring Boot 4.0.8 (Web MVC, Data JPA, Security, OAuth2 Resource Server, Validation)
- PostgreSQL with Flyway migrations (`src/main/resources/db/migration`, V1–V13)
- AWS SDK v2: S3, SQS, Cognito Identity Provider, RDS Data API
- Lombok
- Jackson 3 (`tools.jackson.*`)

## Running locally

### Prerequisites

- JDK 25 (or use Docker)
- Docker, for PostgreSQL
- AWS credentials that can presign S3 URLs and call Cognito admin APIs, if you want uploads and user management to work
- Optional: fyp-ai running on `http://localhost:2024` for AI analysis

### With Maven

```bash
# from the repo root: start only the database (exposed on localhost:5433)
docker compose up -d postgres

# from this folder
./mvnw spring-boot:run
```

The defaults in `application.properties` point at `jdbc:postgresql://localhost:5433/ingestion` (user `postgres`, password `root`), which matches the compose database. Flyway applies the migrations on startup, and Hibernate runs with `ddl-auto=validate`, so the schema must match the entities.

### With Docker Compose

From the repo root:

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export AWS_COGNITO_USER_POOL_ID=... AWS_COGNITO_APP_CLIENT_ID=...
docker compose up --build
```

The compose file requires those four variables and fails fast if they are missing. The service listens on `http://localhost:8080`.

### Tests

```bash
./mvnw test
```

Tests are unit and slice tests under `src/test/java` (controllers, services, security converter, RDS readers).

### First login

A fresh database has no users, and only an admin can create users. There are two ways in:

- `POST /api/users/first-admin` (no auth) creates a SUPER_ADMIN in Cognito and the database. It returns 403 as soon as any user exists.
- Any email listed in `BOOTSTRAP_ADMIN_EMAILS` gets SUPER_ADMIN the first time it signs in. Everyone else is auto-provisioned as `COMPLIANCE_OFFICER` on first sign-in (`UserProvisioningService.DEFAULT_ROLE`).

## API docs

The OpenAPI spec is generated by [springdoc-openapi](https://springdoc.org/) from the controllers and DTOs. It is **off by default** so production never exposes it. Turn it on with `API_DOCS_ENABLED=true`:

```bash
API_DOCS_ENABLED=true ./mvnw spring-boot:run
```

- Swagger UI: http://localhost:8080/swagger-ui.html (click **Authorize** and paste a Cognito token to try calls)
- Spec: http://localhost:8080/v3/api-docs (JSON) or `/v3/api-docs.yaml`

Each operation's role list is copied automatically from its `@PreAuthorize` annotation (`configs/OpenApiConfig.java`), so the annotations stay the only place roles are defined. When you declare any `@ApiResponse` on an endpoint, also declare its success code (`200`/`201`), because springdoc then stops generating the success response itself.

### Regenerating the committed files after an API change

```bash
# with the app running as above
curl -s localhost:8080/v3/api-docs | python3 -m json.tool > docs/openapi.json

# Readable HTML page (single file, works offline)
npx @redocly/cli@2.72.1 build-docs docs/openapi.json -o docs/api.html \
  --title "ClearCall media-processing-service API" --inlineBundle

# PDF of API.md (needs pandoc and Google Chrome). The sed turns links to sibling
# files into plain names, which would otherwise point at the temp folder in the PDF.
sed -E 's/\[([^]]+)\]\((openapi\.json|postman_collection\.json)\)/`\1`/g' docs/API.md > /tmp/API.print.md
pandoc /tmp/API.print.md -f gfm -t html5 -s --embed-resources --css docs/print.css --metadata pagetitle="ClearCall API" -o /tmp/API.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --print-to-pdf=docs/API.pdf file:///tmp/API.html

# Postman collection (grouped by tag)
npx openapi-to-postmanv2@6.4.2 -s docs/openapi.json -o docs/postman_collection.json -p -O folderStrategy=Tags
```

The committed collection was also hand-tuned after conversion: a `raw` URL on every request (the converter leaves it out, which can leave the Postman URL bar empty), collection variables (`baseUrl`, `bearerToken`, `assetId`, `userId`, `advisorId`), `noauth` on the two public endpoints, and realistic request bodies. Re-apply those after regenerating, or edit the existing collection instead.

### Using the Postman collection

Import `docs/postman_collection.json`, then set these collection variables:
- `bearerToken`: a Cognito access or ID token, without the `Bearer ` prefix.
- `baseUrl`: defaults to `http://localhost:8080`.
- `assetId`, `userId`, `advisorId`: real ids, for the requests that need them.

Optional query parameters are included but disabled; tick them to use them.

## Configuration

Every setting has an environment variable. Defaults come from `src/main/resources/application.properties`.

### Database

| Variable | Default | Notes |
|---|---|---|
| `SPRING_DATASOURCE_URL` | `jdbc:postgresql://localhost:5433/ingestion` | |
| `DB_USERNAME` | `postgres` | `SPRING_DATASOURCE_USERNAME` also works (Spring relaxed binding); compose uses that one. |
| `DB_PASSWORD` | `root` | Likewise `SPRING_DATASOURCE_PASSWORD`. |

### AWS

| Variable | Default | Notes |
|---|---|---|
| `AWS_REGION` | `ap-southeast-1` | Region for S3, SQS and RDS Data API clients. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | – | Standard AWS SDK credential chain; on ECS the task role is used. |
| `S3_BUCKET` | `ingestion-audio` | Main upload bucket (has the Transcribe trigger). |
| `S3_BATCH_BUCKET` | blank | Batch bucket. Blank disables batch uploads. |
| `S3_PRESIGNED_URL_EXPIRY_MINUTES` | `15` | Upload URL lifetime. |
| `S3_PLAYBACK_URL_EXPIRY_MINUTES` | `60` | Playback URL lifetime. |
| `TRANSCRIBE_OUTPUT_BUCKET` | `ingestion-audio-output` | Where Transcribe writes `transcripts/<job>.json`. |
| `TRANSCRIBE_STATUS_QUEUE_URL` | a hard-coded queue URL | Transcribe status events. Blank disables the poller, and uploads stay `PENDING`. |
| `TRANSCRIPT_HANDOFF_QUEUE_URL` | blank | Handoff to AI analysis. Blank disables both the publisher and the AI poller. |

### Cognito and security

| Variable | Default | Notes |
|---|---|---|
| `AWS_COGNITO_REGION` | `ap-southeast-1` | |
| `AWS_COGNITO_USER_POOL_ID` | a hard-coded pool id | |
| `AWS_COGNITO_APP_CLIENT_ID` | two hard-coded client ids | Comma-separated. Tokens issued for any other client are rejected. |
| `AWS_COGNITO_ISSUER_URI` | built from region and pool id | JWKS is fetched lazily from `<issuer>/.well-known/jwks.json`. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated. Applies to `/api/**`. |
| `BOOTSTRAP_ADMIN_EMAILS` | two hard-coded emails | Comma-separated. These emails get SUPER_ADMIN on first sign-in only. |
| `DEFAULT_USER_PASSWORD` | `ChangeMe123!` | Set as the permanent Cognito password for admin-created users. Must meet the pool's password policy. |

### Uploads

| Variable | Default |
|---|---|
| `UPLOAD_MAX_FILES_PER_REQUEST` | `10` |
| `UPLOAD_MAX_FILE_SIZE_BYTES` | `104857600` (100 MB) |
| `UPLOAD_ALLOWED_CONTENT_TYPES` | `audio/mpeg,audio/wav,audio/x-wav,audio/mp4,audio/m4a` |
| `UPLOAD_ALLOWED_EXTENSIONS` | `mp3,wav,m4a` |

A file must pass both the content-type and the extension check.

### AI pipeline, checklist and dashboard

| Variable | Default | Notes |
|---|---|---|
| `AI_BASE_URL` | `http://localhost:2024` | fyp-ai LangGraph server. |
| `AI_READ_TIMEOUT_SECONDS` | `600` | `/runs/wait` blocks until the whole graph finishes. |
| `CHECKLIST_JSON` | `{}` | Default checklist as a JSON array of `{"id","question"}`. `{}` means fyp-ai uses its built-in checklist. |
| `DASHBOARD_OVERDUE_REVIEW_DAYS` | `3` | A call awaiting review longer than this counts as overdue. |
| `APP_DASHBOARD_SOURCE` | unset | `rds-data` reads the dashboard from Aurora via the Data API (local development only). |
| `APP_OBSERVABILITY_SOURCE` | unset | `rds-data` does the same for `/api/observability`. |
| `APP_OBSERVABILITY_RDS_CLUSTER_ARN`, `APP_OBSERVABILITY_RDS_SECRET_ARN`, `APP_OBSERVABILITY_RDS_DATABASE` | – / – / `ingestion` | Needed when either source is `rds-data`. See `../docs/production-observability.md`. |

## Security model

- Every endpoint except `GET /api/health` and `POST /api/users/first-admin` needs `Authorization: Bearer <Cognito JWT>`. Both access tokens and ID tokens are accepted (`CognitoClientIdValidator`).
- Cognito only proves who the caller is. **Roles come from `t_user.role`, not from the token** (`CognitoJwtAuthenticationConverter`). The Cognito group is kept in sync for display only.
- Roles: `SUPER_ADMIN`, `ADMIN`, `COMPLIANCE_OFFICER`, `MANAGER`, `DEVELOPER`. Only a SUPER_ADMIN can grant `ADMIN` or `SUPER_ADMIN`. Nobody can change their own role or status, and the last active ADMIN or SUPER_ADMIN cannot be demoted or disabled.
- A `DISABLED` user is rejected with 401 even if their token is still valid.
- Upload object keys are always built on the server from the caller's user id, so a client cannot choose where its file lands.
- Assets that belong to someone else return 404, not 403, so the response does not reveal that the id exists.

## Project layout

```
src/main/java/com/fyp/be/media_processing_service/
├── configs/            AWS clients, AI REST client, upload properties
│   └── security/       Spring Security, Cognito JWT validation, 401/403 handlers
├── controllers/        REST endpoints (see docs/API.md)
├── dtos/               Request/response records
├── exceptions/         Domain exceptions + GlobalExceptionHandler
├── models/             JPA entities and enums
├── repositories/       Spring Data repositories
└── services/           Business logic, SQS pollers, dashboard/metrics, Cognito admin
src/main/resources/
├── application.properties
└── db/migration/       Flyway migrations
```

## Deployment

The `Dockerfile` is a two-stage build (Maven on Corretto 25, then a JRE 25 Alpine image) that runs the jar on port 8080. The service runs on ECS behind infrastructure defined in `../infra`. Release steps are in `../docs/developer-release.md`.
