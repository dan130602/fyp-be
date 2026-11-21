# media-processing-service API

Base URL: `http://localhost:8080` locally. Every path below starts with `/api`.

This document explains behaviour: workflow, ownership rules and dashboard definitions. The exact request and response contract is in [openapi.json](openapi.json), generated from the code. A ready-to-import Postman collection is in [postman_collection.json](postman_collection.json). If this page and the spec ever disagree on a field, trust the spec.

- [Conventions](#conventions)
- [Roles at a glance](#roles-at-a-glance)
- [Health](#health)
- [Users](#users)
- [Advisors](#advisors)
- [Checklists](#checklists)
- [Uploads](#uploads)
- [Media assets](#media-assets)
- [Dashboard](#dashboard)
- [Observability](#observability)
- [Enums](#enums)

## Conventions

**Authentication.** Send a Cognito JWT, either an access token or an ID token, as `Authorization: Bearer <token>`. The token must be issued by the configured user pool to one of the configured app clients. Only `GET /api/health` and `POST /api/users/first-admin` work without a token.

**Authorization.** Your role comes from the service's own `t_user` table, not from the token. The first time a token reaches the service, its owner is created as a `COMPLIANCE_OFFICER` (or `SUPER_ADMIN` for emails in `BOOTSTRAP_ADMIN_EMAILS`). Disabled users get 401.

**JSON.** Field names are camelCase. The one exception is the `processing_mode` request field on the upload endpoint. Timestamps are ISO-8601 with an offset (`2026-11-21T14:03:12.481+08:00`). Dates in query strings are `YYYY-MM-DD`.

**Errors.** Errors raised by the service use this body:

```json
{
  "timestamp": "2026-11-21T14:03:12.481+08:00",
  "status": 400,
  "error": "Bad Request",
  "message": "Request validation failed",
  "details": ["files[0].advisorId: must not be null"]
}
```

`details` is filled only for bean-validation failures; otherwise it is `[]`.

| Status | When |
|---|---|
| 400 | Bean validation failed, a file was rejected, or a review is invalid |
| 401 | Missing, invalid or expired token, or the account is disabled |
| 403 | Your role is not allowed, you tried to grant a role you can't, or setup is already done |
| 404 | The resource doesn't exist, **or exists but isn't yours** (see each endpoint) |
| 409 | Analysis not ready, user already exists, or a self-modification / last-admin rule was broken |
| 500 | Presigning or a Cognito admin call failed |

Some requests fail before they reach the service code, for example a malformed JSON body, an unknown enum value or a non-UUID path id. Those come back as a 4xx from Spring's default error handling, not in the body format above.

## Roles at a glance

SA = SUPER_ADMIN, A = ADMIN, CO = COMPLIANCE_OFFICER, M = MANAGER, D = DEVELOPER.

| Method & path | SA | A | CO | M | D | Notes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `GET /health` | public | | | | | |
| `GET /users/me` | ✓ | ✓ | ✓ | ✓ | ✓ | |
| `GET /users` | ✓ | ✓ | | | | |
| `POST /users/first-admin` | public | | | | | Only works while no users exist |
| `POST /users` | ✓ | ✓ | | | | Only SA can create A/SA |
| `PATCH /users/{id}/role` | ✓ | ✓ | | | | Only SA can grant A/SA |
| `PATCH /users/{id}/status` | ✓ | ✓ | | | | |
| `GET /advisors` | ✓ | ✓ | ✓ | ✓ | | |
| `GET /checklists/default` | ✓ | ✓ | ✓ | | | |
| `POST /uploads/presigned-urls` | ✓ | ✓ | ✓ | | | |
| `GET /uploads/{id}` | ✓ | ✓ | ✓ | | | Owner, or any A/SA |
| `GET /media-assets` | ✓ | ✓ | ✓ | | | Your own uploads only |
| `GET /media-assets/{id}/playback-url` | ✓ | ✓ | ✓ | | | Owner only |
| `GET /media-assets/{id}/analysis` | ✓ | ✓ | ✓ | ✓ | | Any asset |
| `PUT /media-assets/{id}/review` | ✓ | ✓ | ✓ | | | Owner only |
| `GET /dashboard/*` | ✓ | ✓ | ✓ | ✓ | | Org-wide |
| `GET /observability` | ✓ | ✓ | | ✓ | ✓ | |

---

## Health

### `GET /api/health`

Public. Returns `200` with the plain-text body `ok`.

---

## Users

### `UserResponse`

```json
{
  "id": "7d3c1f0e-1b1a-4c3e-9f55-0a2b6a8e4d21",
  "email": "officer@example.com",
  "name": "Jane Tan",
  "role": "COMPLIANCE_OFFICER",
  "status": "ACTIVE",
  "createdAt": "2026-11-01T09:15:00+08:00"
}
```

### `GET /api/users/me`

Returns the caller's `UserResponse`. If this is the caller's first request, the user is created first.

### `GET /api/users`

**Roles:** SA, A. Returns all users as `UserResponse[]`, newest first.

### `POST /api/users/first-admin`

**Public.** Creates the first account as a **SUPER_ADMIN**, in both Cognito and the database. Returns **403** `Setup already completed` as soon as any user exists.

```json
{ "email": "admin@example.com", "name": "Admin" }
```

| Field | Rules |
|---|---|
| `email` | required, valid email |
| `name` | optional |

`201 Created` with a `UserResponse`.

### `POST /api/users`

**Roles:** SA, A. Creates a Cognito login and a matching user row. The Cognito password is set permanently to `DEFAULT_USER_PASSWORD` and no invite email is sent.

```json
{ "email": "new.officer@example.com", "name": "New Officer", "role": "COMPLIANCE_OFFICER" }
```

| Field | Rules |
|---|---|
| `email` | required, valid email |
| `name` | optional |
| `role` | required, a [Role](#enums) |

`201 Created` with a `UserResponse`.

Errors: `403` if an ADMIN tries to create an ADMIN or SUPER_ADMIN. `409` if the user already exists. `500` if Cognito fails.

### `PATCH /api/users/{userId}/role`

**Roles:** SA, A.

```json
{ "role": "MANAGER" }
```

`200` with the updated `UserResponse`. The user is also moved to the matching Cognito group.

Errors:
- `404` if the user is not found.
- `403` if a non-SUPER_ADMIN tries to grant `ADMIN` or `SUPER_ADMIN`.
- `409` `You cannot change your own role`.
- `409` if the change would leave no active ADMIN, or no active SUPER_ADMIN.

### `PATCH /api/users/{userId}/status`

**Roles:** SA, A.

```json
{ "status": "DISABLED" }
```

`200` with the updated `UserResponse`. The Cognito login is disabled or enabled before the database is updated.

Errors:
- `404` if the user is not found.
- `409` `You cannot change your own status`.
- `409` if this would disable the last active ADMIN or SUPER_ADMIN.

---

## Advisors

### `GET /api/advisors`

**Roles:** SA, A, CO, M. Returns every **active** advisor. The list is meant to be searched client-side in the upload dialog.

```json
[
  {
    "id": "3f9a2c4e-6d0b-4f7a-9a51-2c1e8b7d6f30",
    "employeeId": "ADV-0012",
    "fullName": "Marcus Lee",
    "email": "marcus.lee@example.com",
    "department": "Wealth Advisory"
  }
]
```

---

## Checklists

### `GET /api/checklists/default`

**Roles:** SA, A, CO. Returns the deployment's default checklist, taken from `CHECKLIST_JSON`. Returns `[]` when `CHECKLIST_JSON` is unset or not a JSON array, in which case fyp-ai uses its built-in checklist.

```json
[
  { "id": "C1", "question": "Did the advisor disclose the product's risks?" }
]
```

---

## Uploads

Uploading is a two-step flow:

1. `POST /api/uploads/presigned-urls` to get one presigned S3 `PUT` URL per file.
2. `PUT` the raw file bytes to each `presignedUrl`, with the same `Content-Type` you declared. The URL expires after `expiresIn` seconds (15 minutes by default).

Then poll `GET /api/uploads/{id}` or `GET /api/media-assets` to follow progress.

### `POST /api/uploads/presigned-urls`

**Roles:** SA, A, CO.

```json
{
  "processing_mode": "immediate",
  "files": [
    {
      "fileName": "call-2026-11-20.mp3",
      "contentType": "audio/mpeg",
      "fileSize": 4823011,
      "advisorId": "3f9a2c4e-6d0b-4f7a-9a51-2c1e8b7d6f30",
      "checklist": [
        { "id": "C1", "question": "Did the advisor disclose the product's risks?" }
      ]
    }
  ]
}
```

| Field | Rules |
|---|---|
| `processing_mode` | `"immediate"` (default if omitted) or `"batch"`. Note the snake_case name. |
| `files` | required, 1 to `UPLOAD_MAX_FILES_PER_REQUEST` (default 10) items |
| `files[].fileName` | required. The extension must be in `UPLOAD_ALLOWED_EXTENSIONS` (default `mp3, wav, m4a`). |
| `files[].contentType` | required, must be in `UPLOAD_ALLOWED_CONTENT_TYPES` |
| `files[].fileSize` | required in practice: must be > 0 and ≤ `UPLOAD_MAX_FILE_SIZE_BYTES` (default 100 MB) |
| `files[].advisorId` | required, must be an **active** advisor |
| `files[].checklist` | optional list of `{id, question}`, both non-blank. Omit it or send `[]` to use the default checklist. |

The whole request is validated before anything is written. If one file fails, no URLs are issued.

`200 OK`:

```json
{
  "uploads": [
    {
      "id": "b1e7c9d2-0f4a-4b8e-8c3d-5a6f7e8d9c01",
      "fileName": "call-2026-11-20.mp3",
      "objectKey": "uploads/7d3c1f0e-.../2a4b...-call-2026-11-20.mp3",
      "presignedUrl": "https://ingestion-audio.s3.ap-southeast-1.amazonaws.com/uploads/...",
      "expiresIn": 900
    }
  ]
}
```

`id` is the media asset id that every other endpoint uses. The object key is chosen by the server. Batch uploads use `{assetId}/{filename}` in the batch bucket.

Errors:
- `400` if the content type, extension, size or file count is not allowed, the advisor is unknown or inactive, or batch mode is requested but no batch bucket is configured.
- `500` `Failed to generate upload URL`.

### `GET /api/uploads/{id}`

**Roles:** SA, A, CO. The uploader, or any ADMIN/SUPER_ADMIN, can read it. Anyone else gets `404`.

```json
{
  "id": "b1e7c9d2-0f4a-4b8e-8c3d-5a6f7e8d9c01",
  "fileName": "call-2026-11-20.mp3",
  "objectKey": "uploads/7d3c1f0e-.../2a4b...-call-2026-11-20.mp3",
  "status": "COMPLETED",
  "processingMode": "immediate",
  "transcriptUri": "s3://ingestion-audio-output/transcripts/transcribe-7d3c...-9f1e2d3c4b5a6978.json",
  "failureReason": null,
  "updatedAt": "2026-11-20T10:42:31+08:00"
}
```

`status` is a [MediaAssetStatus](#enums). It describes **transcription** only. Use the analysis endpoint to see whether the AI step has finished.

---

## Media assets

### `GET /api/media-assets`

**Roles:** SA, A, CO. Returns **the caller's own** uploads, newest first. Admins also see only their own uploads here.

```json
[
  {
    "id": "b1e7c9d2-0f4a-4b8e-8c3d-5a6f7e8d9c01",
    "originalFilename": "call-2026-11-20.mp3",
    "contentType": "audio/mpeg",
    "fileSizeBytes": 4823011,
    "status": "COMPLETED",
    "processingMode": "immediate",
    "createdAt": "2026-11-20T10:31:02+08:00",
    "advisor": {
      "id": "3f9a2c4e-...",
      "employeeId": "ADV-0012",
      "fullName": "Marcus Lee",
      "email": "marcus.lee@example.com",
      "department": "Wealth Advisory"
    }
  }
]
```

`advisor` is `null` for uploads made before advisor selection existed.

### `GET /api/media-assets/{id}/playback-url`

**Roles:** SA, A, CO. **Uploader only**; anyone else gets `404`.

```json
{ "url": "https://ingestion-audio.s3...amazonaws.com/...", "expiresIn": 3600 }
```

The URL is a presigned S3 `GET` on the main upload bucket.

### `GET /api/media-assets/{id}/analysis`

**Roles:** SA, A, CO, M. Works for **any** asset, because the org-wide dashboard links here.

```json
{
  "mediaAssetId": "b1e7c9d2-0f4a-4b8e-8c3d-5a6f7e8d9c01",
  "fileName": "call-2026-11-20.mp3",
  "assetStatus": "COMPLETED",
  "analysisStatus": "READY",
  "failureReason": null,
  "result": { "final_results": [ { "item_id": "C1", "status": "FAIL", "...": "..." } ], "...": "..." },
  "review": {
    "reviewedBy": "7d3c1f0e-...",
    "reviewedAt": "2026-11-21T09:00:00+08:00",
    "overrides": [
      { "checklistItemId": "C1", "aiVerdict": "FAIL", "decision": "PASS", "comment": "Risk disclosed at 04:12." }
    ],
    "history": [
      {
        "checklistItemId": "C1",
        "aiVerdict": "FAIL",
        "decision": "PASS",
        "comment": "Risk disclosed at 04:12.",
        "decidedBy": "7d3c1f0e-...",
        "decidedAt": "2026-11-21T09:00:00+08:00"
      }
    ]
  }
}
```

| `analysisStatus` | Meaning | `result` / `review` |
|---|---|---|
| `PENDING` | Transcription or AI analysis is still running | `null` |
| `READY` | AI output is stored | `result` is fyp-ai's raw output; `review` is `null` until someone has reviewed |
| `FAILED` | Transcription failed, or the AI output is an error or unreadable | `null`; `failureReason` explains |

`result` is passed through from fyp-ai unchanged. `result.final_results[]` holds one `{item_id, status, ...}` per checklist item, and `status` uses the [verdict vocabulary](#enums).

In `review`, `overrides` is the current state. `history` is every decision ever logged, oldest first. A history entry with `decision: null` means that item was reverted to the AI verdict.

Errors: `404` if the asset is not found.

### `PUT /api/media-assets/{id}/review`

**Roles:** SA, A, CO. **Uploader only**; anyone else gets `404`.

Submits the reviewer's view of the AI verdicts. Send **only the items you disagree with**. An empty list accepts every AI verdict as-is.

```json
{
  "overrides": [
    { "checklistItemId": "C1", "decision": "PASS", "comment": "Risk disclosed at 04:12." }
  ]
}
```

| Field | Rules |
|---|---|
| `overrides` | required (may be `[]`) |
| `overrides[].checklistItemId` | required; must be an `item_id` in the AI's `final_results`; no duplicates |
| `overrides[].decision` | required; one of `PASS`, `FAIL`, `PARTIAL`, `CANNOT_DETERMINE`, `NOT_APPLICABLE`; **must differ from the AI verdict** |
| `overrides[].comment` | required, non-blank |

Each submission replaces the previous one:
- New or changed overrides are added to the decision log.
- Unchanged overrides are not logged again.
- An earlier override that is missing from this submission is logged as a revert to the AI verdict.

The checklist is then marked `REVIEWED`. You can submit again to change the review.

`200 OK` returns a `review` object in the same shape as in the analysis response.

Errors:
- `400` for an unknown item, a duplicate item, an invalid decision, or a decision equal to the AI verdict.
- `404` if the asset is not found or not yours.
- `409` `AI analysis is not ready for review` if there is no AI output yet, or the AI output is an error.

---

## Dashboard

**Roles (all endpoints):** SA, A, CO, M. These figures cover **every** user's uploads, not only the caller's.

Common query parameters:

| Param | Type | Notes |
|---|---|---|
| `from` | `YYYY-MM-DD`, optional | Inclusive; filters on upload time. Omitted means no lower bound. |
| `to` | `YYYY-MM-DD`, optional | Inclusive. Omitted means no upper bound. |
| `department` | string, optional | Exact match on the advisor's department. Blank means all departments. |

The rules below apply to every dashboard endpoint.

**Review stage of a call:**
- `FAILED`: transcription failed, or the AI output is unreadable or an error.
- `PROCESSING`: no AI output yet.
- `AWAITING_REVIEW`: AI output exists but the checklist is not reviewed.
- `REVIEWED`: the checklist has been reviewed.

**Effective verdict of an item:** the reviewer's current override if there is one, otherwise the AI verdict.

**Outcome:** reviewed calls only. Checked in this order:
1. `VIOLATION` if any item is `FAIL` or `PARTIAL`.
2. `INCONCLUSIVE` if any item is `CANNOT_DETERMINE`, or there are no items.
3. Otherwise `COMPLIANT`.

**Percentages** run from 0 to 100, rounded to 1 decimal place, and are `0.0` when the denominator is 0.

### `GET /api/dashboard/compliance-overview`

Covers reviewed calls only. The three counts add up to `reviewedCalls`.

```json
{
  "reviewedCalls": 40, "compliantCalls": 28, "violationCalls": 9, "inconclusiveCalls": 3,
  "compliancePct": 70.0, "violationPct": 22.5, "inconclusivePct": 7.5
}
```

### `GET /api/dashboard/review-status`

```json
{
  "totalCalls": 55, "processing": 4, "awaitingReview": 9, "reviewed": 40, "failed": 2,
  "reviewCoveragePct": 72.7
}
```

### `GET /api/dashboard/outstanding-items`

```json
{
  "awaitingReview": 9,
  "oldestAwaitingReviewAt": "2026-11-14T16:20:00+08:00",
  "overdueReviews": 3,
  "overdueAfterDays": 3,
  "unreviewedEscalations": 2,
  "unreviewedHolds": 1,
  "reviewedWithUndeterminedItems": 5,
  "proposedActionsByOwner": { "advisor": 7, "compliance": 3 }
}
```

- `overdueReviews` counts calls whose AI analysis finished more than `overdueAfterDays` days ago (`DASHBOARD_OVERDUE_REVIEW_DAYS`) and that are still unreviewed.
- `unreviewedEscalations` and `unreviewedHolds` come from the AI summary's `escalation.required` and `escalation.hold_transaction`.
- `proposedActionsByOwner` totals the AI's `summary.actions_by_owner` across the calls in range. Whether an action was completed is not tracked.

### `GET /api/dashboard/checklist-completion`

Covers items on reviewed calls only. `applicableItems` leaves out `NOT_APPLICABLE` items. `determinedItems` counts those that are not `CANNOT_DETERMINE`.

```json
{ "determinedItems": 312, "applicableItems": 340, "completionPct": 91.8 }
```

### `GET /api/dashboard/advisor-compliance`

Lists advisors with at least one call in range, highest `violationPct` first, then by name. `from` and `to` echo the request (`null` if omitted).

```json
{
  "from": "2026-11-01",
  "to": "2026-11-21",
  "advisors": [
    {
      "advisorId": "3f9a2c4e-...", "employeeId": "ADV-0012", "fullName": "Marcus Lee",
      "department": "Wealth Advisory", "status": "ACTIVE",
      "totalCalls": 12, "reviewedCalls": 10, "compliantCalls": 6, "violationCalls": 3,
      "inconclusiveCalls": 1, "redFlagCalls": 2,
      "compliancePct": 60.0, "violationPct": 30.0, "inconclusivePct": 10.0, "reviewCoveragePct": 83.3
    }
  ]
}
```

`redFlagCalls` counts calls where the AI reported at least one red flag (`final_red_flags`).

### `GET /api/dashboard/calls`

A paginated list of calls, newest first.

| Extra param | Default | Notes |
|---|---|---|
| `advisorId` | – | UUID filter |
| `outcome` | – | `COMPLIANT`, `VIOLATION` or `INCONCLUSIVE`. Only reviewed calls have an outcome, so this filter returns only reviewed calls. |
| `page` | `0` | Zero-based; negative values are treated as 0 |
| `size` | `20` | Clamped to 1–100 |

```json
{
  "calls": [
    {
      "mediaAssetId": "b1e7c9d2-...",
      "fileName": "call-2026-11-20.mp3",
      "createdAt": "2026-11-20T10:31:02+08:00",
      "advisor": { "id": "3f9a2c4e-...", "employeeId": "ADV-0012", "fullName": "Marcus Lee", "email": "marcus.lee@example.com", "department": "Wealth Advisory" },
      "reviewStage": "REVIEWED",
      "outcome": "VIOLATION",
      "violatedItems": [ { "itemId": "C3", "question": "Did the advisor confirm suitability?", "verdict": "FAIL" } ],
      "redFlags": 1
    }
  ],
  "page": 0,
  "size": 20,
  "totalCalls": 37
}
```

- `outcome` is `null` until the call is reviewed.
- `advisor` is `null` for uploads made before advisor selection existed.
- `violatedItems` lists items whose effective verdict is `FAIL` or `PARTIAL`.

---

## Observability

### `GET /api/observability?days=7`

**Roles:** SA, A, M, D. Production AI-quality metrics for the developer dashboard. The officers' review decisions are used as the reference answer. They are **not** independent ground truth.

| Param | Default | Rules |
|---|---|---|
| `days` | `7` | 1–90, otherwise `400` |

The response is a JSON object (built as a map in `services/ProductionMetrics.java`, not a typed DTO) with these top-level keys:

| Key | Contents |
|---|---|
| `source`, `generatedAt`, `windowDays` | Metadata (`source` is always `"production"`) |
| `reference`, `population` | Plain-English description of what the numbers measure |
| `counts` | `calls`, `savedResults`, `failedResults`, `checks`, `binaryDecisions`, `unclear`, `predictionFailures`, `notApplicable`, `reviewedCalls`, `reviewedChecks`, `unreviewedChecks`, `staleReviewItems` |
| `metrics` | Ratios, each `{numerator, denominator, percentage}` (`percentage` is `null` when the denominator is 0): `decisionCoverage`, `predictionFailureRate`, `abstentionRate`, `reviewedBreachRecall`, `reviewedBreachPrecision`, `reviewAgreement`, `reviewedBreachDetection`, `reviewedFalseClearanceRate` |
| `matrix` | AI vs officer confusion matrix: `agreedPass`, `aiPassOfficerFail`, `aiFailOfficerPass`, `agreedBreach`, `binaryPairs` |
| `reviewedBreaches` | `total`, `detected`, `unclear`, `predictionFailures`, `notApplicable`, `falseClearances` |
| `cost` | `reportedUsd`, `pricedResults`, `unknownResults`, `scope` |
| `latency` | `p50Ms`, `p95Ms`, `measuredResults`, `unknownResults`, `scope` |
| `overrides` | `activeItems`, `reviewedItems`, `rate`, `eventCount`, `highPriorityEvents`, and `recent` (the 20 most recent verdict changes: `id`, `mediaAssetId`, `checklistItemId`, `aiVerdict`, `from`, `to`, `reverted`, `decidedBy`, `decidedAt`, `priority`; `priority` is `"high"` when the AI said PASS and the officer said FAIL) |
| `stages` | `[{stage, p50Ms, p95Ms, count}]` |
| `models` | `[{model, inputTokens, outputTokens, cacheReadTokens}]` |
| `providerErrors`, `legacyResults` | Counts |

See `../../docs/production-observability.md` for how to run it against Aurora.

---

## Enums

| Enum | Values |
|---|---|
| Role | `SUPER_ADMIN`, `ADMIN`, `COMPLIANCE_OFFICER`, `MANAGER`, `DEVELOPER` |
| UserStatus | `ACTIVE`, `DISABLED` |
| MediaAssetStatus | `PENDING`, `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED` (`QUEUED`/`PROCESSING` occur in batch mode only) |
| ProcessingMode (JSON) | `immediate`, `batch` |
| AnalysisStatus | `PENDING`, `READY`, `FAILED` |
| ReviewStage | `PROCESSING`, `AWAITING_REVIEW`, `REVIEWED`, `FAILED` |
| CallOutcome | `COMPLIANT`, `VIOLATION`, `INCONCLUSIVE` |
| Verdict (AI and reviewer) | `PASS`, `FAIL`, `PARTIAL`, `CANNOT_DETERMINE`, `NOT_APPLICABLE` |
