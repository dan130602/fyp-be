# fyp-be

Backend for **ClearCall**, an AI-powered regulatory compliance system for monitoring advisor calls. Recordings are uploaded, transcribed with Amazon Transcribe, checked against a compliance checklist by the AI pipeline (the separate `fyp-ai` repository), and then reviewed by compliance officers.

## Repository layout

| Path | What it is |
|---|---|
| [`media-processing-service/`](media-processing-service/README.md) | Spring Boot API: uploads, processing status, AI results, officer reviews, users and roles, management dashboard. **Start here for the backend.** |
| `lambda/transcribe-trigger/` | Python Lambdas that start Amazon Transcribe jobs: on S3 upload (`handler.py`), and for the nightly batch path (`enqueuer.py`, `batch_worker.py`). |
| `infra/` | Terraform for AWS (VPC, ALB, CloudFront, ECS, ECR, RDS, S3, SQS, EventBridge, Cognito, Lambda, Secrets Manager, batch processing). |
| [`evaluation-engine/`](evaluation-engine/README.md) | Measures the production speech-to-text step against ground truth. It runs the real Lambda handler unchanged. |
| [`evaluation-dataset/`](evaluation-dataset/README.md) | Evaluation dataset: synthetic advisory calls with ground-truth transcripts and checklist outcomes. |
| `docs/` | Operational notes: [developer release](docs/developer-release.md), [production observability](docs/production-observability.md). |
| `local-dev/` | Local-only helpers, e.g. `seed-dashboard.sql` (fake calls for the officer dashboard). |
| `docker-compose.yml` | Local PostgreSQL plus the backend container. |
| `AGENTS.md`, `STT_HANDOVER.md` | Rules and handover guide for speech-to-text improvement work. |

## How it fits together

```
Frontend ──REST (Cognito JWT)──▶ media-processing-service ──▶ PostgreSQL / Aurora
   │                                   ▲          │
   │ presigned S3 PUT                  │ SQS      │ SQS handoff
   ▼                                   │          ▼
S3 (audio) ──S3 event──▶ transcribe-trigger Lambda ──▶ Amazon Transcribe     fyp-ai (LangGraph compliance graph)
                                       │
                         EventBridge "job state change" ──▶ SQS ──▶ media-processing-service
```

1. The frontend gets a presigned URL from the backend and uploads the audio straight to S3.
2. The S3 event starts a Transcribe job in the Lambda. Batch uploads go to a separate bucket and are processed nightly instead.
3. When the job finishes, EventBridge puts a status event on SQS. The backend picks it up, marks the upload complete and hands the transcript to `fyp-ai`.
4. `fyp-ai` returns a verdict for each checklist item. Officers review those verdicts and can override them in the frontend. Managers see the totals on the dashboard.

The full flow, configuration and security model are described in [`media-processing-service/README.md`](media-processing-service/README.md).

## Quick start (backend, local)

```bash
docker compose up -d postgres                 # PostgreSQL on localhost:5433
cd media-processing-service
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5433/ingestion \
  API_DOCS_ENABLED=true ./mvnw spring-boot:run  # API on http://localhost:8080
```

Then open http://localhost:8080/swagger-ui.html. You need JDK 25 and Docker. Uploads and user management also need AWS credentials. See the service README for every environment variable.

To fill the dashboard with fake data locally:

```bash
docker exec -i ingestion-db psql -U postgres -d ingestion -v ON_ERROR_STOP=1 < local-dev/seed-dashboard.sql
```

## API documentation

All of it is in `media-processing-service/docs/`:

- `api.html`: the full endpoint reference. Open it in a browser; it works offline.
- `API.md` / `API.pdf`: workflow, ownership rules and dashboard definitions.
- `openapi.json`: the OpenAPI 3.1 spec, generated from the code.
- `postman_collection.json`: a ready-to-import Postman collection.

## Deployment

Infrastructure is managed with Terraform in `infra/`. The backend ships as a Docker image to ECR and runs on ECS. Release steps are in [`docs/developer-release.md`](docs/developer-release.md).
