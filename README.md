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
| [`stt-service/meralion/`](stt-service/meralion/README.md) | **The dockerised speech-to-text worker**: a Dockerfile that builds a GPU image (pinned vLLM stack, hash-verified model baked in, S3 input/output and resumable state) around MERaLiON-3-3B-ASR with the long-audio pipeline. Runs our fine-tuned model or the original. See [Production](#from-this-repository-to-production). |
| [`stt-experiments/meralion-sft/`](stt-experiments/meralion-sft/README.md) | Code that trained, exported and evaluated the fine-tuned MERaLiON model (decoder LoRA + encoder LoRA + full speech adapter). |
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

## Fine-tuned speech-to-text model (MERaLiON)

Besides Amazon Transcribe, this repository contains a fine-tuned version of [`MERaLiON-3-3B-ASR`](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) for advisory-call speech:
decoder LoRA + encoder LoRA + a fully trained speech adapter, trained on 140 synthetic development calls (English, Singlish, Mandarin and English-Mandarin code-switching).
On the 75 held-out calls (same calls for every system, scored with the same rules) it makes far fewer errors than Amazon Transcribe:

| Held-out 75 calls | Noisy / channel-degraded: WER, entity WER | Clean: WER, entity WER |
|---|---:|---:|
| Amazon Transcribe | 3.57%, 6.24% | 1.95%, 4.31% |
| MERaLiON + long-audio pipeline, no fine-tuning | 1.80%, 4.01% | 1.01%, 3.59% |
| **Fine-tuned MERaLiON (this model)** | **1.18%, 2.37%** | **0.47%, 1.68%** |

The data is synthetic text-to-speech, and the fine-tuned model is worse than the original at some amounts and quantities (digits dropped in comma-grouped English numbers), so treat it as a
research checkpoint; the full table, training data, known weaknesses and licence are in the model card:
**[huggingface.co/tanedric/meralion-3-3b-asr-clearcall-j5xe](https://huggingface.co/tanedric/meralion-3-3b-asr-clearcall-j5xe)**.

**Get OUR fine-tuned weights (6.6 GB) from Hugging Face and verify them with one command.** The weights are published by this project at
[`tanedric/meralion-3-3b-asr-clearcall-j5xe`](https://huggingface.co/tanedric/meralion-3-3b-asr-clearcall-j5xe), not downloaded from the original MERaLiON repository (the original model is not
needed). On the GPU machine, in a checkout of the `stt/meralion-release` branch, after setting up the environment once
(`pip install --no-deps -r requirements-vllm.lock` and `pip install --no-deps meralion-3-asr==0.1.0`, Linux x86-64 and CPython 3.12; see [`stt-service/meralion`](stt-service/meralion/README.md)):

```bash
pip install -U huggingface_hub && python stt-service/meralion/tools/get_finetuned_model.py
```

It downloads the model to `stt-service/meralion/models/clearcall-j5xe/`, checks all 14 files against the manifest committed here, and prints the command that transcribes your calls:

```bash
cd stt-service/meralion
python -m meralion_worker run --invocation-id my-first-run --jobs jobs.json \
    --model-dir models/clearcall-j5xe/model --config configs/runtime/j5xe-ep2-vllm-rp105-sched0.json \
    --output-dir out --state-dir state --summary-output summary.json
```

Download only, without a checkout: `hf download tanedric/meralion-3-3b-asr-clearcall-j5xe --local-dir clearcall-j5xe`.

The base model's licence (MERaLiON Public Licence, including the Gemma terms) applies to the weights; the development of this work was assisted by MERaLiON, an AI model developed by the Agency for
Science, Technology and Research (A\*STAR). Training and exporting code, and how to reproduce the checkpoint, are in
[`stt-experiments/meralion-sft/`](stt-experiments/meralion-sft/README.md). Speaker attribution is a separate component and is not part of the worker or its Docker image.

### From this repository to production

The worker is a **batch job in a container**, not a service with an API: it reads audio, writes one transcript JSON per call, and exits. The steps, in order:

1. **Build the image with our model.** Download the weights as above, copy `models/clearcall-j5xe/model` to `stt-service/meralion/model`, and build with
   `--build-arg BACKEND=vllm --build-arg RUNTIME_CONFIG=j5xe-ep2-vllm-rp105-sched0.json`. The build and every start check all 14 model files against the committed manifest, so a wrong or
   corrupt model never runs. Exact commands: [`stt-service/meralion`](stt-service/meralion/README.md#docker).
2. **Push to ECR and run it as a GPU batch job** (AWS Batch on EC2, `ap-southeast-1`). In S3 mode the worker takes `--input s3://...`, `--output-bucket` and `--state-uri`, writes
   `transcripts/<job>.json`, and keeps resume state in S3, so an interrupted job continues where it stopped. Registry, IAM, compute-environment and job-definition details are in
   [`stt-service/meralion/tools/ec2/ECR_PUSH_AND_BATCH_SMOKE.md`](stt-service/meralion/tools/ec2/ECR_PUSH_AND_BATCH_SMOKE.md).
3. **Choose the GPU for the configuration.** The fine-tuned configuration uses BF16, which needs a GPU with compute capability 8.0 or higher (for example L4, A10G, L40S, A100). A T4 (`g4dn`)
   needs the FP16 profile; one exists for the original model but not for the fine-tuned one. Measured on the development calls (17.4 hours of audio): about 10 minutes on an L40S and 17 minutes on a
   Blackwell slice. Each start hashes the 6.6 GB model first (a few minutes), so keep job timeouts generous.
4. **Connect it to the product.** The backend and the Lambda still use Amazon Transcribe; this worker is a standalone producer of transcript JSON. Wiring it in means triggering a batch job per upload
   and reading `out/transcripts/<job>.json` (plain text at `results.transcripts[0].transcript`) back into the existing flow. The output has no timestamps or speaker labels, so speaker attribution
   still has to come from a separate component.
5. **Guard what matters.** The fine-tuned model is worse than the original at some amounts and quantities (see the model card). For compliance use, check extracted amounts and quantities against a
   second source (for example the Amazon Transcribe transcript) or route them to review.

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
