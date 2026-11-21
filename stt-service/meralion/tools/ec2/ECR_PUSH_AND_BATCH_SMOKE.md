# ECR push and the first GPU-backed S3 smoke test (prepared, NOT executed)

Nothing in this file has been run against AWS. Placeholders in `<angle brackets>` must be supplied by the person who runs the commands. No credential belongs in any file, command history or chat.

## 1. The image that was built (facts measured on the build host)

| | |
|---|---|
| Local tag | `meralion-stt-worker:vllm-sched0-fp16-082c274` (image id `1a38ad6f03ba`) |
| Source commit | `082c2746219f8e5cbc41f75f9b09b68994472348` (branch `stt/meralion-worker`, clean tree) |
| Selected explicitly | `BACKEND=vllm`, `RUNTIME_CONFIG=fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0-fp16.json` (profile `vllm-rp105-sched0-fp16-v1`) |
| Platform | linux/amd64, Python 3.12.3, CUDA 12.8 devel tools (nvcc 12.8.61), base pinned by digest |
| Size | content size 15.9 GB as Docker reports it; Docker Desktop's image list shows 43.1 GB (it counts unpacked layers). Largest layers: dependency install about 11 GB, checkpoint 6.65 GB. The compressed size ECR will show is not known until the push. |
| Build checks that passed | argument check, `pip check`, wheel sha256 against the package manifest, `verify-environment`, `check-model` (21 files) |
| Entry point | `python -m meralion_worker`, working directory `/opt/meralion` (the worker package is imported from the working directory: do not override `workingDirectory` or the entry point in the job definition) |
| Runtime user | `worker` (uid 10001); no inbound port; writable only `/var/meralion/{out,state}`, `/var/cache/meralion/*`, `/home/worker`, `/tmp` |
| Effective configuration | strategy v2, formatter `percentages-v2+amount-v1`, config fingerprint `2a250efc...2d17d13`, FP16 constructor override, expected effective engine dtype `torch.float16` |

CPU container checks that passed (no GPU involved): `--version`, `--help`, `verify-environment`, `check-model` at container start, `pip check`, installed packages identical to the lock (194 pins plus the verified wheel), torch 2.9.1+cu128, vLLM 0.16.0, the MERaLiON vLLM plugin registered, vLLM `cuda.py` equal to the manifest's ORIGINAL hash, non-root user, read-only root filesystem, usage errors exit 3 with a clear message.

What those checks do NOT establish: any GPU execution; S3 authentication or permissions (the one S3-mode attempt stopped in decoder construction because the build host has no GPU, so it exercised nothing S3-related); the first-run FlashInfer kernel build on a T4; FP16 output quality on a T4; memory on a T4. On a host without a visible GPU the worker stops with `RuntimeError: Device string must not be empty` (exit 1): if that appears in Batch, the job did not get a GPU.

Startup cost to expect: `check-model` hashes the 6.2 GB checkpoint at every start (about 2 minutes during the build, 7.4 minutes on a cold Docker Desktop disk). The EC2 figure is unknown; leave the job timeout generous.

## 2. Push to ECR (Singapore only)

```bash
export AWS_REGION=ap-southeast-1
export ACCOUNT_ID=<12-digit account id>             # check it: aws sts get-caller-identity --query Account --output text
export REPO=meralion-worker                        # NOT the Transcribe Lambda's repository
export TAG=vllm-sched0-fp16-082c274
export LOCAL=meralion-stt-worker:$TAG
export REGISTRY=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

aws sts get-caller-identity --region $AWS_REGION                                    # confirm account and identity first
aws ecr describe-repositories --repository-names $REPO --region $AWS_REGION 2>/dev/null \
  || aws ecr create-repository --repository-name $REPO --region $AWS_REGION \
       --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true \
       --tags Key=Project,Value=meralion-stt-test
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY
docker tag $LOCAL $REGISTRY/$REPO:$TAG
docker push $REGISTRY/$REPO:$TAG                                                    # about 16 GB of layers: expect a long upload
aws ecr describe-images --repository-name $REPO --image-ids imageTag=$TAG --region $AWS_REGION \
  --query 'imageDetails[0].[imageDigest,imageSizeInBytes,imagePushedAt]' --output text
```

Use the DIGEST in the job definition (`$REGISTRY/$REPO@sha256:<digest>`), not the tag, so the test runs exactly this image. Record the digest, the compressed size and the push time next to the commit.

## 3. AWS prerequisites for one GPU-backed S3 audio-to-JSON smoke test

| Area | Requirement |
|---|---|
| Account and region | the intended account, `ap-southeast-1` only; production Lambda/SQS/Transcribe resources untouched |
| Quota | Singapore On-Demand "Running On-Demand G and VT instances" must allow at least 8 vCPUs (one g4dn.2xlarge). A new account is often at 0: request the increase first, it can take a long time |
| Test bucket | dedicated, globally unique, NO event notifications and no production wiring, short lifecycle. The worker writes the fixed key `transcripts/<job>.json`, so isolation is by bucket only |
| Job role (`ecs-tasks.amazonaws.com`) | `s3:GetObject` on the test audio keys; `s3:GetObject` and `s3:PutObject` on `transcripts/*` and `meralion-state/*`; `s3:ListBucket` on the buckets (otherwise a missing key is a 403, not a 404, and the worker fails by design). No `DeleteObject`. KMS only if the bucket uses a customer-managed key |
| Instance role/profile | ECS agent registration, private ECR pull and CloudWatch Logs. On Batch with EC2 the instance role does the image pull and the `awslogs` driver; an execution role is needed only if the selected features (for example secrets) require one |
| Batch service-linked role | `AWSServiceRoleForBatch` (reuse or let AWS create it) |
| Compute environment | managed EC2, On-Demand, `g4dn.2xlarge` only, min and desired vCPUs 0, max vCPUs 8 (not an absolute spending cap), a current ECS GPU-optimised x86-64 AMI whose documented NVIDIA driver supports CUDA 12.8 (record the AMI id and driver; confirm with `nvidia-smi` on the first run) |
| Launch template | encrypted 100 GiB gp3 ROOT volume, delete on termination, mapped to the AMI's real root device name. The default root disk (about 30 GB) is too small for this image |
| Network | an existing public subnet with a route to an internet gateway, public IP assignment on, a security group with NO inbound rules; reachability of ECR (API and Docker), ECS, S3 and CloudWatch Logs. No NAT gateway and no paid interface endpoints are assumed |
| Logs | a dedicated log group with 14-day retention |
| Job definition | EC2 container, `GPU=1`, `VCPU=4`, `MEMORY=24576`; `linuxParameters.sharedMemorySize=4096`, `maxSwap=0`; attempt timeout 3600 s; retry attempts 1 (one attempt, no automatic retry); `AWS_REGION=ap-southeast-1`; ENTRYPOINT preserved; no override of the working directory |
| Submitter | an identity allowed to register job definitions, submit jobs and `iam:PassRole` for the roles used |

Job definition command (image ENTRYPOINT `python -m meralion_worker` is kept, so the command starts with `run`):

```json
["run",
 "--invocation-id", "Ref::invocation_id",
 "--job-name",      "Ref::stt_job_name",
 "--input",         "Ref::input_uri",
 "--output-bucket", "Ref::output_bucket",
 "--state-uri",     "Ref::state_uri"]
```

Parameter rules: `output_bucket` is a plain bucket name (no `s3://`); `state_uri` is `s3://<bucket>/meralion-state/` (the worker appends the invocation id itself); `invocation_id` and `stt_job_name` use only letters, digits, `.`, `_`, `-` (worker limits 128 and 200 characters). Supply every parameter at submission; no dummy defaults.

## 4. The smoke test (run only after the prerequisites exist and you approve)

```bash
aws s3 cp <local test audio, e.g. CALL_064.wav> s3://<TEST_BUCKET>/inputs/meralion-smoke-001.wav
aws batch submit-job --region ap-southeast-1 --job-name meralion-smoke-001 \
  --job-queue <meralion-stt-test queue> --job-definition <meralion-stt-test definition> \
  --parameters stt_job_name=meralion-smoke-001,invocation_id=meralion-smoke-001,input_uri=s3://<TEST_BUCKET>/inputs/meralion-smoke-001.wav,output_bucket=<TEST_BUCKET>,state_uri=s3://<TEST_BUCKET>/meralion-state/
aws batch describe-jobs --jobs <jobId> --query 'jobs[0].[status,statusReason,container.exitCode,container.reason]'
aws logs tail <log group> --follow
aws s3 cp s3://<TEST_BUCKET>/transcripts/meralion-smoke-001.json .
```

Success is a software result only (it is not an accuracy or production-readiness claim): job `SUCCEEDED` with exit code 0; `transcripts/meralion-smoke-001.json` exists; `status` is `COMPLETED`; `results.transcripts[0].transcript` is non-empty; `meralion.decoder.settings_id` is `vllm-rp105-sched0-fp16-v1`; `meralion.decoder.engine_effective.dtype` is `torch.float16`; `meralion.decoder.execution.mode` is `profile-in-process`; every decode has stop reason `end_token`; state objects exist under `meralion-state/meralion-smoke-001/`. Compare the text with the Blackwell BF16 transcript of the same recording for information only: a T4 FP16 run is expected to differ in places.

Optional second step: submit the SAME `invocation_id` again. The worker must finish immediately from its saved state (resume) without decoding again.

Run `s3_storage_check.py` (next section) BEFORE the first Batch job: it needs no GPU and shows S3 permission problems in minutes instead of after an instance start and an image pull.

## 5. Direct S3 check without a decoder (pending: needs test credentials and a test prefix)

`tools/ec2/s3_storage_check.py` exercises the worker's real `S3Storage` (download, existence check on a missing key, conditional write, read-back, second-write refusal) under the state prefix only. It is unit-tested against a fake client and has NOT been run against AWS. Run it inside the image with the job role's credentials (on an instance or task that has them):

```bash
docker run --rm -e AWS_REGION=ap-southeast-1 -e PYTHONPATH=/opt/meralion \
  -v "$PWD/tools/ec2/s3_storage_check.py:/checks/s3_storage_check.py:ro" --entrypoint python <image> \
  /checks/s3_storage_check.py --input-uri s3://<TEST_BUCKET>/inputs/meralion-smoke-001.wav \
  --state-bucket <TEST_BUCKET> --state-prefix meralion-state/s3-check-001 --region ap-southeast-1
```

## 6. Pending and unvalidated

- The Transformers image path is UNVALIDATED (its lock lacks the wheel's declared dependencies) and was not built.
- FP16 on a T4 has never been executed anywhere: the FP16 profile is a configuration, not a validated result.
- The Blackwell BF16 repetitions of the same strategy differed from each other on 123 of 175 recordings (one repetition passed the frozen-historical rules, the other did not): the profile's repeatability is not established.
- No GPU-container run, no S3 or Batch run, no EC2 measurement of memory or time has happened.

## 7. Cleanup after the test

Scale the compute environment to zero (min and desired vCPUs are already 0) and confirm no instance is running; let the bucket lifecycle expire the test objects; keep or delete the ECR image deliberately (storage is billed); delete the job definition revision and queue only if not reused. Production resources are never touched.
