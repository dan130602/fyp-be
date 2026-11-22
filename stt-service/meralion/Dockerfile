# syntax=docker/dockerfile:1.7
# Stage 4 PACKAGING of the MERaLiON STT worker. Nothing here selects or promotes a backend, a profile or a production default:
# the two build arguments below have NO default, so an image always names the configuration it carries, and a build that does not name one fails in the first step.
#
#   docker build --build-arg BACKEND=vllm --build-arg RUNTIME_CONFIG=<file under configs/runtime/> --build-arg WORKER_GIT_COMMIT=$(git rev-parse HEAD) -t <name> stt-service/meralion
#
#   BACKEND         vllm | transformers   selects requirements-<BACKEND>.lock (one image carries ONE backend stack: torch 2.9.1 / vllm 0.16.0 for vllm, torch 2.8.0+cu128 for transformers)
#   RUNTIME_CONFIG  the runtime config file name; its `decoder.backend` must equal BACKEND (checked in the build). Validation status of each profile is NOT decided here: see the validation records.
#
# BACKEND=transformers is UNVALIDATED and is expected to FAIL the `pip check` below: the verified wheel declares vllm==0.16.0 (which needs torch 2.9.1), fastapi, uvicorn, httpx and python-multipart as unconditional
# dependencies, and requirements-transformers.lock (torch 2.8.0, no vllm) has none of them. Do not build or ship that path until it is deliberately packaged (see requirements-transformers.lock). BACKEND=vllm is the path
# that was assembled and checked; a first build and its CPU container checks are recorded separately.
#
# What the image is: linux/amd64, Python 3.12 (Ubuntu 24.04), CUDA 12.8 DEVEL tools (FlashInfer and Triton compile kernels at run time: nvcc, gcc/g++, Python headers and `ninja` on PATH are required), the
# dependency lock installed with --no-deps, the VERIFIED meralion-3-asr wheel (sha256 checked against configs/third_party/meralion-3-asr-0.1.0.json), the worker, the tracked patch manifest (NOT applied: the
# vLLM MIG patch concerns a MIG slice and is inert on a whole GPU), and the complete checkpoint baked in (21 files for the original model, 14 for the fine-tuned model; the model manifest named by the runtime configuration lists them) (hash-verified in the build and at every start; no run-time download).
# What the build verifies: pip check; the wheel hash; `verify-environment` (vendored files and the installed package against their manifests); `check-model` (every checkpoint file against the model manifest).
# The image is built and tested; the checks above run in every build.
ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu24.04@sha256:9a8fffc32a955361aa66d754e7a0cda2513052eaf5aaf8f3ba69b80578d1c6a9
FROM --platform=linux/amd64 ${BASE_IMAGE}

ARG BACKEND
ARG RUNTIME_CONFIG
ARG WORKER_GIT_COMMIT=unknown
ARG WORKER_UID=10001

LABEL org.opencontainers.image.title="meralion-stt-worker" org.opencontainers.image.revision="${WORKER_GIT_COMMIT}" meralion.backend="${BACKEND}" meralion.runtime_config="${RUNTIME_CONFIG}"

RUN case "${BACKEND}" in vllm|transformers) ;; *) echo "build argument BACKEND must be 'vllm' or 'transformers' (it has no default)" >&2; exit 2 ;; esac \
 && test -n "${RUNTIME_CONFIG}" || { echo "build argument RUNTIME_CONFIG is required (it has no default)" >&2; exit 2; }

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev build-essential libsndfile1 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv

ENV PATH=/opt/venv/bin:${PATH} CUDA_HOME=/usr/local/cuda \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HOME=/home/worker \
    FLASHINFER_WORKSPACE_BASE=/var/cache/meralion/flashinfer VLLM_CACHE_ROOT=/var/cache/meralion/vllm TRITON_CACHE_DIR=/var/cache/meralion/triton \
    TORCHINDUCTOR_CACHE_DIR=/var/cache/meralion/inductor XDG_CACHE_HOME=/var/cache/meralion/xdg \
    MERALION_MODEL_DIR=/opt/meralion/model MERALION_CONFIG=/opt/meralion/configs/runtime/${RUNTIME_CONFIG} MERALION_WORKER_GIT_COMMIT=${WORKER_GIT_COMMIT}

# 1. the dependency lock (--no-deps: every dependency is pinned). The Transformers lock's +cu128 torch wheels need the PyTorch index; the vLLM lock uses default PyPI wheels.
COPY requirements-${BACKEND}.lock /opt/meralion/requirements.lock
RUN if [ "${BACKEND}" = "transformers" ]; then EXTRA="--extra-index-url https://download.pytorch.org/whl/cu128"; else EXTRA=""; fi \
 && pip install --no-cache-dir --no-deps ${EXTRA} -r /opt/meralion/requirements.lock

# 2. the VERIFIED meralion-3-asr artifact: installed only if its sha256 equals the one in the package manifest.
COPY artifacts/meralion_3_asr-0.1.0-py3-none-any.whl /tmp/wheel/
COPY configs/third_party/meralion-3-asr-0.1.0.json /tmp/package-manifest.json
RUN want="$(python -c "import json; print(json.load(open('/tmp/package-manifest.json'))['artifact']['sha256'])")" \
 && echo "${want}  /tmp/wheel/meralion_3_asr-0.1.0-py3-none-any.whl" | sha256sum -c - \
 && pip install --no-cache-dir --no-deps /tmp/wheel/meralion_3_asr-0.1.0-py3-none-any.whl \
 && rm -rf /tmp/wheel /tmp/package-manifest.json \
 && pip check

# 3. the worker, its configuration, and the tracked patch manifest (read by the runtime identity; the patch is not applied here).
COPY configs /opt/meralion/configs
COPY meralion_worker /opt/meralion/meralion_worker
COPY patches /opt/meralion/patches
WORKDIR /opt/meralion
RUN test -f "${MERALION_CONFIG}" \
 && python -c "import json, os, sys; b = json.load(open(os.environ['MERALION_CONFIG']))['decoder']['backend']; sys.exit(0 if b == sys.argv[1] else 'RUNTIME_CONFIG names backend ' + b + ', the build is for ' + sys.argv[1])" "${BACKEND}" \
 && python -m meralion_worker verify-environment

# 4. the checkpoint, last (the largest layer): complete, hash-verified in the build and again at every start; never downloaded at run time.
COPY model /opt/meralion/model
RUN python -m meralion_worker check-model

# 5. a non-root runtime user with the only writable locations: the output/state mounts and the kernel/compile caches.
RUN useradd --uid ${WORKER_UID} --create-home worker \
 && mkdir -p /var/meralion/out /var/meralion/state /var/cache/meralion/flashinfer /var/cache/meralion/vllm /var/cache/meralion/triton /var/cache/meralion/inductor /var/cache/meralion/xdg \
 && chown -R worker:worker /var/meralion /var/cache/meralion /home/worker
USER worker
ENTRYPOINT ["python", "-m", "meralion_worker"]
CMD ["--help"]
