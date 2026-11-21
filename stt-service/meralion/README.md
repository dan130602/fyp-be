# MERaLiON speech-to-text worker

A batch worker that transcribes call recordings with [`MERaLiON-3-3B-ASR`](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) on a GPU. It wraps the model in the long-audio pipeline
that the evaluation in `evaluation-engine/` measured: 30-second windows every 15 seconds, overlap merging, bounded retries of uncertain seams, and number / percentage formatting.
It can run the original model or the **fine-tuned ClearCall model** (decoder LoRA + encoder LoRA + full speech adapter, "J5XE").

Output: one JSON per call, `out/transcripts/<job_name>.json`, with the transcript as plain text at `results.transcripts[0].transcript`. No timestamps, no speaker labels, no confidence
scores. Speaker attribution is a separate component and is **not** part of this worker or its Docker image.

> The development of this service was assisted by MERaLiON, an AI model developed by the Agency for Science, Technology and Research (A\*STAR), Singapore. MERaLiON is used under the
> [MERaLiON Public Licence](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-3-Public-Licence.pdf).

## Run the fine-tuned model

**What you need:** Linux x86-64, CPython 3.12, a CUDA 12.8 GPU (24 GB is enough; tested on an RTX PRO 6000 Blackwell slice and an L40S), and a C/C++ toolchain with `nvcc` and `ninja`
(vLLM compiles kernels on first use). The base model is **not** needed: the download below is the complete merged model.

```bash
git clone https://github.com/dan130602/fyp-be.git && cd fyp-be && git checkout stt/meralion-release
cd stt-service/meralion

# 1. the pinned environment (exactly what the Dockerfile installs)
python3.12 -m venv .venv && source .venv/bin/activate
pip install --no-deps -r requirements-vllm.lock
pip install --no-deps meralion-3-asr==0.1.0           # its SHA-256 is pinned in configs/third_party/meralion-3-asr-0.1.0.json

# 2. the weights: download model/ (6.6 GB) from Hugging Face and check all 14 files against configs/models/meralion-3-3b-asr-sft-j5xe-ep2.json
pip install huggingface_hub
python tools/get_finetuned_model.py
```

The script prints the command to transcribe calls; it is

```bash
python -m meralion_worker run --invocation-id my-first-run --jobs jobs.json \
    --model-dir models/clearcall-j5xe/model --config configs/runtime/j5xe-ep2-vllm-rp105-sched0.json \
    --output-dir out --state-dir state --summary-output summary.json
```

`jobs.json` is `{"jobs": [{"job_name": "CALL_001", "input": "/path/to/CALL_001.wav"}]}` (wav, mp3, flac or ogg; mono or stereo is converted to 16 kHz mono). Running again with the same
`--invocation-id` resumes an interrupted run. The weights, model card, metrics and licence are in the Hugging Face repository
[`tanedric/meralion-3-3b-asr-clearcall-j5xe`](https://huggingface.co/tanedric/meralion-3-3b-asr-clearcall-j5xe); it also has the 288 MB trained-parameters-only file.

## Run the original model

Download the pinned base revision into `model/` (`hf download MERaLiON/MERaLiON-3-3B-ASR --revision 9f226d738dee62c60a2f43b0ab7b9a2585a132c3 --local-dir model`) and use
`--model-dir model --config configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json` (the configuration of the reported results).
`python -m meralion_worker check-model --model-dir model --config <that file>` hashes the directory against `configs/models/meralion-3-3b-asr-base-9f226d7.json`.

## Docker

The [Dockerfile](Dockerfile) bakes one model and one runtime configuration into the image and verifies them when it is built and at every start. **The image has not been built or GPU-tested
with either model** (see the header of the Dockerfile for what has and has not been verified). By reading the code, the fine-tuned model needs no Dockerfile change, because the manifest that
the build checks is the one named by the runtime configuration:

```bash
cd stt-service/meralion
mkdir -p artifacts && pip download --no-deps meralion-3-asr==0.1.0 -d artifacts          # the Dockerfile copies the wheel and checks its SHA-256
python tools/get_finetuned_model.py && cp -r models/clearcall-j5xe/model model            # the Docker build context needs the model in ./model
docker build --build-arg BACKEND=vllm --build-arg RUNTIME_CONFIG=j5xe-ep2-vllm-rp105-sched0.json \
             --build-arg WORKER_GIT_COMMIT=$(git rev-parse HEAD) -t meralion-stt-worker:j5xe .
docker run --rm --gpus all -v "$PWD/calls:/in:ro" -v "$PWD/out:/var/meralion/out" -v "$PWD/state:/var/meralion/state" meralion-stt-worker:j5xe \
       run --invocation-id my-first-run --jobs /in/jobs.json --output-dir /var/meralion/out --state-dir /var/meralion/state
```

The Dockerfile's header still describes the 21-file base checkpoint; the fine-tuned model has 14 files, which the manifest lists. To bake the original model instead, put the base model in `model/`
and build with the original model's runtime configuration (see above). A different checkpoint needs a separate image.

## What is in this folder

| Path | Purpose |
|---|---|
| `meralion_worker/` | the worker: windowing (`strategy_fg.py`), retries (`retry_selection.py`), formatting (`formatters.py`), decoders (`decoders/`), vendored merge code (`vendored/`, see `vendored/SOURCES.json`) |
| `configs/runtime/` | runtime configurations; `fg175-...-vllm-rp105-sched0.json` is the evaluated one and `j5xe-ep2-vllm-rp105-sched0.json` runs the fine-tuned model |
| `configs/models/` | model manifests: SHA-256 of every checkpoint file (base model and fine-tuned model) |
| `tools/get_finetuned_model.py` | download and verification of the fine-tuned model |
| `tools/origami/` | Slurm scripts from the SMU Origami GPU cluster; replace `YOUR_USERNAME` in them with your cluster account. Not needed elsewhere |
| `requirements-vllm.lock`, `requirements-transformers.lock`, `Dockerfile` | the pinned environment and its packaging (Linux x86-64, CPython 3.12) |
| `tests/` | worker tests: `pip install -r requirements-dev.txt && python -m pytest tests -q` (CPU only; no GPU or weights; several tests call `git` and `bash`) |

The fine-tuning code that produced the model is in [`stt-experiments/meralion-sft/`](../../stt-experiments/meralion-sft/README.md).
Some files under `tools/` read research files that are not in this repository (`export_vendored.py` regenerates `vendored/` from `stt-experiments/meralion-runner/`, which lives on another
branch); the worker itself does not need them.
