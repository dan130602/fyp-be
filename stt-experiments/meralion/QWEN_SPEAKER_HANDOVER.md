# Teammate handover: original-audio Qwen v8 speaker refinement

Only dan130602/fyp-be requires code changes. gnlyrad/fyp-ai and dan130602/fyp-fe retain their existing transcript consumers. This branch adds experimental STT inference tooling; it does not change the API/downstream schema, deploy a GPU service, alter infrastructure, run paid jobs or publish to Supabase. Existing Lambda/custom-vocabulary code from the earlier STT release is preserved. No denoising implementation is included or enabled. Historical experiment records remain outside the release rather than deleting audit evidence.

Branch: `codex/stt-qwen-v8-no-denoising-20261118`. It extends the earlier `codex/stt-meralion-refinements-20261112` STT release. Review its base against current main before merging; no main history is rewritten.

## 1. Get the code and run CPU checks

```bash
git fetch origin
git switch --track origin/codex/stt-qwen-v8-no-denoising-20261118
python -m pip install -r evaluation-engine/requirements.txt
python -m pip install -r stt-experiments/meralion/qwen-speakers/requirements-labels.txt
python -m unittest discover -s stt-experiments/meralion/qwen-speakers -p 'test_*.py'
```

Use a clean checkout or preserve local work before switching. Boundary tests additionally require numpy. The original evaluator needs its existing requirements. Tests do not download models or contact AWS. Existing STT_HANDOVER.md still governs private development recordings and scoring; never change its datasets, manifests, references, normalizer or held-out split.

## 2. Process existing predictions without a GPU

Follow qwen-speakers/README.md for `pipeline.py`. The function applies v3, v6, then v8, not just the last file. Input Qwen words/timestamps remain frozen. Every input is a model prediction from the same audio, never gold text/roles/timings. This is sufficient to reproduce the historical label refinement when the saved input predictions are available. Those private inputs are not in Git.

## 3. Prepare fresh GPU stages only when needed

Keep audio, models, batches and outputs outside the Git checkout. Reuse the validated Origami environments where possible. The tested ASR stack is Python3.12, torch2.8.0+cu128, transformers4.57.6 and meralion-3-asr0.1.0 with the Transformers backend and batch size1. KMeans Pyannote uses the separate tested Pyannote environment; its clustering-module source fingerprint must match expected-diarizer-model.json. Qwen uses a separate qwen-asr0.0.6 environment. Inspect the recorded expected model receipts before substituting versions. The earlier generic CUDA12.6 notebook environment alone is not the complete runtime. Avoid reinstalling into a working shared environment.

Use these pinned models:

- MERaLiON/MERaLiON-3-3B-ASR, revision `9f226d738dee62c60a2f43b0ab7b9a2585a132c3`, local directory `/private/models/model`.
- pyannote/speaker-diarization-community-1, revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`, accepted HuggingFace conditions and private read-token login/cache required.
- Qwen/Qwen3-ForcedAligner-0.6B, revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, local directory `/private/models/models/forced-aligner`.

Keep HuggingFace tokens out of shell arguments, Git and chat. Authentication belongs in `hf auth login`. No inference stage downloads models implicitly; the pinned snapshots must be complete. If tokenizer assets such as merges.txt are missing, resolve the pinned tokenizer artifact before inference rather than substituting an arbitrary model. Expected checkpoint hashes and original runtime receipts are included as metadata only.

## 4. Private batch manifest

Create `/private/batch/manifest.json` outside Git after validating source files. This is an inference input manifest, not a modification of the dataset's manifest:

```json
{
  "scope": "development-only-original-audio",
  "meralion_revision": "9f226d738dee62c60a2f43b0ab7b9a2585a132c3",
  "qwen_config_sha256": "d616c65d46c4b90bdc651b0a0963ea932732241140f337f9bb6b0335a9c8ef09",
  "calls": [{"call_id": "CALL_001", "audio": "audio/CALL_001.wav", "audio_sha256": "REPLACE_WITH_PLAIN_FILE_SHA256"}]
}
```

Use original .wav/.mp3 file bytes, never derived/denoised recordings. Paths must stay inside the private batch; no overwriting different files. For the environment_v2 development benchmark there are146environment variants plus29original clean calls,175total. Do not substitute old clean recordings or change the source provenance. No held-out audio is allowed in this development tool. IDs route files only; language/condition/reference text is not accepted by inference.

## 5. Run stages in order

From the repository root, using the appropriate environment's Python for each GPU stage:

```bash
python stt-experiments/meralion/qwen-speakers/predict.py preflight --batch /private/batch --models /private/models --output /private/output
# pyannote environment:
python stt-experiments/meralion/qwen-speakers/predict.py diarize --batch /private/batch --models /private/models --output /private/output
# MERaLiON environment:
python stt-experiments/meralion/qwen-speakers/predict.py asr --batch /private/batch --models /private/models --output /private/output
python stt-experiments/meralion/qwen-speakers/predict.py auxiliary --batch /private/batch --models /private/models --output /private/output
# Qwen environment:
python stt-experiments/meralion/qwen-speakers/predict.py align --batch /private/batch --models /private/models --output /private/output
# CPU label environment with original evaluator utilities:
python stt-experiments/meralion/qwen-speakers/predict.py labels --batch /private/batch --output /private/output
```

On Slurm, prefix each GPU command with `srun --jobid YOUR_CURRENT_JOB --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1` inside your existing allocation. Do not paste an old job ID. Completed clips/calls are checkpointed and source/model/settings hashes checked on resume. Diarization decodes audio in memory, avoiding built-in TorchCodec decoding. Speaker labels are anonymous IDs; do not claim they infer employee/client roles. Final files are `/private/output/refined/CALL_xxx.json`, Transcribe-shaped results with frozen words/times and refined speaker labels.

## 6. Evaluate and integrate

Copy final outputs into a new private evaluation run's raw/transcripts directory with matching metadata, then use the original evaluation-engine/scripts/score.py and normalizer1.4.0. Follow STT_HANDOVER.md; do not edit scorer code to improve numbers. Use a fresh matched original-audio control for new inference experiments, with the same resampling, model wrapper, runtime, generation settings and pause strategy. Historical mixed-provenance ASR caches do not establish a fresh preprocessing improvement.

For deployment, the owner must provide GPU hosting, dependencies/model caches, a job runner and the existing Transcribe-shaped output contract. This branch does not replace the production Lambda or alter fyp-ai/fyp-fe consumers. See qwen-speakers/RESULTS.md for the historical99.25%/2.67%/5.77% result and its caveats. No model weights were changed.
