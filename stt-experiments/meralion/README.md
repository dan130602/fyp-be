# STT refinement handover for the project team

Branch: `codex/stt-meralion-refinements-20261112` in `dan130602/fyp-be`.

## Repository decision

Only fyp-be needs this branch. Its Lambda handler, tests and stt-experiments contain the STT work. fyp-ai and fyp-fe have no uncommitted changes, and their developer-prod-observ-dashboard commits already match GitHub. Their existing evaluation uploader and UI can consume supported exports without source changes. This handover does not deploy, publish results to Supabase, merge main or change infrastructure.

## What to use

Keep Amazon Transcribe as the production default. The Lambda changes add optional per-language custom vocabulary settings and a fixed-English CLM guard, preserving existing behavior when the settings are absent. No CLM was trained. Vocabulary pilots did not establish a broader improvement, so do not enable a candidate globally merely because it is included here.

For local MERaLiON, use the pinned original base model, original energy pauses and optional conservative fee formatting. No LoRA adapter or fine-tuning code is included. Both guarded strategies are retained as experimental comparisons, not automatic production choices.

| 175 development calls | WER | Entity WER |
|---|---:|---:|
| AWS dashboard reference | 1.63% | 3.12% |
| MERaLiON original energy + fee | 1.72% | 4.15% |
| MERaLiON guarded 30 s + fee | 2.04% | 4.06% |
| MERaLiON guarded 24 s + fee | 1.81% | 4.16% |

The AWS full-set figures came from the published dashboard, not an independently retrieved native 175-call run. Guarded 24 s improved English and Mandarin but worsened mixed-language calls. It is not a global replacement or a validated language-routing rule. See stt-experiments/meralion/RESULTS.md and MODEL_COMPARISON.md for per-language results and the separate eight-call provider pilot. Earlier experiment reports are historical; the corrected audit supersedes older adapter figures.

## 1. Get the branch

```bash
git clone --branch codex/stt-meralion-refinements-20261112 https://github.com/dan130602/fyp-be.git
cd fyp-be
```

If you already have a clean checkout, fetch origin and switch to that branch. Preserve any local changes first. Read AGENTS.md and STT_HANDOVER.md. This branch does not need matching frontend or AI branches.

## 2. Prepare private development audio

Obtain the original 175 development recordings privately. Follow STT_HANDOVER.md to validate and copy them into the dataset render folders. Do not convert audio, overwrite different content, modify manifests or use the held-out set.

```bash
cd evaluation-engine
python -m pip install -r requirements-dev.txt
EVAL_BUCKET=offline-placeholder python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --dry-run
cd ..
```

The dry run should find 175 calls, approximately 1042 audio minutes. It does not contact AWS. On PowerShell, set `$env:EVAL_BUCKET = 'offline-placeholder'` before running the Python command without the leading shell assignment.

## 3. Validate the code offline

```bash
python -m pip install -r lambda/transcribe-trigger/requirements-dev.txt
python -m pytest lambda/transcribe-trigger/tests -q
python -m unittest discover -s stt-experiments/meralion -p 'test_*.py'
```

MERaLiON boundary tests need numpy, but do not load a model or require a GPU. Existing segmentation tests live under stt-experiments/segmented-v1. Production Lambda tests are mocked and offline.

## 4. Set up MERaLiON separately on a GPU

Use a private working directory OUTSIDE the Git checkout for audio batches, model weights, outputs and logs. Reuse the verified Origami environment if available. The tested GPU stack used Python 3.12, torch 2.8.0 with CUDA 12.8, transformers 4.57.6 and meralion-3-asr 0.1.0. The provisioned CUDA 12.6 notebook environment alone was not the complete tested stack for Blackwell.

```bash
python -m venv ~/.venvs/meralion-stt
source ~/.venvs/meralion-stt/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r stt-experiments/meralion/requirements.txt
python -m pip install --no-deps meralion-3-asr==0.1.0
export MERALION_WORK="$HOME/meralion-stt-private"
mkdir -p "$MERALION_WORK"
python -B stt-experiments/meralion/prepare_audio.py --dataset-root evaluation-dataset/datasets/advisory_execution_multilingual_v1_development --batch "$MERALION_WORK/batch"
hf download MERaLiON/MERaLiON-3-3B-ASR --revision 9f226d738dee62c60a2f43b0ab7b9a2585a132c3 --local-dir "$MERALION_WORK/model"
```

Install the wrapper with --no-deps after the listed dependencies to use the tested Transformers backend without adding vLLM. Install the evaluation-engine dependencies in the environment used for offline scoring. The batch preparation step reads render metadata and recordings, not gold transcripts, and copies audio without conversion after validating the complete input set. Transfer that private batch to the cluster if you prepared it locally. Never commit it. On Origami, load the Python module needed by your virtual environment and set MERALION_PYTHON to its Python executable when necessary.

## 5. Run a smoke test, then the original configuration

Request a NEW active GPU allocation; old Slurm job IDs from the experiment logs have expired. sbatch/srun do not accelerate inference by themselves; they assign the GPU resources. Replace JOB_ID below with your allocation.

```bash
export MERALION_PYTHON="$HOME/.venvs/meralion-stt/bin/python"
bash stt-experiments/meralion/run_origami.sh JOB_ID energy 30 --calls CALL_001 CALL_056
bash stt-experiments/meralion/run_origami.sh JOB_ID energy 30 --all
```

Compatible completed calls resume. The runner prints completed calls out of 175, a progress bar and elapsed time. It freezes model parameters, rejects adapter checkpoints and checks base weight file hashes. Do not attach the earlier private LoRA adapters. Guarded experiments can be reproduced separately with `guarded 30` and `guarded 24`; do not enable them by default.

## 6. Score with the original evaluator

Bring private result JSON files back to the machine with the original development dataset and audio. Use unique run IDs. From the repo root:

```bash
python -B stt-experiments/meralion/score_results.py --results "$MERALION_WORK/results/energy-30" --run-id meralion-energy-new-raw
python -B stt-experiments/meralion/score_results.py --results "$MERALION_WORK/results/energy-30" --run-id meralion-energy-new-fee --fee-formatting
```

The adapter preserves signs, currency and percent suffixes, validates audio identity, writes Transcribe-shaped JSON into git-ignored evaluation-engine/runs and invokes the existing scripts/score.py unchanged. It refuses incomplete full runs unless --allow-partial is explicitly used for smoke testing. Keep raw and formatted runs separate. Fee formatting only renders standalone half-percent phrases equivalently; it does not rewrite consent, negation, order types or monetary amounts.

Verify normalizer 1.4.0 and compare matching reference denominators, audio hashes and normalization rules. Report WER, entity WER and substitutions/deletions/insertions overall and per language. Local MERaLiON does not provide calibrated confidence, word timestamps or speaker labels. Do not present missing-label zero scores as measured diarization accuracy.

## 7. Optional AWS settings and owner deployment

Only use actual vocabulary resources created by the owner in the same AWS region. For a controlled multilingual vocabulary experiment, use TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN, TRANSCRIBE_LANGUAGE_ID_MODE=multiple, TRANSCRIBE_MAX_SPEAKER_LABELS=2 and TRANSCRIBE_VOCABULARIES=en-US=YOUR_EN_RESOURCE,zh-CN=YOUR_ZH_RESOURCE. Leave TRANSCRIBE_VOCABULARIES absent for the baseline. TRANSCRIBE_LANGUAGE_MODEL_NAME is only for an approved fixed en-US experiment with language identification disabled; it is not a Mandarin/mixed solution.

Store API keys and AWS/Supabase credentials only in local git-ignored .env files or approved credential stores. No new paid provider run, Lambda deployment, Terraform change, Supabase publication or model switch is part of checking out this branch. The owner reviews deployment and runs the official held-out set once at the end. Use STT_HANDOVER.md for exporting and publishing a complete run; do not change the publisher, dashboard schema or scorer for this experiment.


## Branch validation

Validated on 12 November 2026: 46 mocked Lambda tests and seven MERaLiON boundary/adapter/formatting tests passed. The portable adapter and fee formatter matched all 175 saved audited transcripts exactly. Changed Python sources compiled, and the staged-file and whitespace checks passed. GPU inference was not repeated just for packaging.
