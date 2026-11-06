# STT development handover

How to set up, run and publish. The rules are in `AGENTS.md` (read it first).

## What you receive and what is in Git

- Git has everything except the recordings: the production Lambda, `evaluation-engine/`, and the **development** dataset's non-audio files
  (`evaluation-dataset/datasets/advisory_execution_multilingual_v1_development/`: `dataset_manifest.json`, `authoring/`, and per call `ground_truth.json`, `transcript.json`, `speaker_roles.json`, `renders/<render>/render_manifest.json`).
  The held-out split and the full release are not in your copy.
- The owner sends the 175 recordings separately (not in Git): a folder of `CALL_xxx.wav` or `CALL_xxx.mp3`, exactly as rendered. Do not convert them.
  The 175 ids are not consecutive; keep them as they are.

## One-time setup: put the recordings in place

The engine reads exactly one file named `audio.wav` or `audio.mp3` inside `calls/CALL_xxx/renders/<render folder>/`. A file named `CALL_xxx.wav` is not found, so each received file is copied there and renamed.

Paste this into your Codex (replace the folder):

```text
Read fyp-be/AGENTS.md and fyp-be/STT_HANDOVER.md. This task authorises offline setup only: no AWS, no paid jobs, no publishing, no commits.
My recordings are in <ABSOLUTE LOCAL FOLDER>, one file per call named CALL_xxx.wav or CALL_xxx.mp3 (they may be in subfolders).

For each call listed in evaluation-dataset/datasets/advisory_execution_multilingual_v1_development/dataset_manifest.json (175 ids):
1. Under calls/<ID>/renders/ there is exactly one folder with a render_manifest.json. Read it. The target name is its "audio_file" (audio.wav or audio.mp3);
   if "audio_file" is absent the name is audio.mp3.
2. Find <ID> with the matching extension (.wav or .mp3) in my folder. The extension must match the target name; never convert a file.
3. Check the file: sha256(file_bytes.replace(b"\r\n", b"\n")) must equal the manifest's "audio_sha256" (compute it in memory; never change the file).
4. COPY (do not move) it to calls/<ID>/renders/<that folder>/<target name>. Skip it if an identical file is already there; never overwrite different content.
Before copying anything, validate all 175 (missing, extra or duplicate files, extension mismatches, hash mismatches). If anything is wrong, copy nothing for it and report it; do not edit manifests or any other file.

Afterwards confirm each of the 175 calls has exactly one audio.wav/audio.mp3 under renders/*/. Then, from evaluation-engine, with EVAL_BUCKET set to any placeholder bucket name,
run: python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --dry-run
(it contacts no AWS) and report the number of calls and minutes it finds. Report the counts and `git status`. Do not commit.
```

Expected: 175 placed, the dry run lists 175 calls (about 1,042 minutes).

## Local setup and commands

From `evaluation-engine`, Python 3.11+: `python -m pip install -r requirements-dev.txt`, then copy `.env.example` to `.env` and fill in (the owner gives the values privately; never write them elsewhere):
`AWS_REGION=ap-southeast-1`, the AWS credentials, `EVAL_BUCKET` (an isolated bucket with no production trigger), and the production language setup:

```dotenv
TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN
TRANSCRIBE_LANGUAGE_ID_MODE=multiple
TRANSCRIBE_MAX_SPEAKER_LABELS=2
```

Without language options the handler defaults to fixed en-US. The checked-in Terraform does not set these, so an evaluation setting is not proof that the live Lambda uses it: always state the effective settings of a run.

```powershell
python -m pytest tests -q
python -B scripts/check_aws.py                      # read-only AWS check
python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --dry-run      # free
python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --calls CALL_059 CALL_064 CALL_094 CALL_147 --run-id dev-<idea>-canary
python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --run-id dev-<idea>-1 --timeout-minutes 120
python -B scripts/score.py --run dev-<idea>-1       # offline, safe to repeat
python -B scripts/export_to_fyp_ai.py --run dev-<idea>-1
```

`--render-id auto` finds the single `renders/*/audio.wav|mp3` of each call. Results are in `runs/<run id>/scores/summary.json` (`runs/` is git-ignored). Run production handler tests from `lambda/transcribe-trigger` with `python -m pytest tests -q`.
Production handler tests and the engine tests are offline.

## Publishing results (Supabase)

Check out `fyp-ai` beside `fyp-be` (the export writes its bundle to `fyp-ai/evaluation/stt_runs/<run id>/`). From `fyp-ai`: `python -m pip install -r evaluation/requirements.txt` and put `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`
(from the owner, privately) in a git-ignored `.env`. The owner must already have applied `supabase/migrations/20261106000000_create_stt_evaluation.sql`.

```powershell
python -B evaluation/scripts/upload_stt_run.py --run-id <run id> --dry-run     # validates, sends nothing
python -B evaluation/scripts/upload_stt_run.py --run-id <run id>
```

Use `upload_stt_run.py`, not the NLP `upload_run.py`. Publish only complete development runs; an identical re-upload is a no-op, a changed run with a used id is refused, and published runs are never edited. Publishing does not deploy anything.
Production deployment (ECR image, Terraform apply) is the owner's job.

## Prompt for the ongoing improvement work

```text
Read AGENTS.md and STT_HANDOVER.md. Improve production speech-to-text using the development data only. Start by diagnosing the baseline's substitutions and deletions
(runs/<baseline>/scores, confusions) and propose a small custom-vocabulary experiment, then run it. Preserve ground truth, audio, scoring and the held-out split.
Production code must not read any benchmark file. Report, per run: what changed, the effective TRANSCRIBE_* settings, and WER / entity WER / speaker attribution overall and per language.
CLM is a separate feasibility experiment: it is not supported with multi-language identification and not for zh-CN, so do not silently turn off code-switching. Do not present scores on the
calls you trained or built a vocabulary from as unseen accuracy. Other STT models go in a new folder and are scored through Transcribe-shaped JSON (see AGENTS.md).
```

## Notes

- Hash convention: the repository's own hash helper normalises CRLF to LF before hashing (even for audio), so a render manifest's `audio_sha256` is not an ordinary file SHA-256. This is why the setup check above uses `replace(b"\r\n", b"\n")`. Never change either hashing implementation.
- `check_semantics.json`, checklists and the original authoring inputs are not read by collect/score/export, so their absence from the development folder is expected.
- GitHub blocks files over 100 MiB and the recordings total about 2.3 GB, which is why audio is shared outside Git.
