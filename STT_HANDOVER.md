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

### Noisy and channel-degraded audio (environment_v2)

Every call has ONE fixed audio environment: C1 clean, N1 office, N2 cafe, N3 MRT/train noise, D1 telephone, D2 muffled client. Only the client's turns are changed;
the RM, the transcripts, the gold and the timings are identical to the clean call. 146 of the 175 development calls have changed audio; the other 29 (C1) are the original recordings.
`--audio-condition clean` (the default) is exactly the behaviour above.

**What is in Git:** `audio_augmentations/environment_v2/{assignments.json, config.json}` (the frozen condition of each call) and, per changed call,
`calls/CALL_xxx/audio_variants/environment_v2/augmentation.json` (how it was made, with the WAV's sha256). **Not in Git:** the 146 `audio.wav` files (about 2.3 GB). The owner sends them as a zip.

**One-time setup (after `git pull`, and after the clean recordings above are in place).** Paste this into your Codex (replace the folder):

```text
Read fyp-be/AGENTS.md and fyp-be/STT_HANDOVER.md. This task authorises offline setup only: no AWS, no paid jobs, no publishing, no commits, no edits to any existing file.
My unzipped folder is <ABSOLUTE LOCAL FOLDER>: it holds 146 files named CALL_xxx.wav (they may be inside an audio/ subfolder). The target dataset is
evaluation-dataset/datasets/advisory_execution_multilingual_v1_development/ (call it DEV).

Validate everything before copying anything:
1. My folder has exactly 146 CALL_xxx.wav files: no extras, no duplicates, no other file types.
2. Each CALL_xxx is in DEV/dataset_manifest.json and already has DEV/calls/CALL_xxx/audio_variants/environment_v2/augmentation.json (that file comes from Git). Exactly 146 calls have one.
3. sha256 of the WAV (plain file bytes; do NOT normalise line endings) equals "output"."sha256" in that augmentation.json, and its "assignment"."condition" is not "C1"
   and equals the call's condition in DEV/audio_augmentations/environment_v2/assignments.json.
4. That assignments.json lists 175 calls, and the 146 above are exactly the calls whose condition is not C1 (C1 = clean: that call keeps its original recording and has no variant file; N1-N3 and D1-D2 are the changed-audio conditions).
If anything is wrong, copy nothing and report it.

Then COPY (never move) each file to DEV/calls/CALL_xxx/audio_variants/environment_v2/audio.wav: skip it if an identical file is already there, never overwrite different content.
Confirm 146 audio.wav now exist and their sha256 still match. Then, from evaluation-engine, with EVAL_BUCKET set to any placeholder bucket name, run these (they contact no AWS):
  python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --audio-condition environment_v2 --skip-clean-controls --dry-run
  python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --audio-condition environment_v2 --dry-run
Report the counts copied, the calls and audio minutes each dry run finds (expected 146 calls / about 870 minutes, and 175 calls / about 1,042 minutes), any error, and `git status`.
```

**Running it (billable; clean calls are not sent again):**

```powershell
python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --audio-condition environment_v2 --calls CALL_033 CALL_064 CALL_147 CALL_214 CALL_028 --run-id dev-v2-<idea>-canary   # 5 calls, about 16 min
python -B scripts/collect.py --dataset advisory_execution_multilingual_v1_development --render-id auto --all --audio-condition environment_v2 --skip-clean-controls --run-id dev-v2-<idea>-1 --timeout-minutes 120   # 146 calls, about 870 min, about US$21
python -B scripts/assemble_v2_run.py --v2-run dev-v2-<idea>-1 --clean-run <clean run id> --out dev-v2-<idea>-1-full   # offline: adds the 29 clean calls' earlier transcripts
python -B scripts/score.py --run dev-v2-<idea>-1-full
python -B scripts/compare_conditions.py --clean-run <clean run id> --v2-run dev-v2-<idea>-1-full                      # offline, paired
```

**One command instead of the steps above** (it stops at the first failing step and never publishes):

```powershell
python -B scripts/run_eval.py --run-id dev-v2-<idea>-1 --audio-condition environment_v2 --clean-run <clean run id>                        # sends all 175 calls (clean ones with their original audio)
python -B scripts/run_eval.py --run-id dev-v2-<idea>-1 --audio-condition environment_v2 --clean-run <clean run id> --skip-clean-controls  # 146 calls; adds the clean transcripts
python -B scripts/run_eval.py --run-id dev-clean-<idea>-1                                                                              # clean recordings: collect, score, export
```

Add `--dry-run` to print the steps with a free collect dry run. The steps are collect, (assemble), score, (compare), export.

**Dashboard.** A noisy run is exported under its own dataset, `<dataset>__environment_v2`, so the fyp-ai dashboard shows a separate card, "... (noisy and channel-degraded)", beside the clean baseline.
Its run page has a "By audio condition" table and the paired clean-versus-changed table (the comparison saved by `compare_conditions.py` is embedded in the exported `run.json`, so run it before the export).
Changed calls play their variant recording. Publishing to Supabase stays a separate, deliberate step from the fyp-ai folder: `python -B evaluation/scripts/upload_stt_run.py --run-id <final run id>` (the assembled `-full` run when you skipped the clean calls).

- `environment_v2` needs `--render-id auto`. The file for each call comes from `assignments.json`: a clean (C1) call still points at its original recording, every other call at `audio_variants/environment_v2/audio.wav`. A missing or changed variant stops the run before anything is uploaded; it never falls back to clean audio.
- `--skip-clean-controls` leaves the C1 calls out (they are the original audio you already transcribed). The run records which were skipped.
- `assemble_v2_run.py` and `compare_conditions.py` reuse the clean run's transcripts only if both runs used the same Transcribe job settings (language options, speaker labels, custom vocabulary) and `TRANSCRIBE_*` settings; a changed handler file is only a warning. Otherwise they stop with "clean controls must be re-transcribed" and you run the 29 clean calls again with the current handler (`--audio-condition environment_v2` without `--skip-clean-controls`).
  The copied transcripts keep their original dates and are marked `reused_from_run`. If you changed the handler to test an idea, the clean baseline must be re-run with that same handler before comparing.
- Compare conditions through the PAIRED difference (same call, clean versus changed audio), never raw WER: the calls in each condition differ.

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
