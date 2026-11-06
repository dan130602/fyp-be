# STT improvement: agent instructions

Scope: improving the production speech-to-text step (Amazon Transcribe today, possibly other STT models later), measured with `evaluation-engine/`.
Read `STT_HANDOVER.md` first (setup, commands, publishing). Unrelated backend work needs its own task scope.

## The goal and the one rule that matters

Lower error rate on the **development** calls, achieved by changing the production speech-to-text code, never the data or the scoring.
`evaluation-engine` loads `lambda/transcribe-trigger/app/handler.py` **unchanged** and runs it, so an improvement to Transcribe usage goes in that handler
(any environment variable it reads is named `TRANSCRIBE_...`; the engine forwards all of them, so no engine change is needed). Anything you change
elsewhere in how a score is produced would make a better number meaningless.

## You MAY change

- Anything under `lambda/` (the transcribe-trigger handler, its helper modules, `requirements*`, Dockerfile and tests). This includes adding another STT engine or open-source model
  there. Keep changes backward compatible: with no new environment variable set the handler must behave exactly as it does now. Add or update tests for what you add.
  Preserve the S3 event interface, the job-name algorithm (shared with Java), the output bucket/key convention, error propagation and the downstream transcript format.
- `evaluation-engine/.env` (git-ignored, yours): `TRANSCRIBE_...` variables for a run.
- New folders for anything that is not the handler, for example `stt-experiments/`: vocabulary lists and the scripts that create them, language-model training text, notes, and experiments
  with other STT models. Never put credentials, audio, raw run outputs or copied reference transcripts there.
- The documentation (`STT_HANDOVER.md`, this file's examples) only to document what you added.

## You must NOT change

- **Ground truth and datasets:** everything under `evaluation-dataset/` (the `ground_truth.json`, `transcript.json`, `authoring/` files, manifests, splits). Never edit, regenerate, re-render, rename or "fix" a
  reference, even if you believe it is wrong: report it to the owner.
- **The scoring:** `evaluation-engine/transcribe_eval/` (normaliser, tokenisation, alignment, entity WER, export), `evaluation-engine/scripts/`, `evaluation-engine/domain_terms.json` and their tests.
  If a score looks unfair (two spellings of one word, say), write it down with examples and send it to the owner. A changed scorer breaks comparison with every earlier run.
- **The split:** tune only on `advisory_execution_multilingual_v1_development`. Do not create, copy, ask for or use the held-out set or the full release. The owner runs the held-out calls once, at the end.
  Production code must never read benchmark gold, call IDs, expected values or scores to produce or choose a transcript.
- The recordings' content: never transcode, re-encode or overwrite audio with different content.
- Secrets: never put an AWS key, Supabase key or any token in a file, commit, log, prompt or chat. They live in git-ignored `.env` files.
- `infra/` (Terraform) and the fyp-ai publisher, dashboard, Supabase schema, policies and migrations. The owner deploys and applies migrations. If your idea needs one of these, describe the exact patch for the owner.

## Running and publishing

- Running `collect.py` on the development set is allowed (Amazon Transcribe is billable: a full 175-call run is roughly 1,042 audio minutes, about US$25). Use a descriptive new `--run-id` each time; never reuse an id; do not retry a failing run in a loop.
- Score offline with `score.py`, export with `export_to_fyp_ai.py`, and publish complete development runs with fyp-ai's `upload_stt_run.py` (see `STT_HANDOVER.md`). Published results are compared in Supabase/the dashboard, so no separate baseline file is needed.
  Publish only complete runs; a published run is immutable.
- Report, for every run: the one-line change made, the effective `TRANSCRIBE_...` settings, and WER, entity WER and speaker attribution overall **and per language** (including what got worse). Do not claim an improvement without a comparable measured run.
- New handler environment variables are not set for you in production: say which ones production would need, so the evaluation and production match.
- To score another STT model with the existing scorer, write its output as Transcribe-shaped JSON (`results.items[]` with `type`, `alternatives[0].content`/`confidence`, `speaker_label`, `start_time`/`end_time`, plus `results.transcripts`) into
  `evaluation-engine/runs/<new run id>/raw/transcripts/CALL_xxx.json` with a `metadata.json` like the one `collect.py` writes, then run `score.py`. If that does not fit, stop and ask: do not edit the scorer.

## Things to know about Amazon Transcribe (checked against the AWS documentation)

- Custom vocabulary works with multi-language identification only through `LanguageIdSettings` (one vocabulary per language) together with `LanguageOptions`. It must exist in `ap-southeast-1` before the job starts.
- Custom language models are **not** supported with multi-language identification, and `zh-CN` has no custom language model. A CLM can help English only, and only if the job stops using multi-language identification.
  Do not drop multi-language identification to try one without telling the owner: Mandarin and mixed calls depend on it.
- Singlish has no Transcribe locale and is transcribed as `en-US`. English is scored per word, Chinese per character.
- Beware overfitting: a vocabulary made of the exact reference words of the 175 development calls looks great and means nothing. Prefer general domain terms (instrument types, currencies, product-name patterns) and say where each term came from.

## Git

Work on a feature branch; do not push to main, rewrite history or change these rules to expand your permissions. Stage named files only (no `git add .`).
Never commit `.env`, `runs/`, audio, or anything under `evaluation-dataset/datasets/`. Do not add co-author or tool-attribution lines to commit messages.
If something is unclear or a rule blocks you, stop and ask the owner with the exact command, the error and what you want to change. Do not work around a rule.
