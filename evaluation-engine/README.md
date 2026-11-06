# Transcription evaluation engine

Evaluates the **Amazon Transcribe step of production** against the evaluation dataset's ground truth. Today it covers transcription only; the
folder is named for the engine so other stage evaluators can sit beside it later.

It does **not** reimplement the transcription job. `transcribe_eval/handler_loader.py` loads `lambda/transcribe-trigger/app/handler.py`
unchanged and calls `handler.handler(event, None)` with an S3 event the engine builds, so a change to the handler (for example adding a custom
vocabulary) is automatically what gets evaluated. Nothing under `lambda/`, `media-processing-service/` or `infra/` is modified.

## Layout

| Path | Purpose |
|---|---|
| `transcribe_eval/config.py` | Settings from this folder's `.env` and the environment |
| `transcribe_eval/handler_loader.py` | Loads the production handler unchanged |
| `transcribe_eval/events.py` | The S3 event and the per-run object-key convention |
| `scripts/check_aws.py` | Read-only connection check (identity, bucket, Transcribe access) |
| `transcribe_eval/collect.py`, `scripts/collect.py` | Collect phase: audio to S3, jobs via the handler, output to `runs/<run id>/` |
| `transcribe_eval/normalize.py`, `transcribe_eval/data/` | Versioned normalisation shared by reference and hypothesis; its pinned word lists (see `data/SOURCES.md`) |
| `transcribe_eval/reference.py`, `hypothesis.py` | Reference turns and entity spans; Transcribe items |
| `transcribe_eval/score.py`, `scripts/score.py` | Score phase (offline) |
| `scripts/term_candidates.py`, `domain_terms.json` | Domain-term candidates for review; the reviewed list |
| `tests/` | Offline tests; the real handler is driven with the AWS client faked |
| `runs/` (git-ignored) | Run output, written by the collect/score phases |

## Setup

1. `pip install -r requirements-dev.txt` (Python 3.11+).
2. Copy `.env.example` to `.env` in **this** folder and fill it in. The outer `fyp-be/.env` is never read (docker-compose uses it).
3. `python -B scripts/check_aws.py` until every line says `ok`.
4. `python -m pytest tests`.

## AWS resources and permissions

One private bucket in `ap-southeast-1` (Singapore, production's region), **not a production bucket and with no S3 trigger** (a trigger would also start
the real Lambda). Audio lives under `eval-audio/` and `uploads/`, Transcribe's output under `transcripts/`.

The AWS user needs `s3:PutObject/GetObject/DeleteObject/ListBucket` on that bucket and
`transcribe:StartTranscriptionJob/GetTranscriptionJob/DeleteTranscriptionJob/ListTranscriptionJobs`.

## Bucket layout (one bucket, prefixes)

One bucket is enough; input and output are separated by prefix. Every local audio file is called `audio.mp3` (the call folder identifies it), so the S3 file
name carries the call id instead. The local files are never renamed (their hashes are recorded by the dataset's render manifests).

```
eval-audio/                                  kept master audio ("input")
  legacy_pilot_v1/                           one folder per dataset (5 calls)
    clean_voice_A/                           one folder per render (voice/noise variant)
      CALL_001.mp3 ... CALL_005.mp3
  advisory_execution_pilot_v1/               the 50-call dataset
    clean_voice_A/
      CALL_001.mp3 ... CALL_050.mp3
uploads/eval-<run id>/<dataset id>/CALL_xxx.mp3   this run's copy handed to the handler (temporary; the run id keeps job names unique)
transcripts/<job name>.json                  Transcribe's output; the handler fixes this location
```

Each run downloads its outputs to `runs/<run id>/raw/` locally, with the audio's sha256 recorded in the run metadata, so a re-rendered call is detected.
Separate input and output buckets would only be worth it for different retention or permissions.

## Rules

- Each run uses its own keys (see the table above): the handler derives the Transcribe job name from the key and a job name can only be used once.
  The master audio stays under `eval-audio/` and is copied inside S3 per run.
- Poll jobs with `get_transcription_job`; never read production's SQS status queue (it would take messages the media-processing service needs).
- Never commit `.env`, never print credentials, never point the engine at a production bucket.
- Record with every run: the handler's git commit, the handler environment, the `get_transcription_job` response, and any vocabulary or language model
  (name and last-modified time), so a score change can be traced to a setting.

## Commands

```
python -B scripts/check_aws.py                                        # connection check, read-only
python -B scripts/collect.py --dataset legacy_pilot_v1 --all --dry-run  # plan only, no AWS
python -B scripts/collect.py --dataset legacy_pilot_v1 --all            # billable Transcribe jobs -> runs/<run id>/
python -B scripts/score.py --run <run id>                               # offline scoring -> runs/<run id>/scores/
python -B scripts/term_candidates.py --dataset advisory_execution_pilot_v1   # candidates for YOUR domain-term review
python -B scripts/compare_scores.py --run <run id>                      # what a normaliser change did (needs runs/<id>/scores_1.2.0/), writes changes_from_1.2.0.json
python -B scripts/export_to_fyp_ai.py --run <run id>                       # local bundle for the fyp-ai Speech-to-text tab (view only)
```

**Changing the normaliser or a scoring rule** (offline, free; no Transcribe or Polly calls): copy `runs/<id>/scores` to `runs/<id>/scores_<old version>`
(and the exported bundle somewhere safe) BEFORE re-scoring; run `score.py`, then `compare_scores.py`, then `export_to_fyp_ai.py --overwrite`.
`compare_scores.py` first proves that scoring with every scoring-only rule switched off reproduces the old scores exactly, then attributes each
removed error to a rule by switching rules off one at a time, and lists the errors that remain and any that newly appeared.

## Metric definitions

Every metric comes from ONE alignment per call (jiwer `process_words`): the whole normalised reference dialogue against all normalised
Transcribe pronunciation items. Corpus figures are summed counts `(S+D+I)/N` across calls, never an average of per-call rates.

- **Normalised WER** (the only WER): `(substitutions + deletions + insertions) / reference words`, after the normalisation below.
  Raw reference and Transcribe text are kept per call in `scores/calls.jsonl` for inspection.
- **Entity WER (project-specific definition).** The selected reference words are the union of
  (a) every annotated value mention (`authoring/check_plans` `value_mentions`, placed in the turn by the dialogue's `event_alignment`),
  including repeated, historical, corrected, distractor and scenario (fee, limit price) values;
  (b) the whole aligned span of every critical event (`VALUE_CORRECTION`, `ORDER_INVALIDATION`, `READBACK`, `FALSE_START`) plus the
  spans of events it links to (`corrects_event_id`, `invalidates_event_id`, `refers_to_event_id`, followed transitively);
  (c) occurrences of terms from `domain_terms.json`, used only once that list is marked `reviewed`.
  Overlapping spans are merged so each reference word and its error count once. S and D are the alignment operations on selected
  words; an insertion counts only when anchored strictly inside a run of selected words (insertions at a boundary are not counted).
  Entity WER = errors / selected reference words. A per-group breakdown (value question, event kind, term) is diagnostic only.
  Datasets without these annotations (the legacy 5-call pilot) report entity WER as not available.
- **S/D/I counts** are reported for both.
- **Average confidence:** mean over every original AWS pronunciation item, each counted once, zero-confidence words included
  (punctuation items are not words). Also reported: the mean for items that aligned correctly and for the rest.
- **Speaker attribution:** counted per original AWS pronunciation item. An item is eligible when every normalised token it produced is
  an alignment hit, and correct when its speaker label, mapped to a role, equals the reference speaker of all those tokens. The
  label-to-role mapping is chosen per call to maximise agreement. Accuracy = correct / eligible.

## Normalisation (version in every score file)

Rules change how something is written, never what was said; they run identically and independently on reference and Transcribe text
and never look at expected values. Ambiguous forms are left unchanged and listed in `scores/unnormalised.json`. Full rules and the
change log are in `transcribe_eval/normalize.py`. Key points:

- Numbers are recognised before punctuation is stripped: decimals (`1.2` = "one point two"), signs, leading zeros (`00123` != `123`)
  and valid thousands separators survive. Spelled numbers, multipliers ("1.2 million"), `500k`, ordinals, an explicit year rule
  ("nineteen ninety-eight" = 1998), "double/triple" digits and "oh" between digits are converted.
- Every number is written one digit per token so a spelled identifier ("S six five ...") matches Transcribe's digits. Consequences:
  numbers carry more weight in WER (a 7-digit identifier = 7 tokens), and a cardinal and its digit reading ("sixty five" / "six five")
  become identical. Confidence and speaker attribution count AWS items, so they are not affected.
- A bare currency symbol on a number reads as the number plus its word ("€200" = "200 euros"; Transcribe writes the symbol for
  the spoken word). Only one-word symbols are mapped (€ $ £ ₹); "¥" and "S$" stay as symbol tokens.
- **Scoring-only equivalences (1.3.0).** The same thing written two ways is not a recognition error. These rules change only the tokens used
  for alignment (raw text is never modified, and nothing here changes what any other system receives); each can be switched off by name
  (`rules=` in `normalize_words` / `score_call` / `score_run`):
  - `dates`: a valid date, day-first or month-first, becomes one order ("4 August 1991" = "August 4th, 1991"); the day must exist in the
    month; "may" and "march" count as months only with an ordinal, an "of" or a year (so a bare "13 May" is left as written).
  - `spelling`: British -> American pairs from OpenAI Whisper's pinned `english.json` (data only; no Whisper code or model, MIT, attributed
    in `data/SOURCES.md`, sha256-checked at import) minus reviewed exclusions.
  - `contractions`: `'ll 're 've 'm`, auxiliary + `n't`, `won't`, `can't`; negation is kept ("won't" is never "will"); `'s` and `'d` are left.
  - `variants`: the reviewed list in `data/spelling_variants.json` (OK/okay, cut-off/cutoff); ignored unless `reviewed` is true.
  - `currency`: in "$50,000 in SGD" the explicit currency already says it, so the "$" is not an extra word. Closed table: "$" agrees with
    SGD / USD / dollar(s) / Singapore or US dollars, "€" with EUR / euro(s). A conflict ("$30,000 in EUR"), a bare "$200" and a bare number
    keep the 1.2.0 behaviour; a conflict is also reported as `currency_conflict`.
  Multi-word matches stay inside one reference turn or one speaker's words and never cross a sentence end; a comma may sit inside a date.
  Applied rules are counted per side in `summary.json` (`scoring_rules_applied`) and tagged on each display unit (`scoring_rules`); the
  rule data and its hashes are in `summary.json` (`normalizer_rules`). Side effect: contractions now count as two reference words.

## Scoring output (`runs/<run id>/scores/`)

`summary.json` (headline metrics), `calls.jsonl` (per call, with raw texts), `units/CALL_xxx.json` (display units: the scoring alignment grouped per reference
word, with token-level parts so merging never hides an error), `spans.jsonl` (each selected span with its errors),
`confusions.json` (top substitutions, deletions, insertions), `unnormalised.json` (forms left unchanged, with examples).

## Known limitations and future recommendations (recorded, not implemented in this phase)

- **Legacy dates.** The legacy calls were rendered with the en-US client voice (Matthew), which says "4 August 1991" as "August fourth ...". The
  legacy reference states the right fact but not the audio's word order. The `dates` rule scores around this; it does not change the reference.
  *Future:* a render hint so day-month-year dates are spoken in the written order.
- **Legacy pronunciation.** The legacy audio was rendered by `scripts/render_polly.py` with plain SSML, so "EUR" and "NRIC" were read as words
  (recognised as "your", "RC"). These errors stay visible on purpose. The advisory renderer's pronunciation hints (`rendering/ssml.py`) do not
  have this defect. *Future:* render only with the hinted renderer, or re-render legacy if wanted.
- **"$" before amounts.** Transcribe (en-US) adds "$" to amounts. It is ignored for scoring only when an explicit agreeing currency follows. A bare
  "$30,000 a year" still counts, because the text alone cannot show whether "dollars" was spoken.
- **Not changed here:** production transcript normalisation, Polly voices and SSML, the Transcribe language code (an en-GB experiment is a
  separate comparison on the same audio), reference transcripts, and spoken-to-written canonicalisation.

## Multilingual datasets (Mandarin, mixed, Singlish)

- **Datasets.** The 250-call release is split by `splits/split.v1.json` into `datasets/advisory_execution_multilingual_v1_development` (175) and `..._heldout` (75), each with
  `calls/`, `authoring/{check_plans,dialogues}` (entity WER needs them) and its own `dataset_manifest.json`. Tune on development; run the held-out set once at the end.
- **Collect.** `python -B scripts/collect.py --dataset <dataset> --render-id auto --all` (or `--split DEVELOPMENT`/`HELDOUT` on the full release). `--render-id auto` uses each call's
  own render, mp3 (Polly) or wav (Gemini, OmniVoice). Billable.
- **Language.** The production handler (`lambda/transcribe-trigger`) identifies the language when `TRANSCRIBE_LANGUAGE_OPTIONS` is set (here `en-US,zh-CN`, mode `multiple`);
  Transcribe has no Singapore English, so Singlish is transcribed as `en-US`. Unset, the handler behaves as before (one `TRANSCRIBE_LANGUAGE_CODE`).
- **Scoring (normaliser 1.4.0).** Every Han character is one scoring unit and English stays per word (`cjk.py`); Chinese numerals equal their digits (`zh_numbers` rule: 七万五千 = 7万5千,
  百分之零点二五 = 0.25%); Chinese 。、，end a phrase. English and Singlish scores are identical to 1.3.0. Re-scoring is offline: `python -B scripts/score.py --run <id>`.
  Not applied, pending review: Chinese variant spellings such as 账/帐 (the same word) are still counted as errors.
