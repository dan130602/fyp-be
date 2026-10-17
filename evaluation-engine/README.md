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
| `transcribe_eval/normalize.py` | Versioned normalisation shared by reference and hypothesis |
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
```

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
- Contractions ("it'll" vs "it will") are not expanded.

## Scoring output (`runs/<run id>/scores/`)

`summary.json` (headline metrics), `calls.jsonl` (per call, with raw texts), `spans.jsonl` (each selected span with its errors),
`confusions.json` (top substitutions, deletions, insertions), `unnormalised.json` (forms left unchanged, with examples).
