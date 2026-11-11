> Historical experiment notes. See `stt-experiments/meralion/README.md` and its audited results for the current unchanged-weight recommendation and corrected model comparisons.

# Local Whisper CPU feasibility trial

Decision: retain Transcribe. Whisper base regressed on all four calls. This is a four-call development canary, not a 175-call result or a held-out evaluation.

## Overall results

| Model | WER | Entity WER | Speaker attribution |
|---|---:|---:|---:|
| Transcribe baseline | 1.72% (47/2731) | 1.44% (5/348) | 99.85% |
| Transcribe + vocabulary | 1.68% (46/2731) | 1.44% (5/348) | 99.85% |
| Whisper base CPU | 11.86% (324/2731) | 8.33% (29/348) | Unavailable (no diarization) |

## Per-language results (one call each)

| Model | Language / call | WER | Entity WER | Speaker attribution |
|---|---|---:|---:|---:|
| Transcribe baseline | Mandarin / CALL_059 | 0.68% | 2.59% | 100.00% |
| Transcribe baseline | Mixed / CALL_064 | 3.45% | 1.45% | 99.76% |
| Transcribe baseline | Singlish / CALL_094 | 1.81% | 0.00% | 100.00% |
| Transcribe baseline | English / CALL_147 | 0.92% | 1.08% | 99.61% |
| Transcribe + vocabulary | Mandarin / CALL_059 | 0.68% | 2.59% | 100.00% |
| Transcribe + vocabulary | Mixed / CALL_064 | 3.45% | 1.45% | 99.76% |
| Transcribe + vocabulary | Singlish / CALL_094 | 1.81% | 0.00% | 100.00% |
| Transcribe + vocabulary | English / CALL_147 | 0.73% | 1.08% | 99.61% |
| Whisper base CPU | Mandarin / CALL_059 | 4.68% | 15.52% | Unavailable |
| Whisper base CPU | Mixed / CALL_064 | 30.37% | 11.59% | Unavailable |
| Whisper base CPU | Singlish / CALL_094 | 8.66% | 2.86% | Unavailable |
| Whisper base CPU | English / CALL_147 | 1.10% | 1.08% | Unavailable |

## Method and limitations

- Same four audio files verified by matching audio hashes across all three runs; 12.84 audio minutes. CPU inference took 382.83 seconds (6.38 minutes), excluding installation/loading/scoring.
- Cached Systran/faster-whisper-base revision ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66; faster-whisper 1.2.1, CTranslate2 4.8.5, PyAV 16.0.1; CPU INT8, four threads, beam size 5, automatic language detection, word timestamps, VAD disabled. No initial prompt or reference-driven routing.
- Transcribe settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2. Baseline has no custom vocabulary; candidate uses the two READY canary vocabularies recorded in its metadata. No new production settings were added by this local trial.
- Existing evaluation scorer and normalizer 1.4.0 unchanged. Chinese is scored per character, English per word. Entity metric uses annotated values/events; domain terms have not been reviewed.
- Whisper detected zh for Mandarin and mixed, ms (Malay) for Singlish, and en for English. The mixed call had 229 errors versus 26 with Transcribe; Singlish had 48 versus 10. Language routing is a follow-up hypothesis, not a measured fix.
- Whisper has no speaker labels here. The unchanged scorer emits 0% for missing labels; that is an unsupported capability, not a valid diarization measurement. Reported as unavailable. Confidence omitted because Whisper probabilities are not calibrated comparably to Transcribe.
- Transcribe-shaped adapter preserves lexical content and word timing, separates punctuation, and invents no speaker/confidence fields. Checks covered punctuation, timing fields and absent speaker/confidence. No audio, reference, scorer, split or production code changed for this trial.
- This result concerns the small base model and these settings only; it does not establish the performance of larger Whisper variants or MERaLiON. No cloud jobs, API credentials or GPU rental were used.
- Initial attempt failed before transcription because PyAV 19 removed an argument expected by faster-whisper; pinning PyAV 16.0.1 resolved it. Dependencies are isolated in the workspace target directory.

## Next experiment

If continuing with this CPU model, evaluate audio-only language selection constrained to the supported English/Mandarin locales, with segment-level handling for code-switching. Do not select language from benchmark labels or force all calls to English. Treat this as a new experiment and retain Transcribe until comparable measurements improve both lexical and entity accuracy.

## Reproducibility

Local script: stt-experiments/local_whisper_trial.py (machine-specific paths recorded in the script). Raw model output and scores stay in git-ignored evaluation-engine/runs, not in this report.

Run identifiers:

- `stt-vocab-canary-baseline-20261107t171418z-d03fd238`
- `stt-vocab-canary-candidate-20261107t171418z-d03fd238`
- `whisper-base-cpu-20261107T173734Z`

CPU inference reference: https://github.com/SYSTRAN/faster-whisper
