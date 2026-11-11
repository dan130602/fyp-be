> Historical experiment notes. See `stt-experiments/meralion/README.md` and its audited results for the current unchanged-weight recommendation and corrected model comparisons.

# Original vocabulary validation and specialist-model research

## Original 11-term vocabulary: broader validation

This tests the original vocabulary that reduced errors from 47 to 46 on the initial four calls. It is distinct from the rejected 33-term expansion. Sixteen additional development calls were evaluated against the existing matched baseline; only the candidate required fresh AWS jobs (98.9 audio minutes).

| Language | Baseline WER | Original vocabulary WER | Baseline entity WER | Vocabulary entity WER | Baseline speaker | Vocabulary speaker |
|---|---:|---:|---:|---:|---:|---:|
| English | 0.90% | 0.90% | 1.83% | 1.83% | 99.17% | 99.17% |
| Mandarin | 1.31% | 1.33% | 2.46% | 2.46% | 99.90% | 99.90% |
| Mixed | 5.63% | 5.60% | 4.49% | 4.49% | 99.52% | 99.52% |
| Singlish | 1.76% | 1.83% | 2.13% | 2.13% | 99.96% | 99.96% |
| Overall | 1.92% | 1.93% | 2.66% | 2.66% | 99.58% | 99.58% |

Total errors: 405 to 408 out of 21140. Entity errors: 50 to 50 out of 1877.
The initial one-error gain did not generalize to this broader sample. Do not promote the vocabulary or spend on a full 175-call candidate run based on this evidence. This small difference does not establish statistical superiority or inferiority; it fails the practical improvement gate.

## Whisper small as an English-configured Singlish specialist

Four predeclared Singlish calls, compared with the same Transcribe calls. This is a specialist feasibility experiment, not an automatic routing evaluation. Whisper receives the same fixed English setting for every file; benchmark labels/answers are not read by its inference path. The cohort is known Singlish for evaluation, so results cannot establish how to identify Singlish in production.

| Model | WER | Entity WER | Speaker attribution |
|---|---:|---:|---:|
| Transcribe baseline | 1.76% | 2.13% | 99.96% |
| Whisper small, English setting | 3.16% | 6.38% | Unavailable |

| Call | Transcribe WER | Whisper WER | Transcribe entity WER | Whisper entity WER |
|---|---:|---:|---:|---:|
| CALL_146 | 1.29% | 3.74% | 2.33% | 9.30% |
| CALL_184 | 2.08% | 2.24% | 1.37% | 9.59% |
| CALL_198 | 2.42% | 3.88% | 1.83% | 5.49% |
| CALL_235 | 1.28% | 2.72% | 3.00% | 3.00% |

Whisper errors: 81/2563 versus Transcribe 45/2563; entity errors 27/423 versus 9/423.
Whisper does not establish a better Singlish specialist on this sample. Retain Transcribe.
CPU inference took 9.45 minutes for 15.12 audio minutes, excluding download/loading/scoring. No paid API or GPU was used for Whisper.
- Systran/faster-whisper-small revision 536b0662742c02347bc0e980a01041f333bce120; CPU INT8, four threads, beam size 5, language en, task transcribe, word timestamps enabled, VAD disabled; no prompt or vocabulary.
- Both model size and language setting differ from the earlier base-model experiment, which also used different calls. Do not attribute changes to model size alone or compare those cohort percentages directly.
- No speaker labels or calibrated confidence were invented. The unchanged scorer reports zero for absent speaker labels, but speaker attribution is unavailable for this standalone Whisper pipeline.
- Adapter checks covered decimals, punctuation, repeated words and absent speaker/confidence. Source audio hashes match the baseline and remain unchanged.

## Research findings

| Candidate | Why consider it | Status here |
|---|---|---|
| Transcribe | Existing multilingual and diarization integration; strongest measured default so far | Baseline and vocabulary experiments measured |
| Whisper small | 244M parameters versus base 74M; local CPU inference is feasible | Measured above; no automatic Singlish routing |
| Whisper large-v3 / turbo | Larger multilingual candidates with different speed/accuracy tradeoffs | Not measured; larger runtime budget and diarization integration needed |
| MERaLiON-3-3B-ASR | Authors explicitly target Singapore/regional speech and code-switching | Researched, not run; no verified runtime or API access available |
| Qwen3-ASR 0.6B / 1.7B | English/Chinese support and language identification; separate forced aligner provides timestamps | Researched, not run; needs its own runtime/adapter validation |

MERaLiON has distinct hosted API and open-weight offerings. The card describes self-hosting the open model on a single 80 GB GPU; that is not established here as a minimum memory requirement or proof CPU inference is impossible. Its published benchmark results are not comparable to these synthetic financial calls. The hosted API advertises timestamps/diarization; those capabilities should not be assumed for the open-weight text-only wrapper.

Qwen documents 30 languages and 22 Chinese dialects, with a separate forced-alignment model. This is a research candidate, not evidence of superior Singlish performance. The documented CUDA examples and FlashAttention build guidance are not a verified CPU benchmark for this laptop.

## What a valid multi-model design requires

1. Establish a specialist advantage on a development cohort, including entity/number errors, before building routing around it.
2. Evaluate proper English and Mandarin alongside Singlish and mixed calls. English language identification alone cannot distinguish proper English from Singlish; mixed calls may switch within an utterance.
3. Route using audio-derived evidence or independently supplied operational metadata, never benchmark labels, IDs or expected answers. Low-confidence or unsupported routes should use the validated default.
4. Measure the complete router-plus-recognizer pipeline on a separate development validation slice: route confusion matrix, WER, entity WER, speaker accuracy, latency and cost. A table that selects the best model using known dataset labels is an oracle comparison, not deployable accuracy.
5. Preserve speaker identity and timing across any fallback pass. Do not select the winning transcript using reference answers or blindly compare confidence numbers from different models.
6. Only after a candidate passes these checks, run all 175 development calls; the owner runs the held-out set and deploys. No full run or deployment is justified by the unsuccessful trials alone.

## Sources

- [OpenAI Whisper models and language behavior](https://github.com/openai/whisper)
- [Faster-whisper CPU inference](https://github.com/SYSTRAN/faster-whisper)
- [MERaLiON official model card](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR)
- [Qwen official model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

## Audit

Vocabulary settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2; TRANSCRIBE_VOCABULARIES=en-US=clearcall-canary-en-us-20261107t171418z-d03fd238,zh-CN=clearcall-canary-zh-cn-20261107t171418z-d03fd238. Actual AWS settings and audio hashes were verified. Baseline omits vocabulary. Normalizer/scorer and entity definitions unchanged. No held-out data, production deployment, publishing, commits or pushes. No persistent environment setting changed.

Run IDs:
- `stt-vocab-broader-baseline-20261107t175401z-bfb58069`
- `stt-vocab-original16-candidate-20261108t100155z-266e558b`
- `whisper-small-en-singlish-20261108T100344Z`
