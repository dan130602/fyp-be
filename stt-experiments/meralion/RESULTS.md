# Completed MERaLiON refinement and model comparison

All three fresh segmentation runs completed 175 development calls. Original energy took 77.3 GPU inference minutes, guarded 30 s took 77.1 minutes and guarded 24 s took 78.1 minutes. Base checkpoint hashes passed before and after inference. No LoRA adapter, training, model merging, audio conversion or held-out calls were used.

## Fresh 175-call experiments

| Configuration | WER | Entity WER | Word substitutions | Word deletions | Word insertions |
|---|---:|---:|---:|---:|---:|
| Original energy raw | 1.74% (3706/213182) | 4.35% (829/19050) | 2349 | 842 | 515 |
| Original energy + fee | 1.72% (3667/213182) | 4.15% (790/19050) | 2323 | 829 | 515 |
| Guarded 30 s raw | 2.06% (4382/213182) | 4.26% (812/19050) | 2659 | 900 | 823 |
| Guarded 30 s + fee | 2.04% (4343/213182) | 4.06% (773/19050) | 2633 | 887 | 823 |
| Guarded 24 s raw | 1.83% (3906/213182) | 4.36% (831/19050) | 2155 | 1268 | 483 |
| Guarded 24 s + fee | 1.81% (3867/213182) | 4.16% (792/19050) | 2129 | 1255 | 483 |

Fee means the same standalone half-percent formatting applied separately to each hypothesis. It does not change the evaluator. Compact monetary expansion and contextual prompting remain excluded.

## Per-language comparison with identical fee formatting

| Language | Calls | Original WER / entity WER | Guarded 30 s WER / entity WER | Guarded 24 s WER / entity WER |
|---|---:|---:|---:|---:|
| English | 52 | 1.87% / 3.34% | 2.06% / 3.73% | 1.55% / 2.71% |
| Mandarin | 53 | 1.22% / 4.14% | 1.17% / 4.05% | 1.18% / 3.79% |
| Mixed | 35 | 2.76% / 5.13% | 4.14% / 4.54% | 3.69% / 7.12% |
| Singlish | 35 | 1.32% / 4.30% | 1.34% / 4.02% | 1.33% / 3.77% |

## What changed relative to original energy + fee

| Candidate | Extra word errors | Extra entity errors | WER difference, percentage points | Entity WER difference, percentage points |
|---|---:|---:|---:|---:|
| Guarded 30 s + fee | +676 | -17 | +0.317 | -0.089 |
| Guarded 24 s + fee | +200 | +2 | +0.094 | +0.010 |

Guarded 30 s has a modest pooled entity gain but a substantial WER regression. English worsens on both metrics. Its mixed-language word errors increase from 1195 to 1790, accounting for 595 of the 676 extra word errors. Mixed insertions increase from 176 to 434; this identifies additional inserted text, not a proven root cause. Mandarin improves slightly, but that is insufficient to adopt the configuration globally.

Guarded 24 s improves English by 177 word errors and 32 entity errors, and Mandarin by 27 word errors and 24 entity errors. It also reduces Singlish entity errors by 17, while adding five Singlish word errors. However, mixed calls add 399 word errors and 75 entity errors. Mixed deletions rise from 436 to 935. Across the whole dataset it therefore adds 200 word errors and two entity errors. This is a useful candidate for targeted investigation, not a safe global replacement or a validated language router.

## Earlier 175-call segmentation results

| Configuration | WER | Entity WER |
|---|---:|---:|
| AWS Transcribe, published dashboard | 1.63% (3468/213182) | 3.12% (594/19050) |
| Fixed | 2.03% (4335/213182) | 5.17% (985/19050) |
| Silero | 1.96% (4180/213182) | 4.43% (843/19050) |
| WebRTC + fee | 1.82% (3889/213182) | 4.49% (855/19050) |
| Midpoint + fee | 1.86% (3959/213182) | 4.18% (796/19050) |

## Other models on the same eight-call pilot

These are two calls per language, not 175-call results. Hosted and local MERaLiON are distinct inference offerings; their exact checkpoint equivalence is unverified. Local MERaLiON includes fee formatting; the other systems use their provider transcripts.

| Model | WER | Entity WER | Speaker attribution |
|---|---:|---:|---:|
| Local MERaLiON energy + fee | 2.44% (151/6185) | 3.24% (31/956) | Unavailable |
| Amazon Transcribe | 2.52% (156/6185) | 2.82% (27/956) | 99.66% (4382/4397) |
| Deepgram Nova-3 | 12.50% (773/6185) | 7.95% (76/956) | 95.16% (4204/4418) |
| Hosted MERaLiON API | 0.91% (56/6185) | 1.88% (18/956) | Unavailable |
| Qwen hosted | 1.46% (90/6185) | 3.56% (34/956) | 99.66% (2898/2908) |
| Fun-ASR Flash | 1.10% (68/6185) | 2.30% (22/956) | Unavailable |
| Whisper small | 8.88% (549/6185) | 10.46% (100/956) | Unavailable |
| Transcribe + 11-term vocabulary | 2.54% (157/6185) | 2.82% (27/956) | 99.66% (4383/4398) |

## Decision

Keep original energy pauses plus conservative fee formatting as the local default. Do not adopt either guarded candidate globally on these results. Retain AWS as the stronger full-set reference, particularly for entity recognition. The full AWS figures are published dashboard values, not an independently re-scored native 175-call output.

Hosted MERaLiON and Fun-ASR warrant broader development testing based on the eight-call pilot, but the pilot does not establish full-set superiority. Qwen was strongest on the two Mandarin pilot calls. Routing to whichever model wins a benchmark language class would not establish a deployable routing policy.

The most useful next unchanged-weight task is targeted review of original-energy entity substitutions and deletions, with audio verification before changing order types, consent, negation or monetary values. Further segmentation sweeps should be small and predeclared, with both WER and entity WER as acceptance criteria. Do not select transcripts using gold labels.

## Scoring checks

All three runs used the unchanged project CLI and normalizer 1.4.0 with identical dates, spelling, contractions, variants, currency and Chinese-number rules. Adapter preservation tests passed for currency, signs and percentages. Scorer fingerprints match across runs. Both reference denominators are unchanged: 213182 word/character tokens and 19050 entity tokens. Speaker attribution, word confidence and word timestamps are unavailable for the text-only local MERaLiON pipeline. The earlier LoRA experiment changes adapter weights and uses a different inner-validation cohort; it is excluded from the unchanged-weight ranking.
