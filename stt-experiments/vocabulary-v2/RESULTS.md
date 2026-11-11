> Historical experiment notes. See `stt-experiments/meralion/README.md` and its audited results for the current unchanged-weight recommendation and corrected model comparisons.

# Broader development vocabulary trial

Expanded each language vocabulary from 11 to 33 finance terms and tested a frozen, stratified random sample of 16 additional development calls (four per language). Same audio hashes, handler and scoring rules across both runs. No benchmark text used to choose vocabulary terms.

## Decision

DO NOT PROMOTE: the candidate failed the predeclared gate. Do not spend on a 175-call candidate run or deploy this vocabulary.

## Results

| Language | Calls | Baseline WER | Candidate WER | Baseline entity WER | Candidate entity WER | Baseline speaker | Candidate speaker |
|---|---:|---:|---:|---:|---:|---:|---:|
| ENGLISH | 4 | 0.90% | 1.00% | 1.83% | 1.83% | 99.17% | 99.17% |
| MANDARIN | 4 | 1.31% | 1.32% | 2.46% | 2.46% | 99.90% | 99.90% |
| EN_ZH_MIXED | 4 | 5.63% | 5.66% | 4.49% | 4.49% | 99.52% | 99.52% |
| SINGLISH | 4 | 1.76% | 1.79% | 2.13% | 2.13% | 99.96% | 99.96% |
| ALL | 16 | 1.92% | 1.96% | 2.66% | 2.66% | 99.58% | 99.57% |

## Error counts

| Language | Baseline S / D / I | Candidate S / D / I | WER errors change | Entity errors change |
|---|---:|---:|---:|---:|
| ENGLISH | 31 / 18 / 8 | 34 / 21 / 8 | +6 | +0 |
| MANDARIN | 95 / 10 / 12 | 94 / 12 / 12 | +1 | +0 |
| EN_ZH_MIXED | 86 / 95 / 5 | 86 / 96 / 5 | +1 | +0 |
| SINGLISH | 28 / 8 / 9 | 29 / 8 / 9 | +1 | +0 |
| ALL | 240 / 131 / 34 | 243 / 137 / 34 | +9 | +0 |

Positive change means more errors. Rates use pooled error counts, not averages of call percentages.

## Paired call changes

| Call | Language | Baseline errors | Candidate errors | Entity errors baseline / candidate |
|---|---|---:|---:|---:|
| CALL_071 | ENGLISH | 9 | 9 | 4 / 4 |
| CALL_090 | ENGLISH | 40 | 46 | 1 / 1 |
| CALL_091 | ENGLISH | 2 | 2 | 0 / 0 |
| CALL_139 | ENGLISH | 6 | 6 | 2 / 2 |
| CALL_003 | MANDARIN | 21 | 21 | 6 / 6 |
| CALL_123 | MANDARIN | 63 | 62 | 1 / 1 |
| CALL_177 | MANDARIN | 17 | 19 | 2 / 2 |
| CALL_244 | MANDARIN | 16 | 16 | 8 / 8 |
| CALL_001 | EN_ZH_MIXED | 49 | 51 | 7 / 7 |
| CALL_035 | EN_ZH_MIXED | 41 | 40 | 3 / 3 |
| CALL_041 | EN_ZH_MIXED | 55 | 55 | 0 / 0 |
| CALL_209 | EN_ZH_MIXED | 41 | 41 | 7 / 7 |
| CALL_146 | SINGLISH | 9 | 10 | 2 / 2 |
| CALL_184 | SINGLISH | 13 | 13 | 1 / 1 |
| CALL_198 | SINGLISH | 15 | 15 | 3 / 3 |
| CALL_235 | SINGLISH | 8 | 8 | 3 / 3 |

## Promotion gates

- Overall WER improves: FAIL
- Overall entity errors do not increase: PASS
- No language WER or entity regression: FAIL
- Speaker attribution drops no more than 0.2 percentage points: PASS

## Setup and evidence

- 98.9 audio minutes per configuration; 32 jobs total, approximately 197.8 billable audio minutes. Billing amount is not verified.
- Seed 20261108. Excluded the earlier four-call canary. Selection used language metadata only for stratification, never for inference routing. The remaining development calls and held-out data were not used for this experiment.
- Effective settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2. Baseline unsets TRANSCRIBE_VOCABULARIES. Candidate uses en-US=clearcall-v2-en-us-20261107t175401z-bfb58069,zh-CN=clearcall-v2-zh-cn-20261107t175401z-bfb58069. No custom language model.
- The runner verifies actual AWS job parameters and paired audio hashes. Normalizer 1.4.0, unchanged entity definitions, no reviewed domain-term list. Speaker rates use eligible pronunciation items; denominators can differ between model outputs.
- 46 production handler tests passed. Vocabulary tables, term provenance and frozen protocol are under stt-experiments/vocabulary-v2. Raw outputs remain under git-ignored evaluation-engine/runs.
- No production deployment, dataset/scorer edits, held-out evaluation, publishing, commit or push. New AWS vocabulary resources and normal evaluation outputs remain available for audit.

## Error-analysis context

Only the earlier four-call detailed runs were available locally; the 175-call screenshot is aggregate evidence, not a downloadable confusion inventory. Earlier errors include language-switch omissions, Chinese homophones and names. Those cannot be reliably solved by adding a list of benchmark answers. The wider sample above provides fresh evidence about how far generic domain vocabulary helps.

## Source provenance

Added 22 concepts per language from [MoneySense bond guidance](https://www.moneysense.gov.sg/understanding-bonds/), [MoneySense ETF comparison guidance](https://www.moneysense.gov.sg/investments/choose-etfs-same-index/), and [SGX trading guidance](https://www.sgx.com/securities/trading). Chinese renderings were drafted for this experiment; they are not claimed to be official translations. The original 11 concepts retain their existing provenance in stt-experiments/README.md.

Run IDs:

- `stt-vocab-broader-baseline-20261107t175401z-bfb58069`
- `stt-vocab-broader-candidate-20261107t175401z-bfb58069`

## Diagnosis and next decision

The broader baseline contains 240 substitutions, 131 deletions and 34 insertions. Mixed-language calls account for 95 of the 131 deletions (72.5%); the candidate increases mixed-language deletions to 96. This is the clearest priority for the next architectural experiment.

Paired model-output inspection found English word-boundary regressions, a two-character Mandarin omission, and an English word changing grammatical form. Capitalization and multiword item packaging also changed, but those formatting changes alone do not explain the scored regression. Two calls improved by one error each, four worsened, and ten were unchanged. Entity-error counts were unchanged on every call.

Reject the 33-term candidate. The earlier 11-term candidate has only a one-error gain on four calls, so it is also not validated for production. Retain the current deployed configuration; the opt-in vocabulary code remains available for future controlled experiments.

The next research direction is audio-only language-switch handling: use timing and language evidence from audio/model output to identify omitted or uncertain spans, then evaluate a segment-level transcription candidate against this frozen baseline. Never use expected text or benchmark labels to route inference. This needs a separately implemented and tested segmentation/reassembly pipeline; it is not implemented or claimed as an improvement here. A full 175-call run and owner deployment remain conditional on a candidate passing the development gate.

The runner supplied settings only to its own process. No evaluation-engine/.env exists in this checkout, so the rejected vocabulary was not saved as a persistent default. Final git diff checks passed, with no tracked scorer or dataset modifications.
