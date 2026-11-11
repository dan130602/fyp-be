> Historical experiment notes. See `stt-experiments/meralion/README.md` and its audited results for the current unchanged-weight recommendation and corrected model comparisons.

# Development STT experiments

The 6 November dashboard screenshots show 175 completed development calls with no custom vocabulary and normaliser 1.4.0. Overall error rate was 3,468 / 213,182 (1.63%), entity error rate was 594 / 19,050 (3.12%), and speaker attribution was 164,370 / 164,852 eligible items (99.71%). Mean confidence was 0.9811. Language rows are transcribed from the screenshot.

| Language group | Calls | Error rate | Entity error rate |
|---|---:|---:|---:|
| English | 52 | 0.94% | 2.65% |
| Singlish | 35 | 1.40% | 2.58% |
| Mandarin | 53 | 1.47% | 3.16% |
| Mixed | 35 | 2.99% | 4.12% |

Mixed calls are the first group to inspect. Substitutions account for 2,154 of all errors (62.1%), followed by 921 deletions (26.6%) and 393 insertions (11.3%). The screenshots do not identify which words were affected. If the baseline score directory is available, review `scores/confusions.json` and per-call scores for repeated substitutions, omitted values and errors around language switches. Compare entity counts and denominators directly. The baseline entity score uses value mentions and meaning-critical spans. A vocabulary table is an input to recognition and does not change the scorer's entity definitions.

## First experiment: custom vocabulary with language switching

`vocab-en-US.txt` and `vocab-zh-CN.txt` are small candidate tables with 11 concepts each. Their concepts come from public Singapore financial education materials, with Simplified Chinese renderings drafted for review. Unit trusts and net asset value are from [unit trust guidance](https://www.moneysense.gov.sg/investments/understanding-unit-trusts/). Exchange-traded funds are from the [ETF guide](https://www.moneysense.gov.sg/guide-to-etfs-understanding-exchange-traded-funds/). Structured deposits and counterparty risk use the product and risk concepts in [structured deposit guidance](https://www.moneysense.gov.sg/investments/understanding-structured-deposits/) and [structured note guidance](https://www.moneysense.gov.sg/investments/understanding-structured-notes/). Structured notes, credit-linked notes, and specified investment products are covered in the [SIP guide](https://www.moneysense.gov.sg/investments/understanding-specified-investment-products/). Investment-linked policies are from the [investment product checklist](https://www.moneysense.gov.sg/investments/what-to-ask-when-buying-an-investment-product/). Treasury bills and real estate investment trusts are from the [investment types guide](https://www.moneysense.gov.sg/investments/types-of-investments/). No reference transcripts were used to choose the terms. The tables use four columns, tab delimiters and LF line endings as required by [AWS vocabulary table guidance](https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary-create-table.html).

After the resource owner reviews the terms and creates matching-language vocabularies in `ap-southeast-1`, use the following settings in the evaluation environment. Resource names below are examples. An owner would use the same settings when deploying the selected configuration.

```dotenv
TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN
TRANSCRIBE_LANGUAGE_ID_MODE=multiple
TRANSCRIBE_MAX_SPEAKER_LABELS=2
TRANSCRIBE_VOCABULARIES=en-US=clearcall-finance-en-v1,zh-CN=clearcall-finance-zh-v1
```

The Lambda sends each name in `LanguageIdSettings` and preserves both language options. The handler requires two to five matching vocabulary entries in identification mode, per the documented map size. For fixed-language jobs, provide a single matching entry and the handler sends it in `Settings.VocabularyName`. Unset `TRANSCRIBE_VOCABULARIES` to run the same configuration without vocabularies. See [StartTranscriptionJob](https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html).

When a billable run is authorised, compare the baseline and candidate on the same four development calls from the handover first, with distinct run IDs. Confirm both resources are applied and jobs complete. If the canary is sound, compare the full 175 with matching audio hashes, speaker settings, scorer version, and entity definitions. Report overall and per-language error rate, entity error rate, speaker attribution and eligible-item coverage. Inspect newly inserted vocabulary terms, missed amounts, currency and negation, and errors around language switches. Also record failures and elapsed time. Pick a candidate only after reviewing the mixed-call and Mandarin slices for regressions. Treat these as development scores.

## Separate English custom language model feasibility check

`TRANSCRIBE_LANGUAGE_MODEL_NAME` is an opt-in for fixed `en-US` jobs. It sets `ModelSettings.LanguageModelName` and refuses to start a job with automatic language identification or a different locale. An owner can compare fixed English with and without a completed CLM on the same English development calls, with the vocabulary setting identical across both runs. Train only on separately approved, representative text that excludes benchmark references. Keep English–Mandarin identification enabled for the multilingual workflow. AWS does not support CLMs with multi-language identification or for `zh-CN`. See [AWS language identification](https://docs.aws.amazon.com/transcribe/latest/dg/lang-id-batch.html) and [CLM language support](https://docs.aws.amazon.com/transcribe/latest/APIReference/API_CreateLanguageModel.html).

## Alternative engines

| Candidate | What to check first | Work needed for a comparable run |
|---|---|---|
| Whisper large-v3 or turbo | Use multilingual transcription and inspect language switches, repeated text and critical values. Time a few full calls on the target hardware. | Pin the model, add word timings and independently generated diarization, and export the existing item schema. See [Whisper](https://github.com/openai/whisper). |
| MERaLiON-3-3B-ASR | Test Singapore English, Singlish, Mandarin and within-call switching. Its published evaluations are on other corpora. | Verify the local runtime, long-audio chunking, alignment and speaker attribution before a full run. See the [MERaLiON model card](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR). |

Run alternatives outside the production Lambda first. Use the same original development audio. Generate timestamps and speaker labels from the candidate pipeline, without borrowing them from dataset references. Keep unavailable confidence fields unavailable and compare confidence only when its meaning is comparable. Write `results.transcripts` and `results.items` with the recognized text and its associated timing and speaker fields, and use the existing offline scorer. A GPU worker is a deployment option if the measured throughput, latency and per-call cost justify it. Confirm downstream compatibility with the existing job name and transcript output contract before any switch.

The local setup and mocked Lambda tests do not measure recognition accuracy. AWS resource registration, CLM training, paid inference, and alternate model benchmarking require a separate run.
